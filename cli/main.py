#!/usr/bin/env python3
"""CLI для morag: индексация документов и поиск."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from morag.config import Config, DenseEmbedderConfig, PdfConfig, SparseEmbedderConfig, load_config
from morag.indexing.bm25 import build_bm25_index
from morag.indexing.chunker import (
    HybridChunker,
    LLMChunker,
    PassthroughChunker,
    SectionChunker,
    SemanticChunker,
)
from morag.indexing.context import LLMContextGenerator, NoopContextGenerator
from morag.indexing.embedder import (
    Embedder,
    HttpEmbedder,
    HttpGteSparseEmbedder,
    SparseEmbedder,
)
from morag.indexing.pipeline import IndexingPipeline
from morag.indexing.status_reporter import (
    FileStatusReporter,
    NullStatusReporter,
    StatusReporter,
)
from morag.indexing.processors import (
    DenseEmbeddingProcessor,
    DocSummaryProcessor,
    DocTitleProcessor,
    DocVectorProcessor,
    LegalDocSummaryProcessor,
    MetadataProcessor,
    PageMarkerProcessor,
    SparseEmbeddingProcessor,
)
from morag.indexing.token_counter import HuggingFaceTokenCounter, TiktokenCounter
from morag.llm.client import GenerationParams, LLMClient
from morag.llm.retry import RetryPolicy
from morag.sources.confluence import ConfluenceSource
from morag.sources.confluence_pdf import ConfluencePdfSource
from morag.sources.jira import JiraSource
from morag.sources.jira_extractor import JiraLinkExtractor
from morag.sources.local import LocalDocumentSource
from morag.sources.pdf_converter import DoclingPdfConverter, PdfConverter, VisionPdfConverter
from morag.sources.pdf_postprocess import CodeFencePostProcessor, DeduplicatePostProcessor, PdfPostProcessor
from morag.storage.collections import (
    ensure_chunks_collection,
    ensure_docs_collection,
    frida_vectors_config,
    gte_sparse_vectors_config,
)
from morag.storage.repository import ChunkRepository, DocRepository

def _build_postprocessors(pdf_config: PdfConfig) -> list[PdfPostProcessor]:
    """Собрать цепочку постпроцессоров по конфигу."""
    processors: list[PdfPostProcessor] = []
    if pdf_config.postprocessing.strip_code_fences:
        processors.append(CodeFencePostProcessor())
    if pdf_config.postprocessing.dedup.enabled:
        processors.append(DeduplicatePostProcessor(
            threshold=pdf_config.postprocessing.dedup.threshold,
            window=pdf_config.postprocessing.dedup.window,
            min_phrase_len=pdf_config.postprocessing.dedup.min_phrase_len,
        ))
    return processors


def _make_dense_embedder(cfg: DenseEmbedderConfig) -> Embedder:
    if cfg.base_url is None:
        raise ValueError(
            'dense_embedder.base_url is required. Native in-process embedder removed, '
            'use HTTP (Ollama / vLLM / OpenAI-compatible) endpoint.',
        )
    return HttpEmbedder(
        cfg.base_url, cfg.model, cfg.dim,
        document_template=cfg.document_template,
        query_template=cfg.query_template,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        max_rpm=cfg.max_rpm,
        max_concurrent=cfg.max_concurrent,
    )


def _make_sparse_embedder(cfg: SparseEmbedderConfig) -> SparseEmbedder:
    if cfg.base_url is None:
        raise ValueError(
            'sparse_embedder.base_url is required. Native in-process embedder removed, '
            'run services/embedder_gte/ (Docker) or any OpenAI-compatible endpoint.',
        )
    return HttpGteSparseEmbedder(
        cfg.base_url, cfg.timeout,
        retry_policy=RetryPolicy(max_retries=cfg.max_retries),
        max_rpm=cfg.max_rpm,
    )


def _make_pdf_converter(
    config: Config,
    vision_client: LLMClient | None,
) -> PdfConverter | None:
    """Создать PDF-конвертер по конфигу."""
    if config.pdf is None:
        return None

    mode = config.pdf.mode
    if mode == 'vision':
        if vision_client is None:
            logger.error('pdf.mode=vision requires llm_vision to be configured')
            return None
        gen_params = GenerationParams(
            temperature=config.pdf.temperature,
            repetition_penalty=config.pdf.repetition_penalty,
            frequency_penalty=config.pdf.frequency_penalty,
            presence_penalty=config.pdf.presence_penalty,
        )
        postprocessors = _build_postprocessors(config.pdf)
        return VisionPdfConverter(
            vision_client=vision_client,
            max_tokens=config.pdf.page_max_tokens,
            dpi=config.pdf.dpi,
            concurrency=config.pdf.concurrency,
            generation_params=gen_params,
            context_tail_lines=config.pdf.context_tail_lines,
            postprocessors=postprocessors,
        )
    elif mode == 'docling':
        return DoclingPdfConverter(
            docling_base_url=config.pdf.docling.base_url,
            docling_timeout=config.pdf.docling.timeout,
            vision_client=vision_client,
            vision_max_tokens=config.indexing.vision_max_tokens,
        )
    else:
        logger.error('Unknown pdf.mode: %s (expected "docling" or "vision")', mode)
        return None


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# Module-level Pydantic-модели для control-plane (FastAPI ругается если nested
# в функцию — не распознаёт как body, парсит как query).
class _StartReq(BaseModel):
    reset: bool = False


class _StopReq(BaseModel):
    grace_seconds: int = 180  # = control_plane.DEFAULT_STOP_GRACE_SECONDS, дублируем чтобы не цикл импортов


def _make_status_reporter() -> StatusReporter:
    """FileStatusReporter если выставлена env MORAG_STATUS_FILE, иначе Null."""
    path = os.environ.get('MORAG_STATUS_FILE')
    if path:
        logger.info('Status reporter: file=%s', path)
        return FileStatusReporter(path)
    return NullStatusReporter()


def _install_signal_handlers(cancel_event: asyncio.Event) -> None:
    """SIGTERM/SIGINT → graceful (set cancel_event). Второй сигнал → force exit.

    На Windows (где add_signal_handler не поддерживается) — no-op с предупреждением.
    """
    loop = asyncio.get_running_loop()

    def _handle(sig_name: str) -> None:
        if cancel_event.is_set():
            logger.warning('Second signal (%s) — forcing immediate exit', sig_name)
            raise KeyboardInterrupt
        logger.info('Signal %s received — graceful shutdown after current document', sig_name)
        cancel_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig.name)
        except NotImplementedError:
            logger.warning('Signal handler for %s not supported on this platform', sig.name)


async def cmd_index(
    config_path: str,
    reset: bool = False,
    cancel_event: asyncio.Event | None = None,
    status_reporter: StatusReporter | None = None,
) -> None:
    """Индексировать документы из источника в Qdrant.

    cancel_event и status_reporter — опциональные. Если None (CLI-standalone) —
    создаются внутри (с signal handler'ами). Если переданы (control-plane через
    cmd_serve) — используются как есть, signal handler'ы НЕ ставятся (родитель
    их уже поставил).
    """
    config = load_config(config_path)
    if config.indexing is None:
        raise ValueError(f'{config_path}: секция `indexing` обязательна для команды `index`')

    if cancel_event is None:
        cancel_event = asyncio.Event()
        _install_signal_handlers(cancel_event)
    reporter = status_reporter if status_reporter is not None else _make_status_reporter()

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(
            host=config.qdrant.host, port=config.qdrant.port,
            timeout=60, pool_size=max(10, config.indexing.concurrency * 4),
        )

    if reset:
        existing = {c.name for c in (await client.get_collections()).collections}
        for name in (config.qdrant.collection_docs, config.qdrant.collection_chunks):
            if name in existing:
                logger.warning('Dropping collection: %s', name)
                await client.delete_collection(name)

    embedder = _make_dense_embedder(config.indexing.dense_embedder)

    logger.info('Ensuring collections...')
    await ensure_docs_collection(
        client, config.qdrant.collection_docs,
        vectors_config=frida_vectors_config(embedder.dim),
        sparse_vectors_config=gte_sparse_vectors_config(),
    )
    await ensure_chunks_collection(
        client, config.qdrant.collection_chunks,
        vectors_config=frida_vectors_config(embedder.dim),
        sparse_vectors_config=gte_sparse_vectors_config(),
    )

    doc_repo = DocRepository(client, config.qdrant.collection_docs)
    chunk_repo = ChunkRepository(client, config.qdrant.collection_chunks)

    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        max_concurrent=config.llm.max_concurrent,
        model_wait_seconds=config.llm.model_wait_seconds,
        model_wait_retries=config.llm.model_wait_retries,
        enable_thinking=config.llm.enable_thinking,
        context_window=config.llm.context_window,
    )

    vision_client = None
    if config.llm_vision:
        vision_client = LLMClient(
            base_url=config.llm_vision.base_url,
            model=config.llm_vision.model,
            api_key=config.llm_vision.api_key,
            timeout=config.llm_vision.timeout,
            max_retries=config.llm_vision.max_retries,
            max_concurrent=config.llm_vision.max_concurrent,
            model_wait_seconds=config.llm_vision.model_wait_seconds,
            model_wait_retries=config.llm_vision.model_wait_retries,
            enable_thinking=config.llm_vision.enable_thinking,
            context_window=config.llm_vision.context_window,
        )
        logger.info('Vision LLM: %s @ %s', config.llm_vision.model, config.llm_vision.base_url)

    # Фабрика PDF-конвертера
    pdf_converter = _make_pdf_converter(config, vision_client)

    local_source = None
    if config.sources.local_documents:
        local_source = LocalDocumentSource(
            root=config.sources.local_documents.path,
            pdf_converter=pdf_converter,
        )
        pdf_mode = config.pdf.mode if config.pdf else 'disabled'
        logger.info('Source: local_documents path=%s (pdf=%s)', config.sources.local_documents.path, pdf_mode)

    sources = []
    if config.sources.confluence:
        sources.append(ConfluenceSource(
            config.sources.confluence,
            vision_client=vision_client,
            vision_max_tokens=config.indexing.vision_max_tokens,
        ))
        logger.info('Source: confluence url=%s (vision=%s)', config.sources.confluence.url, vision_client is not None)
    if not sources and local_source is None:
        logger.error('No sources configured in config.yml')
        return

    llm_counter = TiktokenCounter()  # для LLM context window, doc_summary, doc_title
    embed_tokenizer = (
        config.indexing.dense_embedder.tokenizer
        or config.indexing.dense_embedder.model
    )
    # 'tiktoken' — спец-значение: использовать TikToken вместо HF-токенайзера.
    # Подходит для моделей с большим max_tokens (Qwen3, 1024+) где точность ±40% терпима
    # и позволяет избежать HF-зависимости при индексации.
    if embed_tokenizer == 'tiktoken':
        embed_counter = TiktokenCounter()
        logger.info('Token counters: llm=TikToken, embed=TikToken')
    else:
        embed_counter = HuggingFaceTokenCounter(embed_tokenizer)
        logger.info('Token counters: llm=TikToken, embed=HuggingFace(%s)', embed_tokenizer)
    chunker_mode = config.indexing.chunker.mode
    if chunker_mode == 'semantic':
        chunker = SemanticChunker(
            embed_fn=embedder.embed_batch,
            counter=embed_counter,  # FRIDA tokenizer для точного подсчёта
            min_tokens=config.indexing.chunker.min_tokens,
            max_tokens=config.indexing.chunker.max_tokens,
            accept_pair=config.indexing.chunker.accept_pair,
        )
    elif chunker_mode == 'llm':
        chunker = LLMChunker(
            llm_client,
            token_counter=llm_counter,
            halving_retries=config.indexing.chunker.halving_retries,
            fallback_enabled=config.indexing.chunker.fallback,
        )
    elif chunker_mode in ('hybrid', 'section'):
        oversized_cfg = config.indexing.chunker.oversized
        oversized_strategies = {
            'table': oversized_cfg.table,
            'list': oversized_cfg.list,
            'paragraph': oversized_cfg.paragraph,
            'fence': oversized_cfg.fence,
            'diagram': oversized_cfg.diagram,
        }
        # LLM chunker нужен если хотя бы одна стратегия = llm
        llm_chunker_for_hybrid = None
        if 'llm' in oversized_strategies.values():
            llm_chunker_for_hybrid = LLMChunker(
                llm_client,
                token_counter=llm_counter,
                halving_retries=config.indexing.chunker.halving_retries,
                fallback_enabled=config.indexing.chunker.fallback,
            )
        # Embed fn нужен если хотя бы одна стратегия = embed
        embed_fn = embedder.embed_batch if 'embed' in oversized_strategies.values() else None
        chunker_cls = SectionChunker if chunker_mode == 'section' else HybridChunker
        chunker = chunker_cls(
            counter=embed_counter,
            min_tokens=config.indexing.chunker.min_tokens,
            max_tokens=config.indexing.chunker.max_tokens,
            oversized_strategies=oversized_strategies,
            embed_fn=embed_fn,
            llm_chunker=llm_chunker_for_hybrid,
        )
    else:
        chunker = PassthroughChunker()
    context_generator = (
        LLMContextGenerator(
            llm_client,
            token_counter=llm_counter,
            embed_counter=embed_counter,
            max_output_tokens=config.indexing.context.max_tokens,
            window_tokens=config.indexing.context.window_tokens,
            chunk_max_tokens=config.indexing.context.chunk_max_tokens,
        ) if config.indexing.context.mode == 'llm' else NoopContextGenerator()
    )
    sparse_embedder = _make_sparse_embedder(config.indexing.sparse_embedder)

    doc_processors = []
    if config.indexing.doc_title.max_tokens is not None:
        doc_processors.append(DocTitleProcessor(
            llm_client=llm_client,
            max_tokens=config.indexing.doc_title.max_tokens,
            scan_tokens=config.indexing.doc_title.scan_tokens,
            scan_pages=config.indexing.doc_title.scan_pages,
            token_counter=llm_counter,
        ))
    if config.indexing.doc_summary.max_tokens is not None:
        summary_classes = {
            'default': DocSummaryProcessor,
            'legal': LegalDocSummaryProcessor,
        }
        summary_mode = config.indexing.doc_summary.mode
        summary_cls = summary_classes.get(summary_mode)
        if summary_cls is None:
            raise ValueError(f'Unknown doc_summary mode: {summary_mode!r}')
        doc_processors.append(summary_cls(
            llm_client=llm_client,
            doc_repo=doc_repo,
            max_tokens=config.indexing.doc_summary.max_tokens,
            token_counter=llm_counter,
        ))
    doc_processors.append(DocVectorProcessor(
        dense_embedder=embedder,
        sparse_embedder=sparse_embedder,
        token_counter=embed_counter,
        max_tokens=config.indexing.doc_vector.max_tokens,
    ))

    chunk_processors = [
        PageMarkerProcessor(),
        MetadataProcessor(),
        DenseEmbeddingProcessor(embedder),
        SparseEmbeddingProcessor(
            sparse_embedder,
            include_doc_summary=config.indexing.lexical_doc_summary,
            include_chunk_context=config.indexing.lexical_chunk_context,
        ),
    ]

    # В LLM-режиме блок + ответ LLM должны влезть в контекстное окно.
    # Ответ ≈ такого же размера как вход, поэтому безопасный лимит: (context_window - overhead) / 2.
    _LLM_PROMPT_OVERHEAD = 512  # токенов на системный промпт + запас
    skip_presplit = chunker_mode in ('semantic', 'hybrid', 'section')
    if chunker_mode == 'llm':
        llm_safe_limit = (config.llm.context_window - _LLM_PROMPT_OVERHEAD) // 2
        block_limit = min(config.indexing.chunker.block_limit, llm_safe_limit)
        if block_limit < config.indexing.chunker.block_limit:
            logger.info(
                'LLM block limit capped: %d → %d (context_window=%d, overhead=%d)',
                config.indexing.chunker.block_limit, block_limit,
                config.llm.context_window, _LLM_PROMPT_OVERHEAD,
            )
    else:
        block_limit = config.indexing.chunker.block_limit

    pipeline = IndexingPipeline(
        doc_repo, chunk_repo,
        doc_processors=doc_processors,
        chunker=chunker,
        context_generator=context_generator,
        chunk_processors=chunk_processors,
        block_limit=block_limit,
        token_counter=llm_counter,
        concurrency=config.indexing.concurrency,
        skip_presplit=skip_presplit,
        passthrough_threshold=config.indexing.chunker.passthrough_threshold,
        embed_batch_size=config.indexing.embed_batch_size,
        max_table_rows=config.indexing.chunker.max_table_rows,
        status_reporter=reporter,
        cancel_event=cancel_event,
    )

    logger.info(
        'Chunker: %s, context: %s%s',
        chunker_mode, config.indexing.context.mode,
        f', block_limit: {block_limit}' if not skip_presplit else '',
    )

    try:
        # Локальные документы: композитный source с собственным порядком запуска
        if local_source is not None and not cancel_event.is_set():
            await local_source.run(pipeline)

        # Остальные source (Confluence и т.д.)
        for source in sources:
            if cancel_event.is_set():
                break
            await pipeline.run(source)

        # Confluence PDF attachments: после страниц, чтобы parent pages уже были в базе
        if (not cancel_event.is_set()
                and config.sources.confluence
                and config.sources.confluence.attachments.enabled):
            if pdf_converter is not None:
                confluence_pdf_source = ConfluencePdfSource(
                    config=config.sources.confluence,
                    converter=pdf_converter,
                    doc_repo=doc_repo,
                )
                logger.info(
                    'Source: confluence_pdf (mime_types=%s)',
                    config.sources.confluence.attachments.mime_types,
                )
                await pipeline.run(confluence_pdf_source)
            else:
                logger.warning(
                    'Confluence attachments enabled but pdf is not configured — skipping'
                )

        # Jira: сканируем проиндексированные документы на ссылки, затем индексируем задачи.
        # Удаление устаревших задач происходит каскадно через parent_doc_ids при удалении родительских документов.
        if not cancel_event.is_set() and config.sources.jira:
            logger.info('Source: jira url=%s', config.sources.jira.url)
            all_docs = await doc_repo.scroll_all(exclude_source_types=['attached_jira'])
            logger.info('Scanning %d indexed document(s) for Jira links...', len(all_docs))
            extractor = JiraLinkExtractor(config.sources.jira.url)
            issue_map = extractor.extract_from_docs(all_docs)
            if issue_map:
                logger.info('Found %d Jira issue(s) in indexed documents', len(issue_map))
                # Строим parent_ids_map: {issue_key: [doc_id, ...]} для хранения parent_doc_ids
                path_to_doc_id = {doc.path[0]: doc.id for doc in all_docs if doc.path}
                parent_ids_map: dict[str, list[str]] = {
                    key: [path_to_doc_id[p] for p in paths if p in path_to_doc_id]
                    for key, paths in issue_map.items()
                }
                jira_source = JiraSource(config.sources.jira, issue_map, parent_ids_map)
                await pipeline.run(jira_source)
            else:
                logger.info('No Jira issues found in indexed documents, skipping Jira indexing')

        # Post-indexing: upgrade sparse vectors schema + build BM25 для chunks и docs
        if not cancel_event.is_set():
            reporter.start_phase('post_indexing_bm25', 3)
            from morag.storage.collections import upgrade_sparse_vectors
            await upgrade_sparse_vectors(client, config.qdrant.collection_chunks)
            reporter.document_done('upgrade_sparse_chunks')
            await build_bm25_index(
                client, config.qdrant.collection_chunks,
                include_doc_summary=config.indexing.lexical_doc_summary,
                include_chunk_context=config.indexing.lexical_chunk_context,
            )
            reporter.document_done('bm25_chunks')
            await build_bm25_index(client, config.qdrant.collection_docs)
            reporter.document_done('bm25_docs')

        # Post-indexing: Knowledge Map
        if not cancel_event.is_set() and config.indexing.knowledge_map.enabled:
            reporter.start_phase('knowledge_map', 1)
            from morag.indexing.knowledge_map import KnowledgeMapGenerator
            km_cfg = config.indexing.knowledge_map
            km_generator = KnowledgeMapGenerator(
                client=client,
                llm_client=llm_client,
                doc_repo=doc_repo,
                collection=km_cfg.collection,
                depth=km_cfg.depth,
                max_depth=km_cfg.max_depth,
                node_max_tokens=km_cfg.node_max_tokens,
                node_min_tokens=km_cfg.node_min_tokens,
                prompt_strategy=km_cfg.prompt_strategy,
                prompt_budget=km_cfg.prompt_budget,
                token_counter=llm_counter,
                concurrency=config.indexing.concurrency,
                exclude_source_types=km_cfg.exclude_source_types,
                depth1_section_ids=km_cfg.depth1_section_ids,
                flat_topics_target=km_cfg.flat_topics_target,
                flat_topics_max_input_docs=km_cfg.flat_topics_max_input_docs,
                flat_topics_assign_batch=km_cfg.flat_topics_assign_batch,
            )
            root_ids = set()
            if config.sources.confluence:
                root_ids = set(config.sources.confluence.ancestor_ids)
            logger.info('Generating Knowledge Map (roots=%s)...', root_ids or 'auto')
            await km_generator.generate(root_ids=root_ids or None)
            reporter.document_done('knowledge_map')

        if cancel_event.is_set():
            reporter.finish('cancelled')
        else:
            reporter.finish('completed')
    except Exception as e:
        logger.exception('Indexing failed')
        reporter.finish('failed', error=str(e))
        raise
    finally:
        await client.close()


async def cmd_rebuild_km(
    config_path: str,
    cancel_event: asyncio.Event | None = None,  # noqa: ARG001 — для совместимости с control-plane
    status_reporter: StatusReporter | None = None,
) -> None:
    """Перестроить только Knowledge Map из существующих документов в Qdrant.

    cancel_event пока не используется — KM-генерация быстрая (минуты), без
    fine-grained точек прерывания. Принимается для совместимости с control-plane.
    """
    config = load_config(config_path)
    if config.indexing is None:
        raise ValueError(f'{config_path}: секция `indexing` обязательна для команды `rebuild-km`')

    if not config.indexing.knowledge_map.enabled:
        logger.error('knowledge_map.enabled is false in config — nothing to rebuild')
        return

    reporter = status_reporter if status_reporter is not None else _make_status_reporter()
    reporter.start_phase('knowledge_map_rebuild', 1)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(
        host=config.qdrant.host, port=config.qdrant.port,
        timeout=60,
    )

    doc_repo = DocRepository(client, config.qdrant.collection_docs)
    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        max_concurrent=config.llm.max_concurrent,
        model_wait_seconds=config.llm.model_wait_seconds,
        model_wait_retries=config.llm.model_wait_retries,
        enable_thinking=config.llm.enable_thinking,
        context_window=config.llm.context_window,
    )
    llm_counter = TiktokenCounter()

    from morag.indexing.knowledge_map import KnowledgeMapGenerator
    km_cfg = config.indexing.knowledge_map
    km_generator = KnowledgeMapGenerator(
        client=client,
        llm_client=llm_client,
        doc_repo=doc_repo,
        collection=km_cfg.collection,
        depth=km_cfg.depth,
        max_depth=km_cfg.max_depth,
        node_max_tokens=km_cfg.node_max_tokens,
        node_min_tokens=km_cfg.node_min_tokens,
        prompt_strategy=km_cfg.prompt_strategy,
        prompt_budget=km_cfg.prompt_budget,
        token_counter=llm_counter,
        concurrency=config.indexing.concurrency,
        exclude_source_types=km_cfg.exclude_source_types,
        flat_topics_target=km_cfg.flat_topics_target,
        flat_topics_max_input_docs=km_cfg.flat_topics_max_input_docs,
        flat_topics_assign_batch=km_cfg.flat_topics_assign_batch,
        depth1_section_ids=km_cfg.depth1_section_ids,
    )
    root_ids = set()
    if config.sources.confluence:
        root_ids = set(config.sources.confluence.ancestor_ids)
    logger.info('Rebuilding Knowledge Map (roots=%s)...', root_ids or 'auto')
    try:
        await km_generator.generate(root_ids=root_ids or None)
        reporter.document_done('knowledge_map')
        reporter.finish('completed')
    except Exception as e:
        reporter.finish('failed', error=str(e))
        raise
    finally:
        await client.close()

    logger.info('Knowledge Map rebuild complete')


async def cmd_serve(config_path: str) -> None:
    """Daemon-режим: индексация по cron-расписанию.

    Initial run при старте НЕ запускается (раньше запускался — это ломало
    docker compose up на свежей установке когда config ещё не настроен через
    console). Первый прогон делается либо вручную через console UI («Start»),
    либо когда сработает cron.

    Если schedule не задан — daemon живёт в idle, ждёт изменения конфига
    (требует ручного рестарта контейнера). Container не падает в restart-loop.
    """
    config = load_config(config_path)
    if config.indexing is None:
        logger.error(f'{config_path}: секция `indexing` обязательна для команды `serve`')
        sys.exit(1)

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    from morag.control_plane import (
        AlreadyRunning,
        IndexerControlPlane,
    )

    # state-file для status_reporter и публикации прогресса наружу
    status_path = os.environ.get(
        'MORAG_STATUS_FILE',
        '/data/morag_state/index_status.json',
    )
    control_port = int(os.environ.get('MORAG_CONTROL_PORT', '9090'))

    control_plane = IndexerControlPlane(
        config_path=config_path,
        status_file_path=status_path,
        run_index=lambda **kw: cmd_index(config_path, **kw),
        run_rebuild_km=lambda **kw: cmd_rebuild_km(config_path, **kw),
    )

    # ---- Cron scheduler (если настроен) ----
    scheduler: AsyncIOScheduler | None = None
    if config.indexing.schedule:
        async def cron_trigger() -> None:
            try:
                await control_plane.start_index(reset=False)
                logger.info('Cron-triggered indexing started')
            except AlreadyRunning:
                logger.warning('Cron-trigger skipped: indexing already running')
            except Exception:
                logger.exception('Cron-trigger failed')

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            cron_trigger,
            CronTrigger.from_crontab(config.indexing.schedule),
            max_instances=1,
            coalesce=True,
            misfire_grace_time=None,
        )
        scheduler.start()
        logger.info('Cron schedule active: %s', config.indexing.schedule)
    else:
        logger.info(
            'No cron schedule. Indexing only on-demand via control-plane (HTTP :%d)',
            control_port,
        )

    # ---- HTTP control-plane ----
    from fastapi import FastAPI, HTTPException
    import uvicorn

    app = FastAPI(title='morag-indexer control-plane', version='0.1.0')

    @app.get('/control/status')
    async def _status():
        return control_plane.status()

    @app.post('/control/start')
    async def _start(req: _StartReq):
        try:
            info = await control_plane.start_index(reset=req.reset)
            return {'started_at': info.started_at, 'kind': info.kind, 'reset': info.reset}
        except AlreadyRunning as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.post('/control/stop')
    async def _stop(req: _StopReq):
        result = await control_plane.stop(grace_seconds=req.grace_seconds)
        return {'result': result}

    @app.post('/control/kill')
    async def _kill():
        result = await control_plane.kill()
        return {'result': result}

    @app.post('/control/rebuild-km')
    async def _rebuild_km():
        try:
            info = await control_plane.start_rebuild_km()
            return {'started_at': info.started_at, 'kind': info.kind}
        except AlreadyRunning as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    server_config = uvicorn.Config(
        app, host='0.0.0.0', port=control_port,
        log_level='info', access_log=False,
    )
    server = uvicorn.Server(server_config)

    logger.info('Serve mode started. Control-plane HTTP on :%d', control_port)

    try:
        await server.serve()
    finally:
        if scheduler is not None:
            scheduler.shutdown()
            logger.info('Scheduler stopped')


async def cmd_query(config_path: str, question: str, top_k: int) -> None:
    """Гибридный поиск по вопросу без LLM-ответа (для отладки)."""
    config = load_config(config_path)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(
            host=config.qdrant.host, port=config.qdrant.port,
            timeout=60, pool_size=max(10, config.indexing.concurrency * 4),
        )

    embedder = _make_dense_embedder(config.indexing.dense_embedder)
    sparse_embedder = _make_sparse_embedder(config.indexing.sparse_embedder)

    dense_vec = embedder.embed_query(question)
    sparse_indices, sparse_values = sparse_embedder.embed_query(question)

    from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

    results = await client.query_points(
        collection_name=config.qdrant.collection_chunks,
        prefetch=[
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using='keywords',
                limit=top_k * 2,
            ),
            Prefetch(
                query=dense_vec,
                using='full',
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    print(f'\n=== Результаты поиска ({len(results.points)} чанков) ===\n')
    for i, point in enumerate(results.points):
        payload = point.payload or {}
        print(f'[{i+1}] score={point.score:.4f}  path={payload.get("path", "?")}  order={payload.get("order", "?")}')
        print(f'     creator={payload.get("creator", "-")}  updated_at={payload.get("updated_at", "?")}')
        text = payload.get('text', '')
        print(f'     {text[:200].replace(chr(10), " ")}')
        print()

    await client.close()


try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version('morag')
except Exception:
    _VERSION = 'unknown'

"""
  ░▒▓█████
 ░▒▓███████         Catonmoon
▒▓██(=^.^=)██       Morag v{_VERSION}
 ▓█████████
  ▓███████
"""

LOGO = f"""
    ▄▀▀▀▀▀▀▀▀▄
   █  /\\_/\\   █      Catonmoon
   █ ( =^.^=) █      ╔╦╗ ╔═╗ ┬─┐ ┌─┐ ┌─┐
   █  /> < /  █      ║║║ ║ ║ ├┬┘ ├─┤ │ ┬
    ▀▄▄▄▄▄▄▄▄▀       ╩ ╩ ╚═╝ ┴└─ ┴ ┴ └─┘
                     Indexer      v{_VERSION}
"""


def main() -> None:
    print(LOGO)
    parser = argparse.ArgumentParser(
        description='morag — RAG для локальных MD-файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'morag {_VERSION}')
    parser.add_argument(
        '-v', '--debug', action='store_true',
        help='Включить DEBUG-логирование (показывает сырые ответы LLM)',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)
    index_parser = subparsers.add_parser('index', help='Индексировать документы из источника')
    index_parser.add_argument(
        '--config', default='config.yml', metavar='PATH',
        help='Путь к конфигу (по умолчанию: config.yml)',
    )
    index_parser.add_argument(
        '--reset', action='store_true',
        help='Удалить все коллекции Qdrant перед индексацией (полная переиндексация)',
    )

    km_parser = subparsers.add_parser('rebuild-km', help='Перестроить Knowledge Map без полной индексации')
    km_parser.add_argument(
        '--config', default='config.yml', metavar='PATH',
        help='Путь к конфигу (по умолчанию: config.yml)',
    )

    serve_parser = subparsers.add_parser('serve', help='Daemon-режим: индексация по расписанию из конфига')
    serve_parser.add_argument(
        '--config', default='config.yml', metavar='PATH',
        help='Путь к конфигу (по умолчанию: config.yml)',
    )

    query_parser = subparsers.add_parser('query', help='Гибридный поиск без LLM-ответа (для отладки)')
    query_parser.add_argument(
        '--config', default='config.yml', metavar='PATH',
        help='Путь к конфигу (по умолчанию: config.yml)',
    )
    query_parser.add_argument('question', help='Поисковый вопрос')
    query_parser.add_argument('--top-k', type=int, default=10, help='Количество результатов (по умолчанию: 10)')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger('morag').setLevel(logging.DEBUG)

    if args.command == 'index':
        asyncio.run(cmd_index(args.config, reset=args.reset))
    elif args.command == 'rebuild-km':
        asyncio.run(cmd_rebuild_km(args.config))
    elif args.command == 'serve':
        asyncio.run(cmd_serve(args.config))
    elif args.command == 'query':
        asyncio.run(cmd_query(args.config, args.question, args.top_k))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
