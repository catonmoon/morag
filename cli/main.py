#!/usr/bin/env python3
"""CLI для morag: индексация документов и поиск."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from morag.config import Config, DenseEmbedderConfig, PdfConfig, SparseEmbedderConfig, load_config
from morag.indexing.bm25 import build_bm25_index
from morag.indexing.chunker import (
    BoundaryChunker,
    HybridChunker,
    LLMChunker,
    PassthroughChunker,
    SectionChunker,
    SemanticChunker,
    TranscriptChunker,
)
from morag.indexing.context import LLMContextGenerator, NoopContextGenerator
from morag.indexing.embedder import (
    Embedder,
    HttpEmbedder,
    HttpGteSparseEmbedder,
    SparseEmbedder,
)
from morag.indexing.embedder_fingerprint import compute_embedder_fingerprint
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
    TimestampProcessor,
)
from morag.indexing.token_counter import HuggingFaceTokenCounter, TiktokenCounter
from morag.llm.client import GenerationParams, LLMClient
from morag.llm.retry import RetryPolicy
from morag.run_context import RunContext
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
        api_key=cfg.api_key or 'ollama',
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
        max_concurrent=cfg.max_concurrent,
    )


def _build_llm_clients(llms: list) -> dict[str, LLMClient]:
    """Билд pool LLMClient'ов из config.llms.

    Один LLMClient на каждый name. Multimodal LLM (capabilities=[text,vision])
    — один клиент шарится между indexing.llm и indexing.vision если оба
    указывают на него.
    """
    clients: dict[str, LLMClient] = {}
    for inst in llms:
        clients[inst.name] = LLMClient(
            base_url=inst.base_url,
            model=inst.model,
            api_key=inst.api_key,
            timeout=inst.timeout,
            max_retries=inst.max_retries,
            max_concurrent=inst.max_concurrent,
            model_wait_seconds=inst.model_wait_seconds,
            model_wait_retries=inst.model_wait_retries,
            enable_thinking=inst.enable_thinking,
            context_window=inst.context_window,
        )
    return clients


def build_source_for_instance(
    config: Config,
    kind: str,
    name: str,
    llm_clients: dict[str, LLMClient] | None = None,
):
    """Построить Source для ОДНОГО инстанса `(kind, name)` из конфига — для одиночного
    `load_one` (live-fetch read_pages, Фаза 2). Без оркестрации `cmd_index`: jira-карты
    (`issue_map`/`parent_ids_map`) нужны только для link-extraction при полной индексации,
    для `load_one(key)` — пусты. Confluence `vision_client` берётся из `llm_clients`
    (опционален: None = без описания картинок). Возвращает Source или None если инстанса
    нет в конфиге / kind не поддержан."""
    src_cfg = next(
        (s for s in config.sources if s.kind == kind and s.name == name), None
    )
    if src_cfg is None:
        return None
    idx = config.indexing
    if kind == 'confluence':
        vision = (
            llm_clients.get(idx.vision)
            if (idx and llm_clients and idx.vision) else None
        )
        return ConfluenceSource(
            src_cfg,
            vision_client=vision,
            vision_max_tokens=idx.vision_max_tokens if idx else None,
        )
    if kind == 'jira':
        return JiraSource(src_cfg, {}, {})
    if kind == 'local':
        return LocalDocumentSource(root=src_cfg.path, pdf_converter=None, name=src_cfg.name)
    logger.warning('build_source_for_instance: неподдержанный kind=%s', kind)
    return None


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
            enable_thinking=False,  # vision-индексация без CoT (см. инцидент 2026-05)
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
    # None → берётся config.indexing.stop_grace_seconds (передан control_plane'у при инициализации).
    grace_seconds: int | None = None


class _ReindexReq(BaseModel):
    scope: str = 'all'  # 'all' или имя источника (config.sources[].name)


class _FetchPageReq(BaseModel):
    kind: str            # 'confluence' | 'jira' | 'local'
    name: str            # имя инстанса источника (config.sources[].name)
    external_id: str     # внешний ID страницы (Confluence pageId / Jira issue key)


def _make_status_reporter() -> StatusReporter:
    """FileStatusReporter если выставлена env MORAG_STATUS_FILE, иначе Null."""
    path = os.environ.get('MORAG_STATUS_FILE')
    if path:
        logger.info('Status reporter: file=%s', path)
        return FileStatusReporter(path)
    return NullStatusReporter()


def _reindex_effort_path() -> Path:
    """Путь к маркеру реиндекс-эффорта (ADR-0014).

    Присутствие файла ⟺ незавершённый эффорт; повторный запуск его резюмит.
    """
    env = os.environ.get('MORAG_REINDEX_EFFORT_FILE')
    if env:
        return Path(env)
    status = os.environ.get('MORAG_STATUS_FILE')
    base = Path(status).parent if status else Path('/app/conf/state')
    return base / 'reindex_effort.json'


def _read_reindex_effort() -> dict | None:
    """Прочитать маркер эффорта; None если файла нет / повреждён."""
    path = _reindex_effort_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_reindex_effort(floor: int, scope: str) -> None:
    """Записать маркер эффорта (atomic tmp+rename)."""
    path = _reindex_effort_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps({'floor': floor, 'scope': scope}, indent=2))
    os.replace(tmp, path)


def _clear_reindex_effort() -> None:
    """Снять маркер — эффорт завершён."""
    _reindex_effort_path().unlink(missing_ok=True)


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
    reindex_scope: str | None = None,
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
    # Pydantic-схема ослаблена (для baseline config.yml без секретов), но индексация
    # требует все компоненты. setup_gate проверяет это в control-plane; здесь —
    # повторная защита от прямого CLI вызова без полной настройки.
    missing = []
    if not config.sources:
        missing.append('sources')
    if not config.llms:
        missing.append('llms')
    if config.indexing.dense_embedder is None:
        missing.append('indexing.dense_embedder')
    if config.indexing.llm is None:
        missing.append('indexing.llm')
    if config.indexing.vision is None:
        missing.append('indexing.vision')
    if missing:
        raise ValueError(
            f'{config_path}: для индексации не настроены: {", ".join(missing)}. '
            'Настройте через Console UI (http://localhost:8000) или вручную в config.local.yml.'
        )

    if cancel_event is None:
        cancel_event = asyncio.Event()
        _install_signal_handlers(cancel_event)
    reporter = status_reporter if status_reporter is not None else _make_status_reporter()

    # RunContext: бампит global counter, замораживает indexed_at для всех точек прогона.
    # См. ADR-0012, секция 5.
    run_ctx = RunContext.begin()
    logger.info('Run context: number=%d, indexed_at=%s', run_ctx.run_number, run_ctx.indexed_at)

    # Реиндекс-эффорт (ADR-0014): reindex_scope != None → плавно пересобрать
    # scope. effort_floor — run_number, ниже которого документы считаются
    # необработанными эффортом. Маркер на диске → следующий запуск резюмит
    # незавершённый эффорт (краш/cancel), пропуская уже сделанное.
    effort_floor: int | None = None
    if reindex_scope is not None:
        marker = _read_reindex_effort()
        if marker is not None:
            effort_floor = int(marker['floor'])
            reindex_scope = marker['scope']
            logger.info('Resuming reindex effort: floor=%d scope=%s',
                        effort_floor, reindex_scope)
        else:
            effort_floor = run_ctx.run_number
            _write_reindex_effort(floor=effort_floor, scope=reindex_scope)
            logger.info('Reindex effort started: floor=%d scope=%s',
                        effort_floor, reindex_scope)

    def _in_scope(name: str) -> bool:
        """Реиндекс-эффорт ограничен одним источником (или 'all' = весь корпус)."""
        return reindex_scope in (None, 'all') or name == reindex_scope

    # Fingerprint dense embedder'а — детект stale-векторов при смене модели.
    embedder_fingerprint = compute_embedder_fingerprint(
        config.indexing.dense_embedder.model,
        config.indexing.dense_embedder.dim,
        config.indexing.dense_embedder.base_url,
    )
    logger.info('Embedder fingerprint: %s', embedder_fingerprint)

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

    # ---- LLM pool (см. ADR-0012) ----
    # Один LLMClient на каждый name; multimodal-инстанс шарится между
    # indexing.llm и indexing.vision если оба указывают на него.
    llm_clients = _build_llm_clients(config.llms)
    logger.info('LLM pool: %s', sorted(llm_clients))

    # Резолв ролей через config.indexing.llm.name_for(role) + indexing.vision
    role_mapping = config.indexing.llm
    vision_client = llm_clients[config.indexing.vision]
    vision_inst = config.llm_by_name(config.indexing.vision)
    logger.info('Vision LLM: %s @ %s (capabilities=%s)',
                vision_inst.model, vision_inst.base_url, vision_inst.capabilities)

    # PDF converter использует vision_client как и раньше
    pdf_converter = _make_pdf_converter(config, vision_client)

    # ---- Sources (registry-based loop из config.sources) ----
    enabled_sources = [s for s in config.sources if s.enabled]
    if not enabled_sources:
        logger.error('No enabled sources in config.yml')
        return
    logger.info('Sources: %s', [(s.kind, s.name) for s in enabled_sources])

    llm_counter = TiktokenCounter()  # для LLM context window, doc_summary, doc_title
    embed_tokenizer = (
        config.indexing.dense_embedder.tokenizer
        or config.indexing.dense_embedder.model
    )
    # 'tiktoken' — спец-значение: использовать TikToken вместо HF-токенайзера.
    # Подходит для моделей с большим max_tokens (Qwen3, 1024+) где точность ±40% терпима
    # и позволяет избежать HF-зависимости при индексации.
    # Также fallback на tiktoken если model не выглядит как HF repo id
    # (например 'qwen3-embedding:4b' — Ollama-нотация: AutoTokenizer бы упал).
    looks_like_hf_id = '/' in embed_tokenizer and ':' not in embed_tokenizer
    if embed_tokenizer == 'tiktoken' or not looks_like_hf_id:
        embed_counter = TiktokenCounter()
        if embed_tokenizer != 'tiktoken':
            logger.warning(
                'tokenizer %r не похож на HuggingFace repo id (формат org/name); '
                'использую TikToken-counter (приближение). Задайте explicit '
                'indexing.dense_embedder.tokenizer для точного подсчёта.',
                embed_tokenizer,
            )
        logger.info('Token counters: llm=TikToken, embed=TikToken')
    else:
        embed_counter = HuggingFaceTokenCounter(embed_tokenizer)
        logger.info('Token counters: llm=TikToken, embed=HuggingFace(%s)', embed_tokenizer)
    chunker_mode = config.indexing.chunker.mode
    if chunker_mode == 'semantic':
        chunker = SemanticChunker(
            embed_fn=embedder.embed_batch,
            counter=embed_counter,
            min_tokens=config.indexing.chunker.min_tokens,
            max_tokens=config.indexing.chunker.max_tokens,
            accept_pair=config.indexing.chunker.accept_pair,
        )
    elif chunker_mode == 'llm':
        chunker_llm = llm_clients[role_mapping.name_for('chunker')]
        chunker = LLMChunker(
            chunker_llm,
            token_counter=llm_counter,
            halving_retries=config.indexing.chunker.halving_retries,
            fallback_enabled=config.indexing.chunker.fallback,
        )
    elif chunker_mode == 'boundary':
        chunker_llm = llm_clients[role_mapping.name_for('chunker')]
        chunker = BoundaryChunker(
            chunker_llm,
            token_counter=llm_counter,
            max_tokens=config.indexing.chunker.max_tokens,
        )
    elif chunker_mode == 'transcript':
        # аудио-транскрипты: LLM-границы по репликам, не режет поперёк фразы, адаптивный сплит монолога,
        # start_sec/end_sec/speakers в payload. Зафиксировано экспериментом (max_tokens≈900).
        chunker_llm = llm_clients[role_mapping.name_for('chunker')]
        chunker = TranscriptChunker(
            chunker_llm,
            token_counter=llm_counter,
            max_tokens=config.indexing.chunker.max_tokens,
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
            chunker_llm = llm_clients[role_mapping.name_for('chunker')]
            llm_chunker_for_hybrid = LLMChunker(
                chunker_llm,
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
            llm_clients[role_mapping.name_for('context_generation')],
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
            llm_client=llm_clients[role_mapping.name_for('doc_title')],
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
            llm_client=llm_clients[role_mapping.name_for('doc_summary')],
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
        TimestampProcessor(),
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
    skip_presplit = chunker_mode in ('semantic', 'hybrid', 'section', 'boundary', 'transcript')
    if chunker_mode == 'llm':
        chunker_llm_inst = config.llm_by_name(role_mapping.name_for('chunker'))
        llm_safe_limit = (chunker_llm_inst.context_window - _LLM_PROMPT_OVERHEAD) // 2
        block_limit = min(config.indexing.chunker.block_limit, llm_safe_limit)
        if block_limit < config.indexing.chunker.block_limit:
            logger.info(
                'LLM block limit capped: %d → %d (context_window=%d, overhead=%d)',
                config.indexing.chunker.block_limit, block_limit,
                chunker_llm_inst.context_window, _LLM_PROMPT_OVERHEAD,
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
        narrate_tables_enabled=config.indexing.chunker.narrate_tables.enabled,
        narrate_tables_min_rows=config.indexing.chunker.narrate_tables.min_rows,
        status_reporter=reporter,
        cancel_event=cancel_event,
        run_context=run_ctx,
        embedder_fingerprint=embedder_fingerprint,
        reindex_floor=effort_floor,
    )

    logger.info(
        'Chunker: %s, context: %s%s',
        chunker_mode, config.indexing.context.mode,
        f', block_limit: {block_limit}' if not skip_presplit else '',
    )

    try:
        # ---- Phase 1: source-индексация (всё кроме jira) ----
        # Jira отдельно: ей нужны уже-проиндексированные доки для поиска ссылок.
        for src_cfg in enabled_sources:
            if cancel_event.is_set():
                break
            if src_cfg.kind == 'jira':
                continue
            if not _in_scope(src_cfg.name):
                continue

            if src_cfg.kind == 'local':
                local_source = LocalDocumentSource(
                    root=src_cfg.path,
                    pdf_converter=pdf_converter,
                    name=src_cfg.name,
                )
                pdf_mode = config.pdf.mode if config.pdf else 'disabled'
                logger.info('Source: local[%s] path=%s (pdf=%s)',
                            src_cfg.name, src_cfg.path, pdf_mode)
                await local_source.run(pipeline)

            elif src_cfg.kind == 'confluence':
                confl_source = ConfluenceSource(
                    src_cfg,
                    vision_client=vision_client,
                    vision_max_tokens=config.indexing.vision_max_tokens,
                )
                logger.info('Source: confluence[%s] url=%s', src_cfg.name, src_cfg.url)
                await pipeline.run(confl_source)

            else:
                logger.warning('Unknown source kind=%s name=%s — skipping',
                               src_cfg.kind, src_cfg.name)

        # ---- Phase 2: Confluence PDF attachments per Confluence-instance ----
        for src_cfg in config.sources_by_kind('confluence'):
            if cancel_event.is_set():
                break
            if not _in_scope(src_cfg.name):
                continue
            if not src_cfg.attachments.enabled:
                continue
            if pdf_converter is None:
                logger.warning(
                    'Confluence[%s] attachments enabled but pdf is not configured — skipping',
                    src_cfg.name,
                )
                continue
            attachments_source = ConfluencePdfSource(
                config=src_cfg, converter=pdf_converter, doc_repo=doc_repo,
            )
            logger.info('Source: confluence_pdf[%s] (mime_types=%s)',
                        src_cfg.name, src_cfg.attachments.mime_types)
            await pipeline.run(attachments_source)

        # ---- Phase 3: Multi-Jira (по одному JiraLinkExtractor на инстанс) ----
        # Каждый extractor знает свой base_url, находит «свои» PROJ-XXX ссылки в
        # уже-проиндексированных документах. Удаление устаревших задач —
        # каскадно через parent_doc_ids при удалении родительских документов.
        jira_configs = config.sources_by_kind('jira')
        if jira_configs and not cancel_event.is_set():
            all_docs = await doc_repo.scroll_all(exclude_source_types=['attached_jira'])
            logger.info('Scanning %d indexed document(s) for Jira links...', len(all_docs))
            path_to_doc_id = {doc.path[0]: doc.id for doc in all_docs if doc.path}

            for jira_cfg in jira_configs:
                if cancel_event.is_set():
                    break
                if not _in_scope(jira_cfg.name):
                    continue
                logger.info('Jira[%s]: scanning for links @ %s', jira_cfg.name, jira_cfg.url)
                extractor = JiraLinkExtractor(jira_cfg.url)
                issue_map = extractor.extract_from_docs(all_docs)
                if not issue_map:
                    logger.info('Jira[%s]: no issues found in documents, skipping',
                                jira_cfg.name)
                    continue
                logger.info('Jira[%s]: found %d issue(s)', jira_cfg.name, len(issue_map))
                parent_ids_map: dict[str, list[str]] = {
                    key: [path_to_doc_id[p] for p in paths if p in path_to_doc_id]
                    for key, paths in issue_map.items()
                }
                jira_source = JiraSource(jira_cfg, issue_map, parent_ids_map)
                await pipeline.run(jira_source)

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
                llm_client=llm_clients[role_mapping.name_for('knowledge_map')],
                doc_repo=doc_repo,
                collection=km_cfg.collection,
                node_min_tokens=km_cfg.node_min_tokens,
                prompt_strategy=km_cfg.prompt_strategy,
                prompt_budget=km_cfg.prompt_budget,
                token_counter=llm_counter,
                # KM-override концарренси если задан в conf; иначе общий
                # indexing.concurrency.
                concurrency=km_cfg.concurrency if km_cfg.concurrency is not None
                else config.indexing.concurrency,
                exclude_source_types=km_cfg.exclude_source_types,
                flat_topics_target=km_cfg.flat_topics_target,
                flat_topics_max_input_docs=km_cfg.flat_topics_max_input_docs,
                flat_topics_assign_batch=km_cfg.flat_topics_assign_batch,
                subtree_depth_limit=km_cfg.subtree_depth_limit,
                intermediate_max_tokens=km_cfg.intermediate_max_tokens,
                per_element_min_tokens=km_cfg.per_element_min_tokens,
            )
            # Roots — собираются со ВСЕХ Confluence-инстансов. ancestor_ids
            # в config — raw external IDs, в Qdrant они хранятся prefixed:
            # `confluence:<name>:<id>`.
            root_ids = set()
            for confl_cfg in config.sources_by_kind('confluence'):
                root_ids.update(
                    f'confluence:{confl_cfg.name}:{aid}'
                    for aid in confl_cfg.ancestor_ids
                )
            logger.info('Generating Knowledge Map (roots=%s)...', root_ids or 'auto')
            await km_generator.generate(root_ids=root_ids or None)
            reporter.document_done('knowledge_map')

        if cancel_event.is_set():
            reporter.finish('cancelled')
        else:
            reporter.finish('completed')
            if effort_floor is not None:
                # Эффорт дошёл до конца — снимаем маркер. Краш/cancel оставит
                # маркер на диске, следующий запуск его резюмит (ADR-0014).
                _clear_reindex_effort()
                logger.info('Reindex effort completed, marker cleared')
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

    # KM-llm берётся из pool по роли 'knowledge_map' (или default).
    llm_clients = _build_llm_clients(config.llms)
    km_llm = llm_clients[config.indexing.llm.name_for('knowledge_map')]
    llm_counter = TiktokenCounter()

    from morag.indexing.knowledge_map import KnowledgeMapGenerator
    km_cfg = config.indexing.knowledge_map
    km_generator = KnowledgeMapGenerator(
        client=client,
        llm_client=km_llm,
        doc_repo=doc_repo,
        collection=km_cfg.collection,
        node_min_tokens=km_cfg.node_min_tokens,
        prompt_strategy=km_cfg.prompt_strategy,
        prompt_budget=km_cfg.prompt_budget,
        token_counter=llm_counter,
        # KM-override концарренси если задан в conf; иначе общий indexing.concurrency.
        concurrency=km_cfg.concurrency if km_cfg.concurrency is not None
        else config.indexing.concurrency,
        exclude_source_types=km_cfg.exclude_source_types,
        flat_topics_target=km_cfg.flat_topics_target,
        flat_topics_max_input_docs=km_cfg.flat_topics_max_input_docs,
        flat_topics_assign_batch=km_cfg.flat_topics_assign_batch,
        subtree_depth_limit=km_cfg.subtree_depth_limit,
        intermediate_max_tokens=km_cfg.intermediate_max_tokens,
        per_element_min_tokens=km_cfg.per_element_min_tokens,
    )
    # Roots — со всех Confluence-инстансов, prefix как в ID документов
    root_ids = set()
    for confl_cfg in config.sources_by_kind('confluence'):
        root_ids.update(
            f'confluence:{confl_cfg.name}:{aid}' for aid in confl_cfg.ancestor_ids
        )
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
    from morag.setup_gate import SetupIncomplete, is_setup_complete

    # state-file для status_reporter и публикации прогресса наружу
    status_path = os.environ.get(
        'MORAG_STATUS_FILE',
        '/app/conf/state/index_status.json',
    )
    control_port = int(os.environ.get('MORAG_CONTROL_PORT', '9090'))

    control_plane = IndexerControlPlane(
        config_path=config_path,
        status_file_path=status_path,
        run_index=lambda **kw: cmd_index(config_path, **kw),
        run_rebuild_km=lambda **kw: cmd_rebuild_km(config_path, **kw),
        stop_grace_seconds=config.indexing.stop_grace_seconds,
    )

    # ---- Cron scheduler ----
    # Всегда запускаем scheduler. Job добавляется только если schedule настроен;
    # перенастройка через POST /control/reload-schedule (без рестарта контейнера).
    scheduler = AsyncIOScheduler()

    CRON_JOB_ID = 'indexing-cron'

    async def cron_trigger() -> None:
        try:
            await control_plane.start_index(reset=False)
            logger.info('Cron-triggered indexing started')
        except AlreadyRunning:
            logger.warning('Cron-trigger skipped: indexing already running')
        except SetupIncomplete as e:
            logger.warning('Cron-trigger skipped: setup incomplete: %s', e)
        except Exception:
            logger.exception('Cron-trigger failed')

    def _apply_schedule_from_config() -> str | None:
        """Перечитывает config и пере-устанавливает cron job. Возвращает active expression."""
        cfg = load_config(config_path)
        existing = scheduler.get_job(CRON_JOB_ID)
        new_expr = cfg.indexing.schedule if cfg.indexing else None
        if existing:
            scheduler.remove_job(CRON_JOB_ID)
        if new_expr:
            scheduler.add_job(
                cron_trigger,
                CronTrigger.from_crontab(new_expr),
                id=CRON_JOB_ID,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=None,
            )
        return new_expr

    scheduler.start()
    initial_schedule = _apply_schedule_from_config()
    if initial_schedule:
        logger.info('Cron schedule active: %s', initial_schedule)
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

    @app.get('/control/setup-status')
    async def _setup_status():
        ok, blockers = is_setup_complete(config_path)
        return {'ok': ok, 'blockers': blockers}

    @app.post('/control/start')
    async def _start(req: _StartReq):
        try:
            info = await control_plane.start_index(reset=req.reset)
            return {'started_at': info.started_at, 'kind': info.kind, 'reset': info.reset}
        except AlreadyRunning as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except SetupIncomplete as e:
            raise HTTPException(status_code=412, detail={'blockers': e.blockers}) from e

    @app.post('/control/reindex')
    async def _reindex(req: _ReindexReq):
        """Плавный реиндекс scope (ADR-0014): источник по имени или 'all'.

        Повторный вызов при незавершённом эффорте резюмит его, пропуская
        уже обработанные документы.
        """
        try:
            info = await control_plane.start_index(reindex_scope=req.scope)
            return {'started_at': info.started_at, 'kind': info.kind, 'scope': info.scope}
        except AlreadyRunning as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except SetupIncomplete as e:
            raise HTTPException(status_code=412, detail={'blockers': e.blockers}) from e

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
        except SetupIncomplete as e:
            raise HTTPException(status_code=412, detail={'blockers': e.blockers}) from e

    @app.post('/control/fetch-page')
    async def _fetch_page(req: _FetchPageReq):
        """Live-fetch одной страницы (read_pages Фаза 2): строит Source инстанса и зовёт
        `load_one` → markdown. Read-only, ВНЕ индекс-Lock (не блокируется текущей индексацией).
        Для НЕ-индексированных страниц (особенно Jira — индексятся только прикреплённые)."""
        try:
            cfg = load_config(config_path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f'config error: {e}') from e
        clients = _build_llm_clients(cfg.llms) if cfg.llms else {}
        src = build_source_for_instance(cfg, req.kind, req.name, clients)
        if src is None:
            raise HTTPException(status_code=404, detail=f'нет источника {req.kind}:{req.name}')
        # JiraSource.load_one гейтит по issue_map (link-extraction-флоу полной индексации).
        # Для standalone live-fetch регистрируем запрошенный ключ с пустыми путями —
        # `_fetch_full` дальше тянет issue из Atlassian, doc_paths нужны лишь для display.
        if req.kind == 'jira' and hasattr(src, '_issue_map'):
            src._issue_map[req.external_id] = []
        doc_id = f'{req.kind}:{req.name}:{req.external_id}'
        try:
            doc = await src.load_one(doc_id)
        except Exception as e:  # noqa: BLE001 — фетч/конвертация упали → 502 наверх
            logger.warning('fetch-page %s failed: %s', doc_id, e)
            raise HTTPException(status_code=502, detail=f'fetch failed: {e}') from e
        if doc is None:
            raise HTTPException(status_code=404, detail=f'страница не найдена: {doc_id}')
        return {
            'ok': True,
            'doc_id': doc.id,
            'title': doc.title,
            'text': doc.text,
            'url': doc.url,
            'updated_at': doc.updated_at.isoformat() if doc.updated_at else None,
        }

    @app.post('/control/reload-schedule')
    async def _reload_schedule():
        """Hot-reload: перечитать config.indexing.schedule, перенастроить APScheduler.

        Console дёргает после PATCH /api/config когда изменилось поле schedule —
        чтобы новое расписание подхватилось без рестарта контейнера.
        """
        active = _apply_schedule_from_config()
        return {'schedule': active}

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
   █  /\\__/\\  █      Catonmoon
   █ ( =^.^=) █      ╔╦╗ ╔═╗ ┬─┐ ┌─┐ ┌─┐
   █  / ></   █      ║║║ ║ ║ ├┬┘ ├─┤ │ ┬
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
        '--config', default=os.environ.get('MORAG_CONFIG_PATH', 'config.yml'),
        metavar='PATH',
        help='Путь к конфигу (по умолчанию: env MORAG_CONFIG_PATH или config.yml)',
    )
    index_parser.add_argument(
        '--reset', action='store_true',
        help='Удалить все коллекции Qdrant перед индексацией (полная переиндексация)',
    )

    km_parser = subparsers.add_parser('rebuild-km', help='Перестроить Knowledge Map без полной индексации')
    km_parser.add_argument(
        '--config', default=os.environ.get('MORAG_CONFIG_PATH', 'config.yml'),
        metavar='PATH',
        help='Путь к конфигу (по умолчанию: env MORAG_CONFIG_PATH или config.yml)',
    )

    serve_parser = subparsers.add_parser('serve', help='Daemon-режим: индексация по расписанию из конфига')
    serve_parser.add_argument(
        '--config', default=os.environ.get('MORAG_CONFIG_PATH', 'config.yml'),
        metavar='PATH',
        help='Путь к конфигу (по умолчанию: env MORAG_CONFIG_PATH или config.yml)',
    )

    query_parser = subparsers.add_parser('query', help='Гибридный поиск без LLM-ответа (для отладки)')
    query_parser.add_argument(
        '--config', default=os.environ.get('MORAG_CONFIG_PATH', 'config.yml'),
        metavar='PATH',
        help='Путь к конфигу (по умолчанию: env MORAG_CONFIG_PATH или config.yml)',
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
