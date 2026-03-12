#!/usr/bin/env python3
"""CLI для morag: индексация документов и поиск."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qdrant_client import AsyncQdrantClient

from morag.config import DenseEmbedderConfig, RetryConfig, SparseEmbedderConfig, load_config
from morag.indexing.chunker import LLMChunker, PassthroughChunker, SemanticChunker
from morag.indexing.context import LLMContextGenerator, NoopContextGenerator
from morag.indexing.embedder import (
    Embedder,
    FridaEmbedder,
    GteSparseEmbedder,
    HttpFridaEmbedder,
    HttpGteSparseEmbedder,
    SparseEmbedder,
)
from morag.indexing.pipeline import IndexingPipeline
from morag.indexing.processors import (
    DenseEmbeddingProcessor,
    DocSummaryProcessor,
    MetadataProcessor,
    SparseEmbeddingProcessor,
)
from morag.indexing.token_counter import TiktokenCounter
from morag.llm.client import LLMClient
from morag.llm.retry import RetryPolicy
from morag.sources.confluence import ConfluenceSource
from morag.sources.confluence_pdf import ConfluencePdfSource
from morag.sources.jira import JiraSource
from morag.sources.jira_extractor import JiraLinkExtractor
from morag.sources.local import LocalDocumentSource
from morag.sources.pdf_converter import DoclingPdfConverter
from morag.storage.collections import (
    ensure_chunks_collection,
    ensure_docs_collection,
    frida_vectors_config,
    gte_sparse_vectors_config,
)
from morag.storage.repository import ChunkRepository, DocRepository

def _make_retry(cfg: RetryConfig) -> RetryPolicy:
    return RetryPolicy(max_retries=cfg.max_retries)


def _make_dense_embedder(cfg: DenseEmbedderConfig) -> Embedder:
    if cfg.base_url is not None:
        return HttpFridaEmbedder(cfg.base_url, cfg.dim, cfg.timeout, retry_policy=_make_retry(cfg.retry))
    return FridaEmbedder(cfg.model)


def _make_sparse_embedder(cfg: SparseEmbedderConfig) -> SparseEmbedder:
    if cfg.base_url is not None:
        return HttpGteSparseEmbedder(cfg.base_url, cfg.timeout, retry_policy=_make_retry(cfg.retry))
    return GteSparseEmbedder(cfg.model, device=cfg.device)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


async def cmd_index(config_path: str, reset: bool = False) -> None:
    """Индексировать документы из источника в Qdrant."""
    config = load_config(config_path)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(host=config.qdrant.host, port=config.qdrant.port)

    if reset:
        existing = {c.name for c in (await client.get_collections()).collections}
        for name in (config.qdrant.collection_docs, config.qdrant.collection_chunks):
            if name in existing:
                logger.warning('Dropping collection: %s', name)
                await client.delete_collection(name)

    embedder = _make_dense_embedder(config.indexing.dense_embedder)

    logger.info('Ensuring collections...')
    await ensure_docs_collection(client, config.qdrant.collection_docs)
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
        max_retries=config.llm.retry.max_retries,
        model_wait_seconds=config.llm.model_wait_seconds,
        model_wait_retries=config.llm.model_wait_retries,
    )

    vision_client = None
    if config.llm_vision:
        vision_client = LLMClient(
            base_url=config.llm_vision.base_url,
            model=config.llm_vision.model,
            api_key=config.llm_vision.api_key,
            timeout=config.llm_vision.timeout,
            max_retries=config.llm_vision.retry.max_retries,
        )
        logger.info('Vision LLM: %s @ %s', config.llm_vision.model, config.llm_vision.base_url)

    local_source = None
    if config.sources.local_documents:
        local_source = LocalDocumentSource(
            root=config.sources.local_documents.path,
            docling_base_url=config.docling.base_url if config.docling else None,
            docling_timeout=config.docling.timeout if config.docling else 300,
            vision_client=vision_client,
        )
        logger.info(
            'Source: local_documents path=%s (docling=%s)',
            config.sources.local_documents.path,
            config.docling.base_url if config.docling else 'disabled',
        )

    sources = []
    if config.sources.confluence:
        sources.append(ConfluenceSource(config.sources.confluence, vision_client=vision_client))
        logger.info('Source: confluence url=%s (vision=%s)', config.sources.confluence.url, vision_client is not None)
    if not sources and local_source is None:
        logger.error('No sources configured in config.yml')
        return

    token_counter = TiktokenCounter()
    chunker_mode = config.indexing.chunker.mode
    if chunker_mode == 'semantic':
        chunker = SemanticChunker(
            embed_fn=embedder.embed_batch,
            counter=token_counter,
            min_tokens=config.indexing.chunker.min_tokens,
            max_tokens=config.indexing.chunker.max_tokens,
        )
    elif chunker_mode == 'llm':
        chunker = LLMChunker(
            llm_client,
            token_counter=token_counter,
            embed_fn=embedder.embed,
            halving_retries=config.indexing.chunker.halving_retries,
            fallback_enabled=config.indexing.chunker.fallback,
        )
    else:
        chunker = PassthroughChunker()
    context_generator = (
        LLMContextGenerator(
            llm_client,
            token_counter=token_counter,
            context_window=config.llm.context_window,
            max_output_tokens=config.indexing.context.max_tokens,
        ) if config.indexing.context.mode == 'llm' else NoopContextGenerator()
    )
    sparse_embedder = _make_sparse_embedder(config.indexing.sparse_embedder)

    doc_processors = []
    if config.indexing.doc_summary.max_tokens is not None:
        doc_processors.append(DocSummaryProcessor(
            llm_client=llm_client,
            doc_repo=doc_repo,
            max_tokens=config.indexing.doc_summary.max_tokens,
            token_counter=token_counter,
            context_window=config.llm.context_window,
        ))

    chunk_processors = [
        MetadataProcessor(),
        DenseEmbeddingProcessor(embedder),
        SparseEmbeddingProcessor(sparse_embedder),
    ]

    # В LLM-режиме блок + ответ LLM должны влезть в контекстное окно.
    # Ответ ≈ такого же размера как вход, поэтому безопасный лимит: (context_window - overhead) / 2.
    _LLM_PROMPT_OVERHEAD = 512  # токенов на системный промпт + запас
    skip_presplit = chunker_mode == 'semantic'
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
        token_counter=token_counter,
        concurrency=config.indexing.concurrency,
        skip_presplit=skip_presplit,
    )

    logger.info(
        'Chunker: %s, context: %s%s',
        chunker_mode, config.indexing.context.mode,
        f', block_limit: {block_limit}' if not skip_presplit else '',
    )

    # Локальные документы: композитный source с собственным порядком запуска
    if local_source is not None:
        await local_source.run(pipeline)

    # Остальные source (Confluence и т.д.)
    for source in sources:
        await pipeline.run(source)

    # Confluence PDF attachments: после страниц, чтобы parent pages уже были в базе
    if config.sources.confluence and config.sources.confluence.attachments.enabled:
        if config.docling:
            confluence_pdf_converter = DoclingPdfConverter(
                docling_base_url=config.docling.base_url,
                docling_timeout=config.docling.timeout,
                vision_client=vision_client,
            )
            confluence_pdf_source = ConfluencePdfSource(
                config=config.sources.confluence,
                converter=confluence_pdf_converter,
                doc_repo=doc_repo,
            )
            logger.info(
                'Source: confluence_pdf (mime_types=%s)',
                config.sources.confluence.attachments.mime_types,
            )
            await pipeline.run(confluence_pdf_source)
        else:
            logger.warning(
                'Confluence attachments enabled but docling is not configured — skipping'
            )

    # Jira: сканируем проиндексированные документы на ссылки, затем индексируем задачи.
    # Удаление устаревших задач происходит каскадно через parent_doc_ids при удалении родительских документов.
    if config.sources.jira:
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

    await client.close()


async def cmd_serve(config_path: str) -> None:
    """Запустить daemon-режим: индексация по cron-расписанию из конфига."""
    config = load_config(config_path)
    if not config.indexing.schedule:
        logger.error('indexing.schedule is not set in config.yml — cannot start serve mode')
        sys.exit(1)

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    async def run_index() -> None:
        logger.info('Starting scheduled indexing...')
        try:
            await cmd_index(config_path)
            logger.info('Scheduled indexing complete')
        except Exception:
            logger.exception('Scheduled indexing failed')

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_index,
        CronTrigger.from_crontab(config.indexing.schedule),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
    )
    scheduler.start()
    logger.info('Serve mode started. Schedule: %s', config.indexing.schedule)

    logger.info('Running initial indexing...')
    await run_index()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info('Scheduler stopped')


async def cmd_query(config_path: str, question: str, top_k: int) -> None:
    """Гибридный поиск по вопросу без LLM-ответа (для отладки)."""
    config = load_config(config_path)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(host=config.qdrant.host, port=config.qdrant.port)

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
   ░▒▓██████
  ░▒▓█/\ /\█▓       Catonmoon
  ▒▓█(=^.^=)▒       Morag v{_VERSION}
   ▓████████  
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
    elif args.command == 'serve':
        asyncio.run(cmd_serve(args.config))
    elif args.command == 'query':
        asyncio.run(cmd_query(args.config, args.question, args.top_k))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
