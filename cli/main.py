#!/usr/bin/env python3
"""CLI для morag: индексация документов и поиск."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qdrant_client import AsyncQdrantClient

from morag.config import load_config
from morag.indexing.chunker import LLMChunker, PassthroughChunker
from morag.indexing.context import LLMContextGenerator, NoopContextGenerator
from morag.indexing.embedder import FridaEmbedder, GteSparseEmbedder
from morag.indexing.pipeline import IndexingPipeline
from morag.indexing.processors import DenseEmbeddingProcessor, MetadataProcessor, SparseEmbeddingProcessor
from morag.indexing.token_counter import TiktokenCounter
from morag.llm.client import LLMClient
from morag.sources.confluence import ConfluenceSource
from morag.sources.markdown import MarkdownSource
from morag.storage.collections import (
    ensure_chunks_collection,
    ensure_docs_collection,
    frida_vectors_config,
    gte_sparse_vectors_config,
)
from morag.storage.repository import ChunkRepository, DocRepository

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

    embedder = FridaEmbedder(config.indexing.dense_model)

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
    )

    vision_client = None
    if config.llm_vision:
        vision_client = LLMClient(
            base_url=config.llm_vision.base_url,
            model=config.llm_vision.model,
            api_key=config.llm_vision.api_key,
        )
        logger.info('Vision LLM: %s @ %s', config.llm_vision.model, config.llm_vision.base_url)

    sources = []
    if config.sources.local_documents:
        sources.append(MarkdownSource(config.sources.local_documents.path))
        logger.info('Source: local_documents path=%s', config.sources.local_documents.path)
    if config.sources.confluence:
        sources.append(ConfluenceSource(config.sources.confluence, vision_client=vision_client))
        logger.info('Source: confluence url=%s (vision=%s)', config.sources.confluence.url, vision_client is not None)
    if not sources:
        logger.error('No sources configured in config.yml')
        return

    token_counter = TiktokenCounter()
    chunker = LLMChunker(llm_client) if config.indexing.chunker == 'llm' else PassthroughChunker()
    context_generator = (
        LLMContextGenerator(
            llm_client,
            token_counter=token_counter,
            context_window=config.indexing.llm_context_window,
            max_output_tokens=config.indexing.context_max_output_tokens,
        ) if config.indexing.context == 'llm' else NoopContextGenerator()
    )
    sparse_embedder = GteSparseEmbedder(config.indexing.sparse_model, device=config.indexing.sparse_device)
    chunk_processors = [
        MetadataProcessor(),
        DenseEmbeddingProcessor(embedder),
        SparseEmbeddingProcessor(sparse_embedder),
    ]

    # В LLM-режиме блок + ответ LLM должны влезть в контекстное окно.
    # Ответ ≈ такого же размера как вход, поэтому безопасный лимит: (context_window - overhead) / 2.
    _LLM_PROMPT_OVERHEAD = 512  # токенов на системный промпт + запас
    if config.indexing.chunker == 'llm':
        llm_safe_limit = (config.indexing.llm_context_window - _LLM_PROMPT_OVERHEAD) // 2
        block_limit = min(config.indexing.block_limit, llm_safe_limit)
        if block_limit < config.indexing.block_limit:
            logger.info(
                'LLM block limit capped: %d → %d (context_window=%d, overhead=%d)',
                config.indexing.block_limit, block_limit,
                config.indexing.llm_context_window, _LLM_PROMPT_OVERHEAD,
            )
    else:
        block_limit = config.indexing.block_limit

    pipeline = IndexingPipeline(
        doc_repo, chunk_repo,
        chunker=chunker,
        context_generator=context_generator,
        chunk_processors=chunk_processors,
        block_limit=block_limit,
        token_counter=token_counter,
    )

    logger.info('Chunker: %s, context: %s, block_limit: %d', config.indexing.chunker, config.indexing.context, block_limit)
    for source in sources:
        await pipeline.run(source)

    await client.close()


async def cmd_query(config_path: str, question: str, top_k: int) -> None:
    """Гибридный поиск по вопросу без LLM-ответа (для отладки)."""
    config = load_config(config_path)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(host=config.qdrant.host, port=config.qdrant.port)

    embedder = FridaEmbedder(config.indexing.dense_model)
    sparse_embedder = GteSparseEmbedder(config.indexing.sparse_model, device=config.indexing.sparse_device)

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


LOGO = """
  ░▒▓█████
 ░▒▓███████         Catonmoon
▒▓██(=^.^=)██       Morag v0.1
 ▓█████████
  ▓███████
"""


def main() -> None:
    print(LOGO)
    parser = argparse.ArgumentParser(
        description='morag — RAG для локальных MD-файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    elif args.command == 'query':
        asyncio.run(cmd_query(args.config, args.question, args.top_k))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
