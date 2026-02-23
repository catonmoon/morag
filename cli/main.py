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
from morag.indexing.processors import DenseEmbeddingProcessor, SparseEmbeddingProcessor
from morag.llm.client import LLMClient
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


async def cmd_index(config_path: str) -> None:
    """Индексировать документы из источника в Qdrant."""
    config = load_config(config_path)

    logger.info('Connecting to Qdrant %s:%d', config.qdrant.host, config.qdrant.port)
    client = AsyncQdrantClient(host=config.qdrant.host, port=config.qdrant.port)

    embedder = FridaEmbedder(config.indexing.dense_model)

    logger.info('Ensuring collections...')
    await ensure_docs_collection(client, config.qdrant.collection_docs)
    await ensure_chunks_collection(
        client, config.qdrant.collection_chunks,
        vectors_config=frida_vectors_config(embedder.dim),
        sparse_vectors_config=gte_sparse_vectors_config(),
    )

    source = MarkdownSource(config.sources.markdown.path)
    doc_repo = DocRepository(client, config.qdrant.collection_docs)
    chunk_repo = ChunkRepository(client, config.qdrant.collection_chunks)

    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
    )

    chunker = LLMChunker(llm_client) if config.indexing.chunker == 'llm' else PassthroughChunker()
    context_generator = (
        LLMContextGenerator(llm_client) if config.indexing.context == 'llm' else NoopContextGenerator()
    )
    sparse_embedder = GteSparseEmbedder(config.indexing.sparse_model, device=config.indexing.sparse_device)
    chunk_processors = [
        DenseEmbeddingProcessor(embedder),
        SparseEmbeddingProcessor(sparse_embedder),
    ]

    pipeline = IndexingPipeline(
        doc_repo, chunk_repo,
        chunker=chunker,
        context_generator=context_generator,
        chunk_processors=chunk_processors,
        block_limit=config.indexing.block_limit,
    )

    logger.info('Source: %s', config.sources.markdown.path)
    logger.info('Chunker: %s, context: %s', config.indexing.chunker, config.indexing.context)
    await pipeline.run(source)

    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='morag — RAG для локальных MD-файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', default='config.yml', metavar='PATH',
        help='Путь к конфигу (по умолчанию: config.yml)',
    )

    parser.add_argument(
        '-v', '--debug', action='store_true',
        help='Включить DEBUG-логирование (показывает сырые ответы LLM)',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help='Индексировать документы из источника')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger('morag').setLevel(logging.DEBUG)

    if args.command == 'index':
        asyncio.run(cmd_index(args.config))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
