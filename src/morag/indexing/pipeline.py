from __future__ import annotations

import asyncio
import logging

from morag.indexing.chunker import Chunker, PassthroughChunker
from morag.indexing.context import ContextGenerator, NoopContextGenerator
from morag.indexing.processors import ChunkProcessor, DocumentProcessor
from morag.indexing.splitter import (
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    TableRowSplitter,
    pack_blocks,
)
from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.sources.base import Chunk, Document, Source
from morag.storage.repository import ChunkRepository, DocRepository

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_LIMIT = 2048  # токенов на один блок перед чанкованием


class IndexingPipeline:
    """Оркестратор индексации документов.

    Полный цикл через run():
    1. Загружает документы из Source.
    2. Проверяет актуальность (idempotency).
    3. Прогоняет через DocumentProcessor-цепочку.
    4. Сохраняет документ в Qdrant (коллекция docs).
    5. Разбивает на блоки (RecursiveSplitter + pack_blocks).
    6. Чанкует каждый блок (Chunker).
    7. Генерирует контекстуальное суммари (ContextGenerator).
    8. Прогоняет чанки через ChunkProcessor-цепочку.
    9. Сохраняет чанки в Qdrant (коллекция chunks).
    """

    def __init__(
        self,
        doc_repo: DocRepository,
        chunk_repo: ChunkRepository,
        doc_processors: list[DocumentProcessor] | None = None,
        chunk_processors: list[ChunkProcessor] | None = None,
        chunker: Chunker | None = None,
        context_generator: ContextGenerator | None = None,
        token_counter: TokenCounter | None = None,
        block_limit: int = _DEFAULT_BLOCK_LIMIT,
        concurrency: int = 1,
    ) -> None:
        self._doc_repo = doc_repo
        self._chunk_repo = chunk_repo
        self._doc_processors = doc_processors or []
        self._chunk_processors = chunk_processors or []
        self._chunker = chunker or PassthroughChunker()
        self._context_generator = context_generator or NoopContextGenerator()
        self._token_counter = token_counter or TiktokenCounter()
        self._block_limit = block_limit
        self._concurrency = max(1, concurrency)
        self._splitter = RecursiveSplitter(
            self._token_counter,
            self._block_limit,
            splitters=[
                MarkdownHeaderSplitter(),
                TableRowSplitter(),
                FixedSizeSplitter(self._token_counter, self._block_limit),
            ],
        )

    async def _is_up_to_date(self, stub: Document) -> bool:
        """Проверить актуальность документа по стабу метаданных без загрузки контента."""
        existing = await self._doc_repo.get_by_id(stub.id)
        if existing is None:
            return False

        same_content = existing.updated_at == stub.updated_at
        if stub.source_type not in ('confluence', 'jira'):
            same_content = same_content and existing.size == stub.size
        if not same_content:
            return False

        status = await self._chunk_repo.get_index_status(stub.id)
        if status is None:
            return False
        count, total = status
        return count == total

    async def _prepare_document(self, document: Document, w: str = '') -> Document | None:
        """Проверить idempotency, удалить устаревшее, сохранить документ.

        Возвращает обработанный документ или None если документ актуален.
        """
        logger.info('%sPreparing document: %s (size=%d)', w, document.id, document.size)
        existing = await self._doc_repo.get_by_id(document.id)

        if existing is not None:
            # Для Confluence и Jira size нестабилен, ориентируемся только на дату изменения.
            same_content = existing.updated_at == document.updated_at
            if document.source_type not in ('confluence', 'jira'):
                same_content = same_content and existing.size == document.size
            if same_content:
                status = await self._chunk_repo.get_index_status(document.id)
                if status is not None:
                    count, total = status
                    if count == total:
                        logger.info('%sDocument up to date, skipping: %s', w, document.id)
                        return None

            # Документ изменился или индексация была прервана — удаляем каскадно
            logger.info('%sRe-indexing document: %s', w, document.id)
            await self._chunk_repo.delete_by_doc_id(document.id)
            await self._doc_repo.delete(document.id)

        # Прогоняем через цепочку процессоров
        for processor in self._doc_processors:
            document = processor.process(document)

        # Сохраняем документ до начала чанкования
        await self._doc_repo.upsert(document)
        logger.info('%sDocument saved: %s', w, document.id)

        return document

    async def run(self, source: Source) -> None:
        """Полный цикл индексации с параллельной обработкой документов.

        Сначала загружаются метаданные всех документов и выполняется idempotency-проверка.
        Затем документы, требующие переиндексации, обрабатываются конкурентно — не более
        `concurrency` одновременно. Каждый документ: load_one → prepare → chunk → upsert.
        """
        stubs = await source.get_metadata()
        total = len(stubs)
        logger.info('Loaded metadata for %d document(s) from source (concurrency=%d)', total, self._concurrency)

        sem = asyncio.Semaphore(self._concurrency)

        async def process_one(i: int, stub: Document) -> bool:
            """Обработать один документ. Возвращает True если был проиндексирован."""
            w = f'[W{i}] ' if self._concurrency > 1 else ''
            async with sem:
                try:
                    logger.info('%sChecking [%d/%d]: %s', w, i, total, stub.id)
                    if await self._is_up_to_date(stub):
                        logger.info('%sDocument up to date, skipping: %s', w, stub.id)
                        return False

                    document = await source.load_one(stub.id)
                    if document is None:
                        logger.warning('%sFailed to load document: %s', w, stub.id)
                        return False

                    prepared = await self._prepare_document(document, w=w)
                    if prepared is None:
                        return False

                    await self._chunk_document(prepared, w=w)
                    return True
                except Exception:
                    logger.exception('%sDocument failed, skipping: %s', w, stub.id)
                    return False

        results = await asyncio.gather(*[process_one(i + 1, stub) for i, stub in enumerate(stubs)])
        indexed = sum(results)
        logger.info('Indexing complete: %d indexed, %d skipped', indexed, total - indexed)

    async def _chunk_document(self, document: Document, w: str = '') -> None:
        """Разбить документ на чанки и сохранить в Qdrant."""
        logger.info('%sChunking: %s', w, document.id)

        # Pre-split на блоки + жадная упаковка
        blocks = self._splitter.split(document.text)
        packs = pack_blocks(blocks, self._token_counter, self._block_limit)
        logger.info('%s  Pre-split: %d block(s) -> %d pack(s)', w, len(blocks), len(packs))

        # Chunker: каждая пачка → список текстов чанков
        chunk_texts: list[str] = []
        for i, pack in enumerate(packs):
            block_text = '\n\n'.join(pack)
            logger.info('%s  Chunking pack %d/%d (%d chars)...', w, i + 1, len(packs), len(block_text))
            new_chunks = await self._chunker.chunk(block_text)
            logger.info('%s    -> %d chunk(s)', w, len(new_chunks))
            chunk_texts.extend(new_chunks)

        total = len(chunk_texts)
        logger.info('%s  Total chunks: %d', w, total)

        # Собираем Chunk-объекты с order/total, генерируем context, применяем процессоры
        chunks: list[Chunk] = []
        for order, text in enumerate(chunk_texts):
            logger.info('%s  Processing chunk %d/%d: %s...', w, order + 1, total, repr(text[:60]))
            context = await self._context_generator.generate(document.text, text)

            chunk = Chunk(
                doc_id=document.id,
                path=document.path,
                source_type=document.source_type,
                order=order,
                total=total,
                text=text,
                context=context,
                updated_at=document.updated_at,
            )

            for processor in self._chunk_processors:
                chunk = processor.process(chunk, document)

            vec_summary = ', '.join(
                f"{k}:dense({len(v)})" if isinstance(v, list)
                else f"{k}:sparse({len(v['indices'])})"
                for k, v in chunk.vectors.items()
            )
            logger.info('%s    vectors: [%s]', w, vec_summary)
            chunks.append(chunk)

        await self._chunk_repo.upsert_batch(chunks)
        logger.info('%sChunks saved: %s (%d)', w, document.id, total)
