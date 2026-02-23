from __future__ import annotations

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
    ) -> None:
        self._doc_repo = doc_repo
        self._chunk_repo = chunk_repo
        self._doc_processors = doc_processors or []
        self._chunk_processors = chunk_processors or []
        self._chunker = chunker or PassthroughChunker()
        self._context_generator = context_generator or NoopContextGenerator()
        self._token_counter = token_counter or TiktokenCounter()
        self._block_limit = block_limit
        self._splitter = RecursiveSplitter(
            self._token_counter,
            self._block_limit,
            splitters=[
                MarkdownHeaderSplitter(),
                TableRowSplitter(),
                FixedSizeSplitter(self._token_counter, self._block_limit),
            ],
        )

    async def index_source(self, source: Source) -> list[Document]:
        """Загрузить все документы из source и подготовить к чанкованию.

        Возвращает только те документы которые требуют (пере)индексации.
        Актуальные документы пропускаются.
        """
        documents = source.load()
        logger.info('Loaded %d document(s) from source', len(documents))

        to_index: list[Document] = []
        for document in documents:
            result = await self._prepare_document(document)
            if result is not None:
                to_index.append(result)

        logger.info('Documents to index: %d, skipped: %d', len(to_index), len(documents) - len(to_index))
        return to_index

    async def _prepare_document(self, document: Document) -> Document | None:
        """Проверить idempotency, удалить устаревшее, сохранить документ.

        Возвращает обработанный документ или None если документ актуален.
        """
        logger.info('Preparing document: %s (size=%d)', document.id, document.size)
        existing = await self._doc_repo.get_by_id(document.id)

        if existing is not None:
            if existing.updated_at == document.updated_at and existing.size == document.size:
                status = await self._chunk_repo.get_index_status(document.id)
                if status is not None:
                    count, total = status
                    if count == total:
                        logger.info('Document up to date, skipping: %s', document.id)
                        return None

            # Документ изменился или индексация была прервана — удаляем каскадно
            logger.info('Re-indexing document: %s', document.id)
            await self._chunk_repo.delete_by_doc_id(document.id)
            await self._doc_repo.delete(document.id)

        # Прогоняем через цепочку процессоров
        for processor in self._doc_processors:
            document = processor.process(document)

        # Сохраняем документ до начала чанкования
        await self._doc_repo.upsert(document)
        logger.info('Document saved: %s', document.id)

        return document

    async def run(self, source: Source) -> None:
        """Полный цикл индексации: загрузка → чанкование → сохранение в Qdrant."""
        documents = await self.index_source(source)
        for document in documents:
            await self._chunk_document(document)

    async def _chunk_document(self, document: Document) -> None:
        """Разбить документ на чанки и сохранить в Qdrant."""
        logger.info('Chunking document: %s', document.id)

        # Pre-split на блоки + жадная упаковка
        blocks = self._splitter.split(document.text)
        packs = pack_blocks(blocks, self._token_counter, self._block_limit)
        logger.info('  Pre-split: %d block(s) -> %d pack(s)', len(blocks), len(packs))

        # Chunker: каждая пачка → список текстов чанков
        chunk_texts: list[str] = []
        for i, pack in enumerate(packs):
            block_text = '\n\n'.join(pack)
            logger.info('  Chunking pack %d/%d (%d chars)...', i + 1, len(packs), len(block_text))
            new_chunks = await self._chunker.chunk(block_text)
            logger.info('    -> %d chunk(s)', len(new_chunks))
            chunk_texts.extend(new_chunks)

        total = len(chunk_texts)
        logger.info('  Total chunks: %d', total)

        # Собираем Chunk-объекты с order/total, генерируем context, применяем процессоры
        chunks: list[Chunk] = []
        for order, text in enumerate(chunk_texts):
            logger.info('  Processing chunk %d/%d: %s...', order + 1, total, repr(text[:60]))
            context = await self._context_generator.generate(document.text, text)

            chunk = Chunk(
                doc_id=document.id,
                path=document.path,
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
            logger.info('    vectors: [%s]', vec_summary)
            chunks.append(chunk)

        await self._chunk_repo.upsert_batch(chunks)
        logger.info('Chunks saved: %s (%d)', document.id, total)
