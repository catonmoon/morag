from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from morag.indexing.chunker import PassthroughChunker
from morag.indexing.context import NoopContextGenerator
from morag.indexing.pipeline import IndexingPipeline
from morag.indexing.processors import ChunkProcessor, DocumentProcessor
from morag.sources.base import Chunk, Document, Source
from morag.storage.repository import ChunkRepository, DocRepository


def make_document(doc_id: str = 'test.md', updated_at: datetime | None = None, **kwargs) -> Document:
    defaults = dict(
        id=doc_id,
        path=doc_id,
        text='# Документ',
        updated_at=updated_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=1024,
    )
    defaults.update(kwargs)
    return Document(**defaults)


@pytest.fixture
def doc_repo() -> AsyncMock:
    return AsyncMock(spec=DocRepository)


@pytest.fixture
def chunk_repo() -> AsyncMock:
    return AsyncMock(spec=ChunkRepository)


@pytest.fixture
def pipeline(doc_repo, chunk_repo) -> IndexingPipeline:
    return IndexingPipeline(doc_repo, chunk_repo)


# ---------------------------------------------------------------------------
# IndexingPipeline
# ---------------------------------------------------------------------------

class TestIndexingPipeline:
    async def test_indexes_new_document(self, pipeline, doc_repo, chunk_repo):
        """Новый документ (не найден в Qdrant) должен быть сохранён."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        result = await pipeline.index_source(source)

        assert len(result) == 1
        doc_repo.upsert.assert_called_once()

    async def test_skips_up_to_date_document(self, pipeline, doc_repo, chunk_repo):
        """Документ с совпадающим updated_at, size и полным набором чанков пропускается."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts, size=1024)
        chunk_repo.get_index_status.return_value = (3, 3)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts, size=1024)]

        result = await pipeline.index_source(source)

        assert result == []
        doc_repo.upsert.assert_not_called()
        chunk_repo.delete_by_doc_id.assert_not_called()

    async def test_reindexes_when_updated_at_changed(self, pipeline, doc_repo, chunk_repo):
        """Документ с изменённым updated_at должен быть переиндексирован."""
        old_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)

        doc_repo.get_by_id.return_value = make_document(updated_at=old_ts)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=new_ts)]

        result = await pipeline.index_source(source)

        assert len(result) == 1
        chunk_repo.delete_by_doc_id.assert_called_once_with('test.md')
        doc_repo.delete.assert_called_once_with('test.md')
        doc_repo.upsert.assert_called_once()

    async def test_reindexes_when_size_changed(self, pipeline, doc_repo, chunk_repo):
        """Документ с изменённым size должен быть переиндексирован."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts, size=1024)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts, size=2048)]

        result = await pipeline.index_source(source)

        assert len(result) == 1
        chunk_repo.delete_by_doc_id.assert_called_once_with('test.md')
        doc_repo.delete.assert_called_once_with('test.md')
        doc_repo.upsert.assert_called_once()

    async def test_reindexes_when_chunks_incomplete(self, pipeline, doc_repo, chunk_repo):
        """Если индексация была прервана (count < total), переиндексируем."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = (2, 5)  # неполная индексация

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts)]

        result = await pipeline.index_source(source)

        assert len(result) == 1
        doc_repo.upsert.assert_called_once()

    async def test_reindexes_when_no_chunks_but_doc_exists(self, pipeline, doc_repo, chunk_repo):
        """Документ есть, но чанков нет (get_index_status = None) → переиндексировать."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = None

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts)]

        result = await pipeline.index_source(source)

        assert len(result) == 1
        doc_repo.upsert.assert_called_once()

    async def test_applies_document_processors(self, pipeline, doc_repo, chunk_repo):
        """Процессор должен быть вызван и его результат сохранён."""
        doc_repo.get_by_id.return_value = None

        processor = MagicMock(spec=DocumentProcessor)
        enriched = make_document(payload={'author': 'Алиса'})
        processor.process.return_value = enriched

        pipeline_with_proc = IndexingPipeline(doc_repo, chunk_repo, doc_processors=[processor])

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        result = await pipeline_with_proc.index_source(source)

        processor.process.assert_called_once()
        assert result[0].payload.get('author') == 'Алиса'

    async def test_multiple_processors_applied_in_order(self, pipeline, doc_repo, chunk_repo):
        """Цепочка процессоров применяется последовательно."""
        doc_repo.get_by_id.return_value = None

        calls = []

        class OrderTracker(DocumentProcessor):
            def __init__(self, name):
                self.name = name

            def process(self, document):
                calls.append(self.name)
                return document

        chain = [OrderTracker('first'), OrderTracker('second'), OrderTracker('third')]
        p = IndexingPipeline(doc_repo, chunk_repo, doc_processors=chain)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        await p.index_source(source)

        assert calls == ['first', 'second', 'third']

    async def test_multiple_documents_processed_independently(self, pipeline, doc_repo, chunk_repo):
        """Несколько документов обрабатываются независимо."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        source.load.return_value = [
            make_document('a.md'),
            make_document('b.md'),
            make_document('c.md'),
        ]

        result = await pipeline.index_source(source)

        assert len(result) == 3
        assert doc_repo.upsert.call_count == 3

    async def test_document_saved_before_returned(self, pipeline, doc_repo, chunk_repo):
        """Документ сохраняется в Qdrant до того как возвращается для чанкования."""
        doc_repo.get_by_id.return_value = None
        upsert_called_before_return = []

        async def track_upsert(doc):
            upsert_called_before_return.append(doc.id)

        doc_repo.upsert.side_effect = track_upsert

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        result = await pipeline.index_source(source)

        assert result[0].id in upsert_called_before_return


# ---------------------------------------------------------------------------
# IndexingPipeline.run() — полный цикл
# ---------------------------------------------------------------------------

class TestIndexingPipelineRun:
    def _make_pipeline(self, doc_repo, chunk_repo, **kwargs) -> IndexingPipeline:
        return IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            **kwargs,
        )

    async def test_run_saves_chunks(self, doc_repo, chunk_repo):
        """run() должен сохранить чанки для каждого нового документа."""
        doc_repo.get_by_id.return_value = None
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(text='# Заголовок\n\nТекст.')]

        await pipeline.run(source)

        chunk_repo.upsert_batch.assert_called_once()
        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert len(chunks) == 1

    async def test_run_skips_up_to_date_document(self, doc_repo, chunk_repo):
        """run() не трогает чанки если документ актуален."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts, size=1024)
        chunk_repo.get_index_status.return_value = (1, 1)
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts, size=1024)]

        await pipeline.run(source)

        chunk_repo.upsert_batch.assert_not_called()

    async def test_chunk_has_correct_doc_id(self, doc_repo, chunk_repo):
        """Чанки ссылаются на id документа."""
        doc_repo.get_by_id.return_value = None
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document('guide.md')]

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.doc_id == 'guide.md' for c in chunks)

    async def test_chunk_order_and_total(self, doc_repo, chunk_repo):
        """order и total выставлены корректно."""
        doc_repo.get_by_id.return_value = None
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        # Три блока которые PassthroughChunker вернёт как три чанка
        from morag.indexing.chunker import Chunker
        class TripleChunker(Chunker):
            def chunk(self, block):
                return ['A', 'B', 'C']

        pipeline._chunker = TripleChunker()

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        await pipeline.run(source)

        chunks: list[Chunk] = chunk_repo.upsert_batch.call_args[0][0]
        assert len(chunks) == 3
        assert [c.order for c in chunks] == [0, 1, 2]
        assert all(c.total == 3 for c in chunks)

    async def test_noop_context_sets_empty_string(self, doc_repo, chunk_repo):
        """NoopContextGenerator оставляет context пустым."""
        doc_repo.get_by_id.return_value = None
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.context == '' for c in chunks)

    async def test_chunk_processor_is_applied(self, doc_repo, chunk_repo):
        """ChunkProcessor вызывается для каждого чанка."""
        doc_repo.get_by_id.return_value = None

        class TagProcessor(ChunkProcessor):
            def process(self, chunk, document):
                chunk.payload['tagged'] = True
                return chunk

        pipeline = self._make_pipeline(doc_repo, chunk_repo, chunk_processors=[TagProcessor()])

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document()]

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.payload.get('tagged') is True for c in chunks)

    async def test_run_multiple_documents(self, doc_repo, chunk_repo):
        """Несколько документов — несколько вызовов upsert_batch."""
        doc_repo.get_by_id.return_value = None
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [
            make_document('a.md'),
            make_document('b.md'),
        ]

        await pipeline.run(source)

        assert chunk_repo.upsert_batch.call_count == 2

    async def test_chunks_inherit_updated_at(self, doc_repo, chunk_repo):
        """updated_at чанков совпадает с updated_at документа."""
        doc_repo.get_by_id.return_value = None
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        pipeline = self._make_pipeline(doc_repo, chunk_repo)

        source = MagicMock(spec=Source)
        source.load.return_value = [make_document(updated_at=ts)]

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.updated_at == ts for c in chunks)
