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
        path=[doc_id],
        text='# Документ',
        updated_at=updated_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=1024,
    )
    defaults.update(kwargs)
    return Document(**defaults)


def make_stub(doc_id: str = 'test.md', updated_at: datetime | None = None, **kwargs) -> Document:
    """Стаб метаданных документа (text='')."""
    defaults = dict(
        id=doc_id,
        path=[doc_id],
        text='',
        updated_at=updated_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=1024,
    )
    defaults.update(kwargs)
    return Document(**defaults)


def setup_source(source: MagicMock, docs: list[Document]) -> None:
    """Настроить мок источника: get_metadata возвращает стабы, load_one — полные документы."""
    stubs = [make_stub(d.id, d.updated_at, size=d.size, source_type=d.source_type) for d in docs]
    source.get_metadata.return_value = stubs
    docs_by_id = {d.id: d for d in docs}
    source.load_one.side_effect = lambda doc_id: docs_by_id.get(doc_id)


@pytest.fixture
def doc_repo() -> AsyncMock:
    mock = AsyncMock(spec=DocRepository)
    mock.get_ids_by_source_type.return_value = set()
    return mock


@pytest.fixture
def chunk_repo() -> AsyncMock:
    return AsyncMock(spec=ChunkRepository)


@pytest.fixture
def pipeline(doc_repo, chunk_repo) -> IndexingPipeline:
    return IndexingPipeline(
        doc_repo, chunk_repo,
        chunker=PassthroughChunker(),
        context_generator=NoopContextGenerator(),
    )


# ---------------------------------------------------------------------------
# IndexingPipeline.run() — полный цикл
# ---------------------------------------------------------------------------

class TestIndexingPipelineRun:

    async def test_indexes_new_document(self, pipeline, doc_repo, chunk_repo):
        """Новый документ (не найден в Qdrant) должен быть сохранён."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        doc_repo.upsert.assert_called_once()

    async def test_skips_up_to_date_document(self, pipeline, doc_repo, chunk_repo):
        """Документ с совпадающим updated_at и полным набором чанков пропускается."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = (3, 3)

        source = MagicMock(spec=Source)
        source.get_metadata.return_value = [make_stub(updated_at=ts)]

        await pipeline.run(source)

        doc_repo.upsert.assert_not_called()
        chunk_repo.delete_by_doc_id.assert_not_called()
        source.load_one.assert_not_called()

    async def test_reindexes_when_updated_at_changed(self, pipeline, doc_repo, chunk_repo):
        """Документ с изменённым updated_at должен быть переиндексирован."""
        old_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)

        doc_repo.get_by_id.return_value = make_document(updated_at=old_ts)

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=new_ts)])

        await pipeline.run(source)

        chunk_repo.delete_by_doc_id.assert_called_once_with('test.md')
        doc_repo.delete.assert_called_once_with('test.md')
        doc_repo.upsert.assert_called_once()

    async def test_skips_when_size_changed_but_updated_at_same(self, pipeline, doc_repo, chunk_repo):
        """Документ с изменённым size, но тем же updated_at пропускается (size не проверяется)."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts, size=1024)
        chunk_repo.get_index_status.return_value = (3, 3)

        source = MagicMock(spec=Source)
        source.get_metadata.return_value = [make_stub(updated_at=ts, size=2048)]

        await pipeline.run(source)

        doc_repo.upsert.assert_not_called()
        source.load_one.assert_not_called()

    async def test_delete_attached_called_on_reindex(self, pipeline, doc_repo, chunk_repo):
        """При переиндексации вызывается delete_attached для удаления attached-детей."""
        old_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=old_ts)

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=new_ts)])

        await pipeline.run(source)

        doc_repo.delete_attached.assert_called_once_with('test.md', chunk_repo)

    async def test_delete_attached_not_called_for_new_document(self, pipeline, doc_repo, chunk_repo):
        """Для нового документа delete_attached не вызывается."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        doc_repo.delete_attached.assert_not_called()

    async def test_reindexes_when_chunks_incomplete(self, pipeline, doc_repo, chunk_repo):
        """Если индексация была прервана (count < total), переиндексируем."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = (2, 5)

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=ts)])

        await pipeline.run(source)

        doc_repo.upsert.assert_called_once()

    async def test_reindexes_when_no_chunks_but_doc_exists(self, pipeline, doc_repo, chunk_repo):
        """Документ есть, но чанков нет (get_index_status = None) → переиндексировать."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=ts)])

        await pipeline.run(source)

        doc_repo.upsert.assert_called_once()

    async def test_applies_document_processors(self, doc_repo, chunk_repo):
        """Процессор должен быть вызван и его результат сохранён."""
        doc_repo.get_by_id.return_value = None

        processor = MagicMock(spec=DocumentProcessor)
        enriched = make_document(payload={'author': 'Алиса'})
        processor.process = AsyncMock(return_value=enriched)

        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            doc_processors=[processor],
        )

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        processor.process.assert_called_once()
        saved_doc = doc_repo.upsert.call_args[0][0]
        assert saved_doc.payload.get('author') == 'Алиса'

    async def test_multiple_processors_applied_in_order(self, doc_repo, chunk_repo):
        """Цепочка процессоров применяется последовательно."""
        doc_repo.get_by_id.return_value = None

        calls = []

        class OrderTracker(DocumentProcessor):
            def __init__(self, name):
                self.name = name

            async def process(self, document):
                calls.append(self.name)
                return document

        chain = [OrderTracker('first'), OrderTracker('second'), OrderTracker('third')]
        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            doc_processors=chain,
        )

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        assert calls == ['first', 'second', 'third']

    async def test_multiple_documents_processed_independently(self, pipeline, doc_repo, chunk_repo):
        """Несколько документов обрабатываются независимо."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [
            make_document('a.md'),
            make_document('b.md'),
            make_document('c.md'),
        ])

        await pipeline.run(source)

        assert doc_repo.upsert.call_count == 3
        assert chunk_repo.upsert_batch.call_count == 3

    async def test_document_saved_before_chunks(self, pipeline, doc_repo, chunk_repo):
        """Документ сохраняется в Qdrant до сохранения чанков."""
        doc_repo.get_by_id.return_value = None
        call_order = []

        async def track_doc_upsert(doc):
            call_order.append('doc_upsert')

        async def track_chunk_upsert(chunks):
            call_order.append('chunk_upsert')

        doc_repo.upsert.side_effect = track_doc_upsert
        chunk_repo.upsert_batch.side_effect = track_chunk_upsert

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        assert call_order == ['doc_upsert', 'chunk_upsert']

    async def test_load_one_not_called_for_up_to_date(self, pipeline, doc_repo, chunk_repo):
        """load_one не вызывается для актуальных документов."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        doc_repo.get_by_id.return_value = make_document(updated_at=ts)
        chunk_repo.get_index_status.return_value = (5, 5)

        source = MagicMock(spec=Source)
        source.get_metadata.return_value = [make_stub(updated_at=ts)]

        await pipeline.run(source)

        source.load_one.assert_not_called()

    async def test_load_one_called_for_stale_document(self, pipeline, doc_repo, chunk_repo):
        """load_one вызывается для устаревших документов."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        doc = make_document()
        source.get_metadata.return_value = [make_stub()]
        source.load_one.return_value = doc

        await pipeline.run(source)

        source.load_one.assert_called_once_with('test.md')

    async def test_run_saves_chunks(self, pipeline, doc_repo, chunk_repo):
        """run() сохраняет чанки для каждого нового документа."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(text='# Заголовок\n\nТекст.')])

        await pipeline.run(source)

        chunk_repo.upsert_batch.assert_called_once()
        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert len(chunks) == 1

    async def test_chunk_has_correct_doc_id(self, pipeline, doc_repo, chunk_repo):
        """Чанки ссылаются на id документа."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document('guide.md')])

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.doc_id == 'guide.md' for c in chunks)

    async def test_chunk_order_and_total(self, doc_repo, chunk_repo):
        """order и total выставлены корректно."""
        doc_repo.get_by_id.return_value = None

        from morag.indexing.chunker import Chunker

        class TripleChunker(Chunker):
            async def chunk(self, block):
                return ['A', 'B', 'C']

        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=TripleChunker(),
            context_generator=NoopContextGenerator(),
        )

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        chunks: list[Chunk] = chunk_repo.upsert_batch.call_args[0][0]
        assert len(chunks) == 3
        assert [c.order for c in chunks] == [0, 1, 2]
        assert all(c.total == 3 for c in chunks)

    async def test_noop_context_sets_empty_string(self, pipeline, doc_repo, chunk_repo):
        """NoopContextGenerator оставляет context пустым."""
        doc_repo.get_by_id.return_value = None

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

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

        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            chunk_processors=[TagProcessor()],
        )

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.payload.get('tagged') is True for c in chunks)

    async def test_chunks_inherit_updated_at(self, pipeline, doc_repo, chunk_repo):
        """updated_at чанков совпадает с updated_at документа."""
        doc_repo.get_by_id.return_value = None
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=ts)])

        await pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.updated_at == ts for c in chunks)
