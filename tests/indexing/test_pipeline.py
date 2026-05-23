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
    mock.get_ids_by_source_instance.return_value = set()
    mock.get_payloads_by_ids.return_value = {}
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


@pytest.fixture
def run_ctx():
    from morag.run_context import RunContext
    return RunContext(run_number=42, indexed_at='2026-05-02T10:00:00+00:00')


@pytest.fixture
def stamped_pipeline(doc_repo, chunk_repo, run_ctx) -> IndexingPipeline:
    """Pipeline с RunContext и embedder fingerprint — payload-stamping включён."""
    return IndexingPipeline(
        doc_repo, chunk_repo,
        chunker=PassthroughChunker(),
        context_generator=NoopContextGenerator(),
        run_context=run_ctx,
        embedder_fingerprint='fp-v1-abc',
    )


@pytest.fixture
def reindex_pipeline(doc_repo, chunk_repo) -> IndexingPipeline:
    """Pipeline в режиме реиндекс-эффорта (ADR-0014): floor=50, прогон #51 (резюм)."""
    from morag.run_context import RunContext
    return IndexingPipeline(
        doc_repo, chunk_repo,
        chunker=PassthroughChunker(),
        context_generator=NoopContextGenerator(),
        run_context=RunContext(run_number=51, indexed_at='2026-05-18T10:00:00+00:00'),
        reindex_floor=50,
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

    async def test_document_committed_after_chunks(self, pipeline, doc_repo, chunk_repo):
        """Точка документа коммитится ПОСЛЕ чанков (replace-not-delete, ADR-0014).

        doc point — точка коммита: пока чанки не записаны, документ для резюма
        считается недоделанным.
        """
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

        assert call_order == ['chunk_upsert', 'doc_upsert']

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
            async def process(self, chunk, document):
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


# ---------------------------------------------------------------------------
# Run versioning + embedder fingerprint stamping (см. ADR-0012, секции 4-5)
# ---------------------------------------------------------------------------

class TestPayloadStamping:

    async def test_doc_payload_has_run_number_and_indexed_at(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await stamped_pipeline.run(source)

        upserted = doc_repo.upsert.call_args[0][0]
        assert upserted.payload['run_number'] == 42
        assert upserted.payload['indexed_at'] == '2026-05-02T10:00:00+00:00'
        assert upserted.payload['embedder_fingerprint'] == 'fp-v1-abc'

    async def test_doc_version_starts_at_1_for_new_doc(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await stamped_pipeline.run(source)

        upserted = doc_repo.upsert.call_args[0][0]
        assert upserted.payload['version'] == 1

    async def test_doc_version_increments_on_reindex(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        # Existing doc с версией 5
        old_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        existing = make_document(updated_at=old_ts)
        existing.payload = {'version': 5}
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=new_ts)])

        await stamped_pipeline.run(source)

        upserted = doc_repo.upsert.call_args[0][0]
        assert upserted.payload['version'] == 6

    async def test_doc_version_unchanged_on_pure_reindex(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        """Чистый реиндекс того же контента (ADR-0014): version не бампается.

        version — ось контента: реиндекс по причине, не связанной с правкой
        документа (здесь — смена embedder'а), version не двигает.
        """
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = make_document(updated_at=ts)
        # Тот же updated_at, но старый fingerprint → reindex без смены контента
        existing.payload = {'version': 5, 'embedder_fingerprint': 'fp-v0-old'}
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=ts)])

        await stamped_pipeline.run(source)

        source.load_one.assert_called_once()  # реиндекс действительно произошёл
        upserted = doc_repo.upsert.call_args[0][0]
        assert upserted.payload['version'] == 5  # не 6 — контент не менялся

    async def test_chunk_payload_inherits_version_and_run(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document(text='# hi\n\nbody.')])

        await stamped_pipeline.run(source)

        chunks = chunk_repo.upsert_batch.call_args[0][0]
        assert all(c.payload['run_number'] == 42 for c in chunks)
        assert all(c.payload['indexed_at'] == '2026-05-02T10:00:00+00:00' for c in chunks)
        assert all(c.payload['version'] == 1 for c in chunks)
        assert all(c.payload['embedder_fingerprint'] == 'fp-v1-abc' for c in chunks)

    async def test_no_stamping_without_run_context(
        self, pipeline, doc_repo, chunk_repo,
    ):
        """Pipeline без RunContext (CLI-режим без env) — поля не пишутся."""
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        upserted = doc_repo.upsert.call_args[0][0]
        assert 'run_number' not in upserted.payload
        assert 'indexed_at' not in upserted.payload
        assert 'embedder_fingerprint' not in upserted.payload


class TestEmbedderFingerprint:

    async def test_skip_when_fingerprint_matches(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = make_document(updated_at=ts)
        existing.payload = {'embedder_fingerprint': 'fp-v1-abc'}
        doc_repo.get_by_id.return_value = existing
        chunk_repo.get_index_status.return_value = (3, 3)

        source = MagicMock(spec=Source)
        source.source_type = 'markdown'
        source.get_metadata.return_value = [
            make_stub(updated_at=ts),
        ]

        await stamped_pipeline.run(source)

        # fingerprint match → skip → load_one не вызывается
        source.load_one.assert_not_called()

    async def test_reindex_when_fingerprint_mismatches(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = make_document(updated_at=ts)
        existing.payload = {'embedder_fingerprint': 'fp-v0-old'}  # старый fingerprint
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(updated_at=ts)])

        await stamped_pipeline.run(source)

        # mismatch → reindex (load_one вызывается, doc upsert'ится)
        source.load_one.assert_called_once()
        doc_repo.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# replace-not-delete (ADR-0014)
# ---------------------------------------------------------------------------

class TestReplaceNotDelete:
    """Swap-delete вместо delete-upfront: корпус остаётся запрашиваемым."""

    async def test_swap_delete_called_with_run_number(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        """С RunContext старые чанки сметаются swap-delete'ом по run_number,
        delete-upfront не вызывается."""
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document(text='# hi\n\nbody.')])

        await stamped_pipeline.run(source)

        chunk_repo.delete_stale_chunks.assert_called_once_with('test.md', 42)
        chunk_repo.delete_by_doc_id.assert_not_called()

    async def test_swap_delete_after_chunk_upsert(
        self, stamped_pipeline, doc_repo, chunk_repo,
    ):
        """Порядок: вставка новых чанков → swap-delete старых → коммит документа."""
        doc_repo.get_by_id.return_value = None
        call_order: list[str] = []
        chunk_repo.upsert_batch.side_effect = lambda c: call_order.append('chunk_upsert')
        chunk_repo.delete_stale_chunks.side_effect = (
            lambda d, r: call_order.append('swap_delete')
        )
        doc_repo.upsert.side_effect = lambda d: call_order.append('doc_commit')

        source = MagicMock(spec=Source)
        setup_source(source, [make_document(text='# hi\n\nbody.')])

        await stamped_pipeline.run(source)

        assert call_order == ['chunk_upsert', 'swap_delete', 'doc_commit']

    async def test_legacy_delete_upfront_without_run_context(
        self, pipeline, doc_repo, chunk_repo,
    ):
        """Без RunContext — fallback на delete-upfront (swap невозможен)."""
        doc_repo.get_by_id.return_value = None
        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await pipeline.run(source)

        chunk_repo.delete_by_doc_id.assert_called_once_with('test.md')
        chunk_repo.delete_stale_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Реиндекс-эффорт и резюм (ADR-0014)
# ---------------------------------------------------------------------------

class TestReindexEffort:
    """При reindex_floor актуальность определяется run_number — резюм по floor."""

    async def test_skips_doc_already_at_floor(
        self, reindex_pipeline, doc_repo, chunk_repo,
    ):
        """Документ с run_number >= floor уже обработан эффортом → пропуск."""
        existing = make_document()
        existing.payload = {'run_number': 50}  # == floor → готов
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        source.get_metadata.return_value = [make_stub()]

        await reindex_pipeline.run(source)

        source.load_one.assert_not_called()

    async def test_processes_doc_below_floor(
        self, reindex_pipeline, doc_repo, chunk_repo,
    ):
        """Документ с run_number < floor эффортом ещё не тронут → реиндекс."""
        existing = make_document()
        existing.payload = {'run_number': 49}  # < floor → не готов
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await reindex_pipeline.run(source)

        source.load_one.assert_called_once()

    async def test_processes_doc_without_run_number(
        self, reindex_pipeline, doc_repo, chunk_repo,
    ):
        """Документ старого индекса без run_number → реиндекс (floor не определить)."""
        existing = make_document()
        existing.payload = {}
        doc_repo.get_by_id.return_value = existing

        source = MagicMock(spec=Source)
        setup_source(source, [make_document()])

        await reindex_pipeline.run(source)

        source.load_one.assert_called_once()


# ---------------------------------------------------------------------------
# Full-sync orphan scope — должен скоупиться per-instance
# ---------------------------------------------------------------------------

class TestFullSyncScope:
    """При нескольких инстансах одного source_type (multi-Confluence) full-sync
    второго не должен сметать документы первого как orphans."""

    async def test_orphan_filter_uses_kind_and_name(self, pipeline, doc_repo, chunk_repo):
        """get_ids_by_source_instance вызывается с (source_type, kind, name)."""
        source = MagicMock(spec=Source)
        source.source_type = 'confluence'
        source.kind = 'confluence'
        source.name = 'corp'
        source.get_metadata.return_value = []

        await pipeline.run(source)

        doc_repo.get_ids_by_source_instance.assert_called_once_with(
            'confluence', 'confluence', 'corp',
        )

    async def test_second_instance_does_not_delete_first(self, pipeline, doc_repo, chunk_repo):
        """Прогон Confluence B не должен трогать документы Confluence A."""
        source_b = MagicMock(spec=Source)
        source_b.source_type = 'confluence'
        source_b.kind = 'confluence'
        source_b.name = 'B'
        source_b.get_metadata.return_value = [make_stub('confluence:B:1', source_type='confluence')]

        # Репо возвращает только ids своего инстанса (новая семантика метода) —
        # ids инстанса A в выборку не попадают, orphan-delete их не коснётся.
        doc_repo.get_ids_by_source_instance.return_value = {'confluence:B:1'}
        doc_repo.get_by_id.return_value = None

        await pipeline.run(source_b)

        for call in doc_repo.cascade_delete.call_args_list:
            doc_id = call.args[0]
            assert not doc_id.startswith('confluence:A:'), (
                f'orphan-delete захватил документ соседнего инстанса: {doc_id}'
            )
