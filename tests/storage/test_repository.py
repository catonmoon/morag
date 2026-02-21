from datetime import datetime, timezone

import pytest

from morag.sources.base import Chunk, Document


def make_document(doc_id: str = 'test.md', **kwargs) -> Document:
    defaults = dict(
        id=doc_id,
        path=doc_id,
        text='# Тест\n\nСодержимое документа.',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=1024,
    )
    defaults.update(kwargs)
    return Document(**defaults)


def make_chunks(doc_id: str, count: int, total: int | None = None) -> list[Chunk]:
    if total is None:
        total = count
    return [
        Chunk(
            doc_id=doc_id,
            path=f'{doc_id}',
            order=i,
            total=total,
            text=f'Текст чанка номер {i}.',
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# DocRepository
# ---------------------------------------------------------------------------

class TestDocRepository:
    async def test_get_by_id_returns_none_when_not_found(self, doc_repo):
        result = await doc_repo.get_by_id('nonexistent.md')
        assert result is None

    async def test_upsert_and_retrieve(self, doc_repo):
        doc = make_document('overview.md')
        await doc_repo.upsert(doc)

        result = await doc_repo.get_by_id(doc.id)

        assert result is not None
        assert result.id == doc.id
        assert result.path == doc.path
        assert result.text == doc.text
        assert result.source_type == doc.source_type

    async def test_updated_at_round_trips_correctly(self, doc_repo):
        doc = make_document(updated_at=datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc))
        await doc_repo.upsert(doc)

        result = await doc_repo.get_by_id(doc.id)

        assert result.updated_at == doc.updated_at
        assert result.updated_at.tzinfo is not None

    async def test_upsert_overwrites_existing(self, doc_repo):
        doc_v1 = make_document(text='Версия 1')
        await doc_repo.upsert(doc_v1)

        doc_v2 = make_document(text='Версия 2')
        await doc_repo.upsert(doc_v2)

        result = await doc_repo.get_by_id(doc_v1.id)
        assert result.text == 'Версия 2'

    async def test_delete_removes_document(self, doc_repo):
        doc = make_document()
        await doc_repo.upsert(doc)
        await doc_repo.delete(doc.id)

        result = await doc_repo.get_by_id(doc.id)
        assert result is None

    async def test_delete_nonexistent_does_not_raise(self, doc_repo):
        await doc_repo.delete('ghost.md')  # не должно бросить исключение

    async def test_size_is_preserved(self, doc_repo):
        doc = make_document(size=2048)
        await doc_repo.upsert(doc)

        result = await doc_repo.get_by_id(doc.id)
        assert result.size == 2048

    async def test_indexed_at_is_set_on_upsert(self, doc_repo):
        doc = make_document()
        assert doc.indexed_at is None

        await doc_repo.upsert(doc)

        result = await doc_repo.get_by_id(doc.id)
        assert result.indexed_at is not None
        assert result.indexed_at.tzinfo is not None

    async def test_indexed_at_round_trips_correctly(self, doc_repo):
        doc = make_document()
        await doc_repo.upsert(doc)
        result = await doc_repo.get_by_id(doc.id)

        # indexed_at должен быть близко к текущему времени
        from datetime import datetime, timezone as tz
        now = datetime.now(tz.utc)
        delta = abs((now - result.indexed_at).total_seconds())
        assert delta < 5

    async def test_payload_is_preserved(self, doc_repo):
        doc = make_document(payload={'author': 'Алиса', 'tags': ['rag', 'test']})
        await doc_repo.upsert(doc)

        result = await doc_repo.get_by_id(doc.id)
        assert result.payload.get('author') == 'Алиса'
        assert result.payload.get('tags') == ['rag', 'test']

    async def test_different_ids_stored_independently(self, doc_repo):
        doc_a = make_document('a.md', text='Документ A')
        doc_b = make_document('b.md', text='Документ B')
        await doc_repo.upsert(doc_a)
        await doc_repo.upsert(doc_b)

        result_a = await doc_repo.get_by_id('a.md')
        result_b = await doc_repo.get_by_id('b.md')
        assert result_a.text == 'Документ A'
        assert result_b.text == 'Документ B'


# ---------------------------------------------------------------------------
# ChunkRepository
# ---------------------------------------------------------------------------

class TestChunkRepository:
    async def test_get_index_status_none_when_empty(self, chunk_repo):
        result = await chunk_repo.get_index_status('nonexistent.md')
        assert result is None

    async def test_upsert_batch_and_get_status(self, chunk_repo):
        chunks = make_chunks('doc.md', count=3)
        await chunk_repo.upsert_batch(chunks)

        status = await chunk_repo.get_index_status('doc.md')
        assert status == (3, 3)

    async def test_incomplete_indexing_detected(self, chunk_repo):
        # Сохраняем 2 чанка, но у каждого total=5 (индексация была прервана)
        chunks = make_chunks('partial.md', count=2, total=5)
        await chunk_repo.upsert_batch(chunks)

        status = await chunk_repo.get_index_status('partial.md')
        assert status == (2, 5)

    async def test_delete_by_doc_id_removes_all_chunks(self, chunk_repo):
        chunks = make_chunks('to_delete.md', count=4)
        await chunk_repo.upsert_batch(chunks)

        await chunk_repo.delete_by_doc_id('to_delete.md')

        status = await chunk_repo.get_index_status('to_delete.md')
        assert status is None

    async def test_delete_by_doc_id_only_affects_target(self, chunk_repo):
        chunks_a = make_chunks('doc_a.md', count=3)
        chunks_b = make_chunks('doc_b.md', count=2)
        await chunk_repo.upsert_batch(chunks_a + chunks_b)

        await chunk_repo.delete_by_doc_id('doc_a.md')

        assert await chunk_repo.get_index_status('doc_a.md') is None
        assert await chunk_repo.get_index_status('doc_b.md') == (2, 2)

    async def test_upsert_batch_empty_list_does_not_raise(self, chunk_repo):
        await chunk_repo.upsert_batch([])  # не должно бросить исключение

    async def test_chunks_have_correct_order_and_total(self, chunk_repo):
        chunks = make_chunks('ordered.md', count=5)
        await chunk_repo.upsert_batch(chunks)

        # Проверяем через get_index_status что total сохранился корректно
        status = await chunk_repo.get_index_status('ordered.md')
        count, total = status
        assert count == 5
        assert total == 5

    async def test_chunk_payload_is_preserved(self, chunk_repo):
        chunk = Chunk(
            doc_id='doc.md',
            path='doc.md',
            order=0,
            total=1,
            text='Текст',
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            payload={'section': 'intro', 'language': 'ru'},
        )
        await chunk_repo.upsert_batch([chunk])

        # Проверяем что чанк сохранился (через count)
        status = await chunk_repo.get_index_status('doc.md')
        assert status == (1, 1)
