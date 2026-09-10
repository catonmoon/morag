import asyncio
from datetime import datetime, timezone

import pytest

from morag.indexing.embedder import Embedder, SparseEmbedder
from morag import shortid
from morag.indexing.processors import (
    ChunkProcessor,
    DenseEmbeddingProcessor,
    DocumentProcessor,
    PageMarkerProcessor,
    ShortIdProcessor,
    SparseEmbeddingProcessor,
)
from morag.sources.base import Chunk, Document


class FakeSparseEmbedder(SparseEmbedder):
    """Детерминированный sparse-эмбеддер для тестов."""

    async def embed(self, text: str) -> tuple[list[int], list[float]]:
        indices = [hash(text) % 1000, (hash(text) + 1) % 1000]
        values = [0.7, 0.3]
        return indices, values

    async def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return await self.embed(text)


class FakeEmbedder(Embedder):
    """Детерминированный эмбеддер для тестов: вектор уникален для каждого текста."""

    DIM = 4

    async def embed(self, text: str) -> list[float]:
        h = float(hash(text) % 100000)
        return [h, float(len(text)), 1.0, 0.0]

    async def embed_query(self, text: str) -> list[float]:
        h = float(hash(text) % 100000)
        return [0.0, h, float(len(text)), 1.0]

    @property
    def dim(self) -> int:
        return self.DIM


def make_document() -> Document:
    return Document(
        id='test.md',
        path=['test.md'],
        text='# Тест',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
    )


def make_chunk() -> Chunk:
    return Chunk(
        doc_id='test.md',
        path=['test.md'],
        order=0,
        total=1,
        text='Текст чанка.',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# DocumentProcessor
# ---------------------------------------------------------------------------

class TestDocumentProcessor:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            DocumentProcessor()  # нельзя создать напрямую

    async def test_concrete_implementation_works(self):
        class AddAuthorProcessor(DocumentProcessor):
            async def process(self, document: Document) -> Document:
                document.payload['author'] = 'Тест'
                return document

        processor = AddAuthorProcessor()
        doc = make_document()
        result = await processor.process(doc)
        assert result.payload.get('author') == 'Тест'

    async def test_processor_chain_applies_sequentially(self):
        class TagProcessor(DocumentProcessor):
            def __init__(self, tag: str):
                self.tag = tag

            async def process(self, document: Document) -> Document:
                tags = document.payload.get('tags', [])
                document.payload['tags'] = [*tags, self.tag]
                return document

        chain = [TagProcessor('rag'), TagProcessor('test')]
        doc = make_document()
        for p in chain:
            doc = await p.process(doc)

        assert doc.payload['tags'] == ['rag', 'test']


# ---------------------------------------------------------------------------
# ChunkProcessor
# ---------------------------------------------------------------------------

class TestChunkProcessor:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ChunkProcessor()  # нельзя создать напрямую

    async def test_concrete_implementation_works(self):
        class LengthProcessor(ChunkProcessor):
            async def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.payload['char_count'] = len(chunk.text)
                return chunk

        processor = LengthProcessor()
        chunk = make_chunk()
        doc = make_document()
        result = await processor.process(chunk, doc)
        assert result.payload.get('char_count') == len(chunk.text)

    async def test_processor_can_add_vector(self):
        class FakeEmbedProcessor(ChunkProcessor):
            async def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.vectors['text'] = [0.1, 0.2, 0.3]
                return chunk

        processor = FakeEmbedProcessor()
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        assert 'text' in result.vectors
        assert result.vectors['text'] == [0.1, 0.2, 0.3]

    async def test_processor_receives_document_context(self):
        """Процессор может использовать данные документа при обработке чанка."""
        class SourceTypeProcessor(ChunkProcessor):
            async def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.payload['source_type'] = document.source_type
                return chunk

        processor = SourceTypeProcessor()
        doc = make_document()
        chunk = make_chunk()
        result = await processor.process(chunk, doc)
        assert result.payload.get('source_type') == 'markdown'


# ---------------------------------------------------------------------------
# DenseEmbeddingProcessor
# ---------------------------------------------------------------------------

class TestDenseEmbeddingProcessor:
    def test_is_chunk_processor(self):
        assert isinstance(DenseEmbeddingProcessor(FakeEmbedder()), ChunkProcessor)

    async def test_adds_full_vector(self):
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        assert 'full' in result.vectors

    async def test_vector_is_list_of_floats(self):
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        vec = result.vectors['full']
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    async def test_vector_length_matches_embedder_dim(self):
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        assert len(result.vectors['full']) == embedder.dim

    async def test_full_text_includes_path(self):
        """Вектор зависит от path чанка."""
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)

        chunk_a = make_chunk()
        chunk_a.path = ['docs/guide.md']
        chunk_b = make_chunk()
        chunk_b.path = ['docs/faq.md']

        result_a = await processor.process(chunk_a, make_document())
        result_b = await processor.process(chunk_b, make_document())
        assert result_a.vectors['full'] != result_b.vectors['full']

    async def test_full_text_includes_context(self):
        """Вектор зависит от context чанка."""
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)

        chunk_a = make_chunk()
        chunk_a.context = 'Контекст А'
        chunk_b = make_chunk()
        chunk_b.context = 'Контекст Б'

        result_a = await processor.process(chunk_a, make_document())
        result_b = await processor.process(chunk_b, make_document())
        assert result_a.vectors['full'] != result_b.vectors['full']

    async def test_does_not_overwrite_other_vectors(self):
        """Процессор не затирает уже существующие векторы."""
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        chunk.vectors['existing'] = [9.0, 8.0, 7.0, 6.0]
        result = await processor.process(chunk, make_document())
        assert result.vectors['existing'] == [9.0, 8.0, 7.0, 6.0]
        assert 'full' in result.vectors


# ---------------------------------------------------------------------------
# SparseEmbeddingProcessor
# ---------------------------------------------------------------------------

class TestSparseEmbeddingProcessor:
    def test_is_chunk_processor(self):
        assert isinstance(SparseEmbeddingProcessor(FakeSparseEmbedder()), ChunkProcessor)

    async def test_adds_keywords_vector(self):
        processor = SparseEmbeddingProcessor(FakeSparseEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        assert 'keywords' in result.vectors

    async def test_keywords_vector_is_dict(self):
        processor = SparseEmbeddingProcessor(FakeSparseEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        vec = result.vectors['keywords']
        assert isinstance(vec, dict)

    async def test_keywords_vector_has_indices_and_values(self):
        processor = SparseEmbeddingProcessor(FakeSparseEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        vec = result.vectors['keywords']
        assert 'indices' in vec
        assert 'values' in vec

    async def test_keywords_indices_and_values_same_length(self):
        processor = SparseEmbeddingProcessor(FakeSparseEmbedder())
        chunk = make_chunk()
        result = await processor.process(chunk, make_document())
        vec = result.vectors['keywords']
        assert len(vec['indices']) == len(vec['values'])

    async def test_keywords_uses_chunk_text(self):
        """Sparse-вектор зависит от текста чанка."""
        embedder = FakeSparseEmbedder()
        processor = SparseEmbeddingProcessor(embedder)

        chunk_a = make_chunk()
        chunk_a.text = 'Первый текст'
        chunk_b = make_chunk()
        chunk_b.text = 'Второй текст'

        result_a = await processor.process(chunk_a, make_document())
        result_b = await processor.process(chunk_b, make_document())
        assert result_a.vectors['keywords'] != result_b.vectors['keywords']

    async def test_does_not_overwrite_other_vectors(self):
        """Процессор не затирает уже существующие векторы."""
        processor = SparseEmbeddingProcessor(FakeSparseEmbedder())
        chunk = make_chunk()
        chunk.vectors['full'] = [1.0, 2.0, 3.0, 4.0]
        result = await processor.process(chunk, make_document())
        assert result.vectors['full'] == [1.0, 2.0, 3.0, 4.0]
        assert 'keywords' in result.vectors


# ---------------------------------------------------------------------------
# PageMarkerProcessor
# ---------------------------------------------------------------------------

class TestPageMarkerProcessor:
    async def test_extracts_single_page(self):
        processor = PageMarkerProcessor()
        chunk = make_chunk()
        chunk.text = '<!-- page:3 -->\nТекст третьей страницы.'
        result = await processor.process(chunk, make_document())
        assert result.payload['pages'] == [3]
        assert '<!-- page' not in result.text
        assert result.text == 'Текст третьей страницы.'

    async def test_extracts_multiple_pages(self):
        processor = PageMarkerProcessor()
        chunk = make_chunk()
        chunk.text = '<!-- page:2 -->\nНачало.\n\n<!-- page:3 -->\nПродолжение.'
        result = await processor.process(chunk, make_document())
        assert result.payload['pages'] == [2, 3]
        assert '<!-- page' not in result.text

    async def test_no_markers_no_pages_key(self):
        processor = PageMarkerProcessor()
        chunk = make_chunk()
        chunk.text = 'Обычный текст без маркеров.'
        result = await processor.process(chunk, make_document())
        assert 'pages' not in result.payload
        assert result.text == 'Обычный текст без маркеров.'

    async def test_deduplicates_page_numbers(self):
        processor = PageMarkerProcessor()
        chunk = make_chunk()
        chunk.text = '<!-- page:5 -->\nА.\n<!-- page:5 -->\nБ.'
        result = await processor.process(chunk, make_document())
        assert result.payload['pages'] == [5]

    async def test_strips_marker_newline(self):
        processor = PageMarkerProcessor()
        chunk = make_chunk()
        chunk.text = '<!-- page:1 -->\nТекст.'
        result = await processor.process(chunk, make_document())
        assert result.text == 'Текст.'


# ---------------------------------------------------------------------------
# ShortIdProcessor
# ---------------------------------------------------------------------------

def make_doc(doc_id: str, structural: bool = False) -> Document:
    return Document(
        id=doc_id,
        path=[doc_id],
        text='' if structural else '# Тест',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        structural=structural,
    )


class TestShortIdProcessor:
    async def test_writes_a_valid_code_into_payload(self):
        taken: dict[str, str] = {}
        doc = await ShortIdProcessor(taken).process(make_doc('local:demo:a/doc.md'))
        code = doc.payload['short_id']
        assert shortid.parse(code) == (shortid.DOC, shortid.body_chars('local:demo:a/doc.md', 5))
        assert taken[code] == 'local:demo:a/doc.md', 'выданный код обязан стать занятым'

    async def test_structural_document_gets_the_section_letter(self):
        """Раздел и документ различаются буквой, потому что в базе это один и тот же вид записи, а
        инструменты у них разные: `search(section_ids=…)` против `get_doc`."""
        doc = await ShortIdProcessor({}).process(make_doc('local:demo:a/', structural=True))
        assert doc.payload['short_id'][0] == shortid.SECTION

    async def test_respects_codes_already_taken(self):
        """⚠️ Занятые коды приходят из базы: документ, проиндексированный год назад, свой код не
        отдаёт. Иначе ссылки в диалоге и подсказки в отказе начали бы указывать в чужие документы."""
        doc_id = 'local:demo:a/doc.md'
        squatted = shortid.make(doc_id)
        taken = {squatted: 'local:demo:someone/else.md'}
        doc = await ShortIdProcessor(taken).process(make_doc(doc_id))
        assert doc.payload['short_id'] != squatted
        assert taken[squatted] == 'local:demo:someone/else.md'

    async def test_concurrent_documents_never_share_a_code(self):
        """Инвариант: сколько документов — столько разных кодов, даже когда их обрабатывают вместе.

        ⓘ Тест закрепляет РЕЗУЛЬТАТ, а не механизм: проверено, что он проходит и без замка вокруг
        выдачи — прерваться там негде, участок синхронный. Механизм охраняется комментарием в
        процессоре: появится `await` между выдачей и занятием — race вернётся, и этот тест его
        не поймает.
        """
        taken: dict[str, str] = {}
        processor = ShortIdProcessor(taken)
        docs = [make_doc(f'local:demo:a/doc{i}.md') for i in range(200)]
        done = await asyncio.gather(*(processor.process(d) for d in docs))
        codes = [d.payload['short_id'] for d in done]
        assert len(set(codes)) == len(codes)
        assert len(taken) == len(docs)

    async def test_length_is_configurable(self):
        doc = await ShortIdProcessor({}, length=7).process(make_doc('local:demo:a/doc.md'))
        assert len(doc.payload['short_id']) == 9  # буква типа + 7 + контрольный знак
