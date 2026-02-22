from datetime import datetime, timezone

import pytest

from morag.indexing.embedder import Embedder
from morag.indexing.processors import ChunkProcessor, DenseEmbeddingProcessor, DocumentProcessor
from morag.sources.base import Chunk, Document


class FakeEmbedder(Embedder):
    """Детерминированный эмбеддер для тестов: вектор уникален для каждого текста."""

    DIM = 4

    def embed(self, text: str) -> list[float]:
        h = float(hash(text) % 100000)
        return [h, float(len(text)), 1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        h = float(hash(text) % 100000)
        return [0.0, h, float(len(text)), 1.0]

    @property
    def dim(self) -> int:
        return self.DIM


def make_document() -> Document:
    return Document(
        id='test.md',
        path='test.md',
        text='# Тест',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
    )


def make_chunk() -> Chunk:
    return Chunk(
        doc_id='test.md',
        path='test.md',
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

    def test_concrete_implementation_works(self):
        class AddAuthorProcessor(DocumentProcessor):
            def process(self, document: Document) -> Document:
                document.payload['author'] = 'Тест'
                return document

        processor = AddAuthorProcessor()
        doc = make_document()
        result = processor.process(doc)
        assert result.payload.get('author') == 'Тест'

    def test_processor_chain_applies_sequentially(self):
        class TagProcessor(DocumentProcessor):
            def __init__(self, tag: str):
                self.tag = tag

            def process(self, document: Document) -> Document:
                tags = document.payload.get('tags', [])
                document.payload['tags'] = [*tags, self.tag]
                return document

        chain = [TagProcessor('rag'), TagProcessor('test')]
        doc = make_document()
        for p in chain:
            doc = p.process(doc)

        assert doc.payload['tags'] == ['rag', 'test']


# ---------------------------------------------------------------------------
# ChunkProcessor
# ---------------------------------------------------------------------------

class TestChunkProcessor:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ChunkProcessor()  # нельзя создать напрямую

    def test_concrete_implementation_works(self):
        class LengthProcessor(ChunkProcessor):
            def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.payload['char_count'] = len(chunk.text)
                return chunk

        processor = LengthProcessor()
        chunk = make_chunk()
        doc = make_document()
        result = processor.process(chunk, doc)
        assert result.payload.get('char_count') == len(chunk.text)

    def test_processor_can_add_vector(self):
        class FakeEmbedProcessor(ChunkProcessor):
            def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.vectors['text'] = [0.1, 0.2, 0.3]
                return chunk

        processor = FakeEmbedProcessor()
        chunk = make_chunk()
        result = processor.process(chunk, make_document())
        assert 'text' in result.vectors
        assert result.vectors['text'] == [0.1, 0.2, 0.3]

    def test_processor_receives_document_context(self):
        """Процессор может использовать данные документа при обработке чанка."""
        class SourceTypeProcessor(ChunkProcessor):
            def process(self, chunk: Chunk, document: Document) -> Chunk:
                chunk.payload['source_type'] = document.source_type
                return chunk

        processor = SourceTypeProcessor()
        doc = make_document()
        chunk = make_chunk()
        result = processor.process(chunk, doc)
        assert result.payload.get('source_type') == 'markdown'


# ---------------------------------------------------------------------------
# DenseEmbeddingProcessor
# ---------------------------------------------------------------------------

class TestDenseEmbeddingProcessor:
    def test_is_chunk_processor(self):
        assert isinstance(DenseEmbeddingProcessor(FakeEmbedder()), ChunkProcessor)

    def test_adds_full_vector(self):
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        result = processor.process(chunk, make_document())
        assert 'full' in result.vectors

    def test_vector_is_list_of_floats(self):
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        result = processor.process(chunk, make_document())
        vec = result.vectors['full']
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_vector_length_matches_embedder_dim(self):
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)
        chunk = make_chunk()
        result = processor.process(chunk, make_document())
        assert len(result.vectors['full']) == embedder.dim

    def test_full_text_includes_path(self):
        """Вектор зависит от path чанка."""
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)

        chunk_a = make_chunk()
        chunk_a.path = 'docs/guide.md'
        chunk_b = make_chunk()
        chunk_b.path = 'docs/faq.md'

        result_a = processor.process(chunk_a, make_document())
        result_b = processor.process(chunk_b, make_document())
        assert result_a.vectors['full'] != result_b.vectors['full']

    def test_full_text_includes_context(self):
        """Вектор зависит от context чанка."""
        embedder = FakeEmbedder()
        processor = DenseEmbeddingProcessor(embedder)

        chunk_a = make_chunk()
        chunk_a.context = 'Контекст А'
        chunk_b = make_chunk()
        chunk_b.context = 'Контекст Б'

        result_a = processor.process(chunk_a, make_document())
        result_b = processor.process(chunk_b, make_document())
        assert result_a.vectors['full'] != result_b.vectors['full']

    def test_does_not_overwrite_other_vectors(self):
        """Процессор не затирает уже существующие векторы."""
        processor = DenseEmbeddingProcessor(FakeEmbedder())
        chunk = make_chunk()
        chunk.vectors['existing'] = [9.0, 8.0, 7.0, 6.0]
        result = processor.process(chunk, make_document())
        assert result.vectors['existing'] == [9.0, 8.0, 7.0, 6.0]
        assert 'full' in result.vectors
