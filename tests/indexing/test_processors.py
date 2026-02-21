from datetime import datetime, timezone

import pytest

from morag.indexing.processors import ChunkProcessor, DocumentProcessor
from morag.sources.base import Chunk, Document


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
