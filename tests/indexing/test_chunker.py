import pytest

from morag.indexing.chunker import Chunker, LLMChunker, PassthroughChunker


class TestPassthroughChunker:
    def test_is_chunker(self):
        assert isinstance(PassthroughChunker(), Chunker)

    def test_returns_block_as_single_chunk(self):
        chunker = PassthroughChunker()
        result = chunker.chunk('Текст блока')
        assert result == ['Текст блока']

    def test_returns_list_of_one(self):
        chunker = PassthroughChunker()
        result = chunker.chunk('Любой текст')
        assert len(result) == 1

    def test_preserves_text_exactly(self):
        text = '# Заголовок\n\nПервый абзац.\n\nВторой абзац.'
        chunker = PassthroughChunker()
        assert chunker.chunk(text)[0] == text

    def test_empty_string(self):
        chunker = PassthroughChunker()
        result = chunker.chunk('')
        assert result == ['']


class TestLLMChunker:
    def test_is_chunker(self):
        assert isinstance(LLMChunker(), Chunker)

    def test_raises_not_implemented(self):
        chunker = LLMChunker()
        with pytest.raises(NotImplementedError):
            chunker.chunk('Текст')
