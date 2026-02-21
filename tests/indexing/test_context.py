import pytest

from morag.indexing.context import ContextGenerator, LLMContextGenerator, NoopContextGenerator


class TestNoopContextGenerator:
    async def test_is_context_generator(self):
        assert isinstance(NoopContextGenerator(), ContextGenerator)

    async def test_returns_empty_string(self):
        gen = NoopContextGenerator()
        result = await gen.generate('Текст документа', 'Текст чанка')
        assert result == ''

    async def test_ignores_doc_text(self):
        gen = NoopContextGenerator()
        r1 = await gen.generate('Один документ', 'Чанк')
        r2 = await gen.generate('Другой документ', 'Чанк')
        assert r1 == r2 == ''

    async def test_ignores_chunk_text(self):
        gen = NoopContextGenerator()
        r1 = await gen.generate('Документ', 'Чанк A')
        r2 = await gen.generate('Документ', 'Чанк B')
        assert r1 == r2 == ''


class TestLLMContextGenerator:
    async def test_is_context_generator(self):
        assert isinstance(LLMContextGenerator(), ContextGenerator)

    async def test_raises_not_implemented(self):
        gen = LLMContextGenerator()
        with pytest.raises(NotImplementedError):
            await gen.generate('Документ', 'Чанк')
