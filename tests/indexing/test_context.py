from unittest.mock import AsyncMock

import pytest

from morag.indexing.context import ContextGenerator, LLMContextGenerator, NoopContextGenerator


class TestNoopContextGenerator:
    async def test_is_context_generator(self):
        assert isinstance(NoopContextGenerator(), ContextGenerator)

    async def test_returns_empty_string(self):
        result = await NoopContextGenerator().generate('Текст документа', 'Текст чанка')
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


# ---------------------------------------------------------------------------
# LLMContextGenerator
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def generator(mock_client) -> LLMContextGenerator:
    return LLMContextGenerator(mock_client)


class TestLLMContextGenerator:
    def test_is_context_generator(self, mock_client):
        assert isinstance(LLMContextGenerator(mock_client), ContextGenerator)

    async def test_returns_llm_response(self, generator, mock_client):
        mock_client.complete.return_value = 'Документ описывает архитектуру LLM.'
        result = await generator.generate('Полный текст документа', 'Текст чанка')
        assert result == 'Документ описывает архитектуру LLM.'

    async def test_prompt_contains_doc_text(self, generator, mock_client):
        mock_client.complete.return_value = 'ok'
        await generator.generate('Уникальный текст документа', 'Чанк')
        messages = mock_client.complete.call_args[0][0]
        assert 'Уникальный текст документа' in messages[0]['content']

    async def test_prompt_contains_chunk_text(self, generator, mock_client):
        mock_client.complete.return_value = 'ok'
        await generator.generate('Документ', 'Уникальный текст чанка')
        messages = mock_client.complete.call_args[0][0]
        assert 'Уникальный текст чанка' in messages[0]['content']

    async def test_uses_temperature_03(self, generator, mock_client):
        mock_client.complete.return_value = 'ok'
        await generator.generate('Документ', 'Чанк')
        _, kwargs = mock_client.complete.call_args
        assert kwargs.get('temperature') == 0.3

    async def test_fallback_on_exception(self, generator, mock_client):
        mock_client.complete.side_effect = Exception('network error')
        result = await generator.generate('Документ', 'Чанк')
        assert result == ''

    async def test_fallback_returns_empty_string_not_raises(self, generator, mock_client):
        mock_client.complete.side_effect = RuntimeError('timeout')
        result = await generator.generate('Документ', 'Чанк')
        assert isinstance(result, str)
