from unittest.mock import AsyncMock

import pytest

from morag.indexing.context import (
    ContextGenerator, LLMContextGenerator, NoopContextGenerator, _extract_window,
)
from morag.indexing.token_counter import TiktokenCounter


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

    async def test_uses_deterministic_params(self, generator, mock_client):
        from morag.indexing.context import _llm_params
        mock_client.complete.return_value = 'ok'
        await generator.generate('Документ', 'Чанк')
        _, kwargs = mock_client.complete.call_args
        assert kwargs.get('params') == _llm_params()

    async def test_raises_on_exception(self, generator, mock_client):
        mock_client.complete.side_effect = Exception('network error')
        with pytest.raises(Exception, match='network error'):
            await generator.generate('Документ', 'Чанк')

    async def test_raises_on_timeout(self, generator, mock_client):
        mock_client.complete.side_effect = RuntimeError('timeout')
        with pytest.raises(RuntimeError):
            await generator.generate('Документ', 'Чанк')

    async def test_doc_summary_in_prompt(self, mock_client):
        mock_client.complete.return_value = 'ok'
        gen = LLMContextGenerator(mock_client)
        await gen.generate('Документ', 'Чанк', doc_summary='Краткое описание документа')
        messages = mock_client.complete.call_args[0][0]
        assert 'Краткое описание документа' in messages[0]['content']

    async def test_doc_summary_empty(self, mock_client):
        mock_client.complete.return_value = 'ok'
        gen = LLMContextGenerator(mock_client)
        # Не должен упасть с пустым summary
        result = await gen.generate('Документ', 'Чанк', doc_summary='')
        assert result == 'ok'

    async def test_window_tokens(self, mock_client):
        mock_client.complete.return_value = 'ok'
        gen = LLMContextGenerator(mock_client, window_tokens=100)
        doc = '<!-- page:1 -->\nFirst page.\n<!-- page:2 -->\nSecond page.\n<!-- page:3 -->\nThird page.'
        chunk = '<!-- page:2 -->\nSecond page.'
        await gen.generate(doc, chunk)
        messages = mock_client.complete.call_args[0][0]
        prompt = messages[0]['content']
        # Окно должно содержать текст вокруг page:2
        assert 'Second page' in prompt


# ---------------------------------------------------------------------------
# _extract_window
# ---------------------------------------------------------------------------

class TestExtractWindow:
    def test_page_found(self):
        counter = TiktokenCounter()
        doc = '<!-- page:1 -->\nFirst.\n<!-- page:2 -->\nSecond.\n<!-- page:3 -->\nThird.'
        chunk = '<!-- page:2 -->\nSecond.'
        result = _extract_window(doc, chunk, 1000, counter)
        assert 'Second' in result

    def test_no_page_marker_in_chunk(self):
        counter = TiktokenCounter()
        doc = '<!-- page:1 -->\nFirst.\n<!-- page:2 -->\nSecond.'
        chunk = 'No page marker here.'
        result = _extract_window(doc, chunk, 1000, counter)
        # Fallback: truncate от начала
        assert 'First' in result

    def test_window_smaller_than_doc(self):
        counter = TiktokenCounter()
        pages = [f'<!-- page:{i} -->\n' + f'Page {i} content. ' * 50 for i in range(1, 11)]
        doc = '\n'.join(pages)
        chunk = '<!-- page:5 -->\nPage 5 content.'
        result = _extract_window(doc, chunk, 100, counter)
        # Должен содержать page 5 но не весь документ
        assert len(result) < len(doc)
        assert 'Page 5' in result

    def test_window_larger_than_doc(self):
        counter = TiktokenCounter()
        doc = '<!-- page:1 -->\nShort doc.'
        chunk = '<!-- page:1 -->\nShort doc.'
        result = _extract_window(doc, chunk, 10000, counter)
        assert result == doc

    def test_no_pages_in_doc(self):
        counter = TiktokenCounter()
        doc = 'Document without any page markers.'
        chunk = 'Some chunk text.'
        result = _extract_window(doc, chunk, 100, counter)
        assert result  # Не пустой
