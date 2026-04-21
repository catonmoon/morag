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
    client = AsyncMock()
    client.context_window = 32768  # real LLMClient exposes as property
    return client


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
        doc = 'First page.\n\nSecond page.\n\nThird page.'
        chunk = 'Second page.'
        offset = doc.index('Second page.')
        await gen.generate(doc, chunk, char_offset=offset)
        messages = mock_client.complete.call_args[0][0]
        prompt = messages[0]['content']
        assert 'Second page' in prompt


# ---------------------------------------------------------------------------
# _extract_window
# ---------------------------------------------------------------------------

class TestExtractWindow:
    def test_offset_found(self):
        counter = TiktokenCounter()
        doc = 'First.\n\nSecond.\n\nThird.'
        offset = doc.index('Second.')
        result = _extract_window(doc, offset, 1000, counter)
        assert 'Second' in result

    def test_zero_offset_returns_from_start(self):
        counter = TiktokenCounter()
        doc = 'First.\n\nSecond.\n\nThird.'
        result = _extract_window(doc, 0, 1000, counter)
        assert 'First' in result

    def test_window_smaller_than_doc(self):
        counter = TiktokenCounter()
        pages = [f'Page {i} content. ' * 50 for i in range(1, 11)]
        doc = '\n\n'.join(pages)
        offset = doc.index('Page 5')
        result = _extract_window(doc, offset, 100, counter)
        assert len(result) < len(doc)
        assert 'Page 5' in result

    def test_window_larger_than_doc(self):
        counter = TiktokenCounter()
        doc = 'Short doc.'
        result = _extract_window(doc, 0, 10000, counter)
        assert result == doc

    def test_offset_near_end(self):
        counter = TiktokenCounter()
        doc = 'A. ' * 100 + 'TARGET. ' + 'B. ' * 10
        offset = doc.index('TARGET.')
        result = _extract_window(doc, offset, 50, counter)
        assert 'TARGET' in result

    def test_window_centered_around_offset(self):
        """Окно примерно центрировано вокруг offset."""
        counter = TiktokenCounter()
        before = 'Before. ' * 50
        target = 'TARGET_TEXT. '
        after = 'After. ' * 50
        doc = before + target + after
        offset = doc.index('TARGET_TEXT')
        result = _extract_window(doc, offset, 40, counter)
        assert 'TARGET_TEXT' in result
        # Должен содержать и текст до, и текст после
        assert 'Before' in result
        assert 'After' in result

    def test_offset_at_very_start(self):
        """Offset 0 — окно начинается с начала документа."""
        counter = TiktokenCounter()
        doc = 'Start. ' * 100
        result = _extract_window(doc, 0, 20, counter)
        assert result.startswith('Start')

    def test_offset_beyond_doc_length(self):
        """Offset за пределами документа — не падает."""
        counter = TiktokenCounter()
        doc = 'Short doc.'
        result = _extract_window(doc, 9999, 100, counter)
        assert result  # не пустой

    def test_negative_offset_treated_as_zero(self):
        """Отрицательный offset — как 0."""
        counter = TiktokenCounter()
        doc = 'Some text here.'
        result = _extract_window(doc, -5, 100, counter)
        assert 'Some text' in result
