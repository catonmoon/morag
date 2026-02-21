from unittest.mock import AsyncMock

import pytest

from morag.indexing.chunker import Chunker, LLMChunker, PassthroughChunker


class TestPassthroughChunker:
    def test_is_chunker(self):
        assert isinstance(PassthroughChunker(), Chunker)

    async def test_returns_block_as_single_chunk(self):
        result = await PassthroughChunker().chunk('Текст блока')
        assert result == ['Текст блока']

    async def test_returns_list_of_one(self):
        result = await PassthroughChunker().chunk('Любой текст')
        assert len(result) == 1

    async def test_preserves_text_exactly(self):
        text = '# Заголовок\n\nПервый абзац.\n\nВторой абзац.'
        assert (await PassthroughChunker().chunk(text))[0] == text

    async def test_empty_string(self):
        result = await PassthroughChunker().chunk('')
        assert result == ['']


# ---------------------------------------------------------------------------
# LLMChunker
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def chunker(mock_client) -> LLMChunker:
    return LLMChunker(mock_client)


class TestLLMChunker:
    def test_is_chunker(self, mock_client):
        assert isinstance(LLMChunker(mock_client), Chunker)

    async def test_returns_chunks_from_llm(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['Часть A', 'Часть B', 'Часть C']}
        result = await chunker.chunk('Длинный текст')
        assert result == ['Часть A', 'Часть B', 'Часть C']

    async def test_passes_block_to_llm(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['ok']}
        block = 'Мой текстовый блок'
        await chunker.chunk(block)
        messages = mock_client.complete_json.call_args[0][0]
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == block

    async def test_fallback_on_invalid_json(self, chunker, mock_client):
        mock_client.complete_json.side_effect = ValueError('invalid JSON')
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_missing(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'result': 'something else'}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_empty_list(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': []}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_not_a_list(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': 'not a list'}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_filters_out_empty_strings(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['Чанк A', '   ', 'Чанк B']}
        result = await chunker.chunk('Текст')
        assert result == ['Чанк A', 'Чанк B']

    async def test_system_prompt_is_sent(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['ok']}
        await chunker.chunk('Текст')
        messages = mock_client.complete_json.call_args[0][0]
        system_message = next(m for m in messages if m['role'] == 'system')
        assert len(system_message['content']) > 0
