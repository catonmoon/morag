from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.llm.client import LLMClient


def make_completion(content: str):
    """Build a fake ChatCompletion response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture
def mock_openai():
    """Patch AsyncOpenAI so no real HTTP calls are made."""
    with patch('morag.llm.client.AsyncOpenAI') as cls:
        instance = AsyncMock()
        instance.chat = AsyncMock()
        instance.chat.completions = AsyncMock()
        cls.return_value = instance
        yield instance


@pytest.fixture
def client(mock_openai) -> LLMClient:
    return LLMClient(base_url='http://localhost:11434/v1', model='llama3.2')


class TestLLMClient:
    async def test_complete_returns_text(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('Hello!')
        result = await client.complete([{'role': 'user', 'content': 'Hi'}])
        assert result == 'Hello!'

    async def test_complete_passes_messages(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('ok')
        messages = [{'role': 'system', 'content': 'You are helpful'}, {'role': 'user', 'content': 'Hi'}]
        await client.complete(messages)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['messages'] == messages

    async def test_complete_passes_model(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('ok')
        await client.complete([{'role': 'user', 'content': 'Hi'}])

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'llama3.2'

    async def test_complete_returns_empty_string_when_content_none(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion(None)
        result = await client.complete([{'role': 'user', 'content': 'Hi'}])
        assert result == ''

    async def test_complete_json_parses_json(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('{"key": "value"}')
        result = await client.complete_json([{'role': 'user', 'content': 'return json'}])
        assert result == {'key': 'value'}

    async def test_complete_json_requests_json_format(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('{}')
        await client.complete_json([{'role': 'user', 'content': 'return json'}])

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['response_format'] == {'type': 'json_object'}

    async def test_complete_json_raises_on_invalid_json(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion('not json at all')
        with pytest.raises(ValueError, match='invalid JSON'):
            await client.complete_json([{'role': 'user', 'content': 'return json'}])

    async def test_complete_json_returns_empty_dict_on_empty_content(self, client, mock_openai):
        mock_openai.chat.completions.create.return_value = make_completion(None)
        result = await client.complete_json([{'role': 'user', 'content': 'return json'}])
        assert result == {}

    async def test_client_passes_base_url_and_api_key(self):
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            cls.return_value = AsyncMock()
            LLMClient(base_url='http://example.com/v1', model='gpt-4', api_key='sk-test')
            _, kwargs = cls.call_args
            assert kwargs.get('base_url') == 'http://example.com/v1'
            assert kwargs.get('api_key') == 'sk-test'

    async def test_default_api_key_is_ollama(self):
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            cls.return_value = AsyncMock()
            LLMClient(base_url='http://localhost:11434/v1', model='llama3.2')
            _, kwargs = cls.call_args
            assert kwargs.get('api_key') == 'ollama'
