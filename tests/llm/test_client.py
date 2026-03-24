from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.llm.client import GenerationParams, LLMClient


def make_completion(content: str, reasoning: str | None = None):
    """Build a fake ChatCompletion response object."""
    message = MagicMock()
    message.content = content
    message.reasoning = reasoning
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
        schema = {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}
        mock_openai.chat.completions.create.return_value = make_completion('{"key": "value"}')
        result = await client.complete_json([{'role': 'user', 'content': 'return json'}], schema=schema)
        assert result == {'key': 'value'}

    async def test_complete_json_requests_json_format(self, client, mock_openai):
        schema = {'type': 'object', 'properties': {}, 'required': []}
        mock_openai.chat.completions.create.return_value = make_completion('{}')
        await client.complete_json([{'role': 'user', 'content': 'return json'}], schema=schema, schema_name='test')

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert call_kwargs['response_format']['type'] == 'json_schema'
        assert call_kwargs['response_format']['json_schema']['name'] == 'test'
        assert call_kwargs['response_format']['json_schema']['schema'] == schema

    async def test_complete_json_raises_on_invalid_json(self, client, mock_openai):
        schema = {'type': 'object', 'properties': {}, 'required': []}
        mock_openai.chat.completions.create.return_value = make_completion('not json at all')
        with pytest.raises(ValueError, match='invalid JSON'):
            await client.complete_json([{'role': 'user', 'content': 'return json'}], schema=schema)

    async def test_complete_json_returns_empty_dict_on_empty_content(self, client, mock_openai):
        schema = {'type': 'object', 'properties': {}, 'required': []}
        mock_openai.chat.completions.create.return_value = make_completion(None)
        result = await client.complete_json([{'role': 'user', 'content': 'return json'}], schema=schema)
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


# ---------------------------------------------------------------------------
# Model wait
# ---------------------------------------------------------------------------

def make_empty_completion():
    """Build a fake ChatCompletion with choices=None (model reloading)."""
    completion = MagicMock()
    completion.choices = None
    return completion


class TestModelWait:
    """Тесты ожидания перезагрузки модели в LLMClient."""

    async def test_waits_and_retries_on_model_not_found(self):
        """При model_not_found ждёт и повторяет; если модель загрузилась — успех."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            cls.return_value = instance

            call_count = 0

            async def side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception('400 - Model not found')
                return make_completion('Hello!')

            instance.chat.completions.create.side_effect = side_effect

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=10, model_wait_retries=3,
                )
                result = await client.complete([{'role': 'user', 'content': 'Hi'}])

            assert result == 'Hello!'
            assert call_count == 3
            assert mock_sleep.call_count == 2
            # С jitter: sleep вызывается с wait_seconds + random(0, wait*0.5)
            for call in mock_sleep.call_args_list:
                wait_val = call[0][0]
                assert 10 <= wait_val <= 15  # 10 + jitter(0, 5)

    async def test_waits_on_empty_response(self):
        """Wait срабатывает при пустом ответе (choices is None)."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            cls.return_value = instance

            call_count = 0

            async def side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return make_empty_completion()
                return make_completion('ok')

            instance.chat.completions.create.side_effect = side_effect

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=10, model_wait_retries=3,
                )
                result = await client.complete([{'role': 'user', 'content': 'Hi'}])

            assert result == 'ok'
            assert call_count == 2
            assert mock_sleep.call_count == 1

    async def test_model_wait_exhausted_raises(self):
        """Если все попытки ожидания исчерпаны — пробрасывает ошибку."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            instance.chat.completions.create.side_effect = Exception('400 - Model not found')
            cls.return_value = instance

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock):
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=5, model_wait_retries=2,
                )
                with pytest.raises(Exception, match='Model not found'):
                    await client.complete([{'role': 'user', 'content': 'Hi'}])

            # 1 initial + 2 wait attempts = 3
            assert instance.chat.completions.create.call_count == 3

    async def test_no_wait_when_disabled(self):
        """Если model_wait_retries=0 — не ждём, сразу пробрасываем ошибку."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            instance.chat.completions.create.side_effect = Exception('400 - Model not found')
            cls.return_value = instance

            client = LLMClient(
                base_url='http://localhost/v1', model='test',
            )
            with pytest.raises(Exception, match='Model not found'):
                await client.complete([{'role': 'user', 'content': 'Hi'}])

            assert instance.chat.completions.create.call_count == 1

    async def test_empty_response_without_wait_raises_attribute_error(self):
        """Пустой ответ без model_wait → AttributeError (совместимость с чанкером)."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            instance.chat.completions.create.return_value = make_empty_completion()
            cls.return_value = instance

            client = LLMClient(
                base_url='http://localhost/v1', model='test',
            )
            with pytest.raises(AttributeError, match='choices'):
                await client.complete([{'role': 'user', 'content': 'Hi'}])

    async def test_wait_continues_on_any_error_during_wait(self):
        """Во время ожидания любые ошибки не прерывают цикл — ждём дальше."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            cls.return_value = instance

            call_count = 0

            async def side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception('400 - Model not found')
                if call_count == 2:
                    return make_empty_completion()
                if call_count == 3:
                    raise ConnectionError('connection refused')
                return make_completion('ok')

            instance.chat.completions.create.side_effect = side_effect

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=5, model_wait_retries=5,
                )
                result = await client.complete([{'role': 'user', 'content': 'Hi'}])

            assert result == 'ok'
            assert call_count == 4
            assert mock_sleep.call_count == 3

    async def test_non_model_error_not_caught(self):
        """Ошибки, не связанные с моделью, пробрасываются сразу без ожидания."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            instance.chat.completions.create.side_effect = ConnectionError('refused')
            cls.return_value = instance

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=10, model_wait_retries=3,
                )
                with pytest.raises(ConnectionError):
                    await client.complete([{'role': 'user', 'content': 'Hi'}])

            # Не должно быть ожидания — ошибка не связана с моделью
            mock_sleep.assert_not_called()
            assert instance.chat.completions.create.call_count == 1

    async def test_model_wait_works_with_complete_json(self):
        """Model wait работает и для complete_json."""
        with patch('morag.llm.client.AsyncOpenAI') as cls:
            instance = AsyncMock()
            instance.chat = AsyncMock()
            instance.chat.completions = AsyncMock()
            cls.return_value = instance

            call_count = 0

            async def side_effect(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return make_empty_completion()
                return make_completion('{"key": "value"}')

            instance.chat.completions.create.side_effect = side_effect

            with patch('morag.llm.client.asyncio.sleep', new_callable=AsyncMock):
                client = LLMClient(
                    base_url='http://localhost/v1', model='test',
                    model_wait_seconds=5, model_wait_retries=2,
                )
                schema = {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}
                result = await client.complete_json(
                    [{'role': 'user', 'content': 'Hi'}], schema=schema,
                )

            assert result == {'key': 'value'}
            assert call_count == 2


# ---------------------------------------------------------------------------
# Penalty kwargs (Grok compatibility)
# ---------------------------------------------------------------------------

class TestPenaltyKwargs:
    def test_zero_penalties_empty(self):
        result = LLMClient._penalty_kwargs(GenerationParams())
        assert result == {}

    def test_nonzero_frequency(self):
        params = GenerationParams(frequency_penalty=0.5)
        result = LLMClient._penalty_kwargs(params)
        assert result == {'frequency_penalty': 0.5}

    def test_nonzero_presence(self):
        params = GenerationParams(presence_penalty=0.3)
        result = LLMClient._penalty_kwargs(params)
        assert result == {'presence_penalty': 0.3}

    def test_both_nonzero(self):
        params = GenerationParams(frequency_penalty=0.5, presence_penalty=0.3)
        result = LLMClient._penalty_kwargs(params)
        assert result == {'frequency_penalty': 0.5, 'presence_penalty': 0.3}

    def test_zero_not_included(self):
        params = GenerationParams(frequency_penalty=0.0, presence_penalty=0.0)
        result = LLMClient._penalty_kwargs(params)
        assert 'frequency_penalty' not in result
        assert 'presence_penalty' not in result


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_no_limiter_by_default(self):
        with patch('morag.llm.client.AsyncOpenAI'):
            client = LLMClient(base_url='http://test', model='test')
            assert client._rate_limiter is None

    def test_limiter_created_with_max_rpm(self):
        with patch('morag.llm.client.AsyncOpenAI'):
            client = LLMClient(base_url='http://test', model='test', max_rpm=60)
            assert client._rate_limiter is not None
