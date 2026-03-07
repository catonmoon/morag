from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from morag.llm.retry import RetryPolicy

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Параметры семплинга для LLM."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None


class LLMClient:
    """Async OpenAI-compatible LLM client.

    Works with OpenAI, Ollama, LM Studio and any OpenAI-compatible server
    via the base_url parameter.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = 'ollama',
        timeout: int = 180,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._retry = retry_policy or RetryPolicy(max_retries=0)

    async def complete(
        self,
        messages: list[dict],
        params: GenerationParams | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        return await self._retry.call(
            lambda: self._do_complete(messages, params, max_tokens),
        )

    async def _do_complete(
        self,
        messages: list[dict],
        params: GenerationParams | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if params is None:
            params = GenerationParams()
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
        )
        if params.seed is not None:
            kwargs['seed'] = params.seed
        if params.top_k != 0:
            kwargs['extra_body'] = {'top_k': params.top_k}
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ''

    async def complete_vision(self, prompt: str, image_base64: str, media_type: str = 'image/png') -> str:
        """Описать изображение через multimodal LLM (Vision).

        Принимает изображение в формате base64 и текстовый запрос.
        Возвращает текстовое описание изображения.
        """
        return await self._retry.call(
            lambda: self._do_complete_vision(prompt, image_base64, media_type),
        )

    async def _do_complete_vision(self, prompt: str, image_base64: str, media_type: str) -> str:
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{media_type};base64,{image_base64}'},
                    },
                    {'type': 'text', 'text': prompt},
                ],
            }
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ''

    async def complete_json(
        self,
        messages: list[dict],
        schema: dict,
        schema_name: str = 'response',
        params: GenerationParams | None = None,
    ) -> dict:
        """Send a chat completion request expecting a JSON response matching the given schema.

        Passes response_format={"type": "json_schema", ...} to enforce structured output.
        Raises ValueError if the response cannot be parsed.
        """
        return await self._retry.call(
            lambda: self._do_complete_json(messages, schema, schema_name, params),
        )

    async def _do_complete_json(
        self,
        messages: list[dict],
        schema: dict,
        schema_name: str = 'response',
        params: GenerationParams | None = None,
    ) -> dict:
        if params is None:
            params = GenerationParams()
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            response_format={
                'type': 'json_schema',
                'json_schema': {'name': schema_name, 'schema': schema},
            },
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
        )
        if params.seed is not None:
            kwargs['seed'] = params.seed
        if params.top_k != 0:
            kwargs['extra_body'] = {'top_k': params.top_k}
        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or '{}'
        logger.debug('LLM raw response: %s', content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning('LLM returned invalid JSON: %s\nRaw content: %s', e, content)
            raise ValueError(f'LLM returned invalid JSON: {e}') from e
