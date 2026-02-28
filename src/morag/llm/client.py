from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Async OpenAI-compatible LLM client.

    Works with OpenAI, Ollama, LM Studio and any OpenAI-compatible server
    via the base_url parameter.
    """

    def __init__(self, base_url: str, model: str, api_key: str = 'ollama') -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Send a chat completion request and return the response text."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ''

    async def complete_vision(self, prompt: str, image_base64: str, media_type: str = 'image/png') -> str:
        """Описать изображение через multimodal LLM (Vision).

        Принимает изображение в формате base64 и текстовый запрос.
        Возвращает текстовое описание изображения.
        """
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

    async def complete_json(self, messages: list[dict]) -> dict:
        """Send a chat completion request expecting a JSON response.

        Passes response_format={"type": "json_object"} to instruct the model
        to return valid JSON. Raises ValueError if the response cannot be parsed.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={'type': 'json_object'},
        )
        content = response.choices[0].message.content or '{}'
        logger.debug('LLM raw response: %s', content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning('LLM returned invalid JSON: %s\nRaw content: %s', e, content)
            raise ValueError(f'LLM returned invalid JSON: {e}') from e
