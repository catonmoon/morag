from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

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


def _is_model_unavailable(exc: Exception) -> bool:
    """Определить, связана ли ошибка с недоступностью модели."""
    msg = str(exc).lower()
    if 'model not found' in msg or 'model_not_found' in msg:
        return True
    return False


class LLMClient:
    """Async OpenAI-compatible LLM client.

    Works with OpenAI, Ollama, LM Studio and any OpenAI-compatible server
    via the base_url parameter.

    При обнаружении недоступности модели (пустой ответ, Model not found)
    ожидает перезагрузки модели: model_wait_seconds × model_wait_retries.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = 'ollama',
        timeout: int = 180,
        max_retries: int = 3,
        model_wait_seconds: int = 0,
        model_wait_retries: int = 0,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
        self._model = model
        self._model_wait_seconds = model_wait_seconds
        self._model_wait_retries = model_wait_retries

    async def _create(self, **kwargs):
        """Обёртка над chat.completions.create с model-wait логикой.

        Обнаруживает два сигнала недоступности модели:
        1. response.choices is None — сервер вернул 200, но пустое тело (перезагрузка)
        2. BadRequestError "Model not found" — модель ещё не загружена

        При обнаружении ждёт model_wait_seconds и повторяет до model_wait_retries раз.
        Любая ошибка во время ожидания считается «модель ещё не готова» — ждём дальше.
        """
        # Первая попытка (без ожидания)
        try:
            response = await self._client.chat.completions.create(**kwargs)
            if response.choices is None:
                raise _ModelUnavailableError('Empty response from server (choices is None)')
            return response
        except Exception as exc:
            if not (isinstance(exc, _ModelUnavailableError) or _is_model_unavailable(exc)):
                raise
            if self._model_wait_retries == 0:
                if isinstance(exc, _ModelUnavailableError):
                    raise AttributeError(
                        "'NoneType' object has no attribute 'choices'"
                    ) from exc
                raise
            last_exc = exc

        # Цикл ожидания перезагрузки модели
        for wait_attempt in range(1, self._model_wait_retries + 1):
            logger.warning(
                'LLMClient: model unavailable, waiting %ds (attempt %d/%d)...',
                self._model_wait_seconds, wait_attempt, self._model_wait_retries,
            )
            await asyncio.sleep(self._model_wait_seconds)

            try:
                response = await self._client.chat.completions.create(**kwargs)
                if response.choices is None:
                    raise _ModelUnavailableError('Empty response from server (choices is None)')
                return response
            except Exception as exc:
                last_exc = exc
                # Любая ошибка во время ожидания = модель ещё не готова, ждём дальше

        # Все попытки исчерпаны — пробрасываем последнюю ошибку
        if isinstance(last_exc, _ModelUnavailableError):
            raise AttributeError(
                "'NoneType' object has no attribute 'choices'"
            ) from last_exc
        raise last_exc

    async def complete(
        self,
        messages: list[dict],
        params: GenerationParams | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
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
        response = await self._create(**kwargs)
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
        response = await self._create(
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
        response = await self._create(**kwargs)
        content = response.choices[0].message.content or '{}'
        logger.debug('LLM raw response: %s', content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning('LLM returned invalid JSON: %s\nRaw content: %s', e, content)
            raise ValueError(f'LLM returned invalid JSON: {e}') from e


class _ModelUnavailableError(Exception):
    """Модель временно недоступна (перезагрузка / пустой ответ)."""
