from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

from aiolimiter import AsyncLimiter
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
    repetition_penalty: float | None = None  # vLLM/omlx: >1.0 штрафует повторы
    seed: int | None = None
    enable_thinking: bool | None = None  # включить/выключить thinking; None = default


_THINKING_RE = re.compile(r'<think>.*?</think>\s*', flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Удалить блоки <think>...</think> из ответа thinking-моделей."""
    return _THINKING_RE.sub('', text)


def _extract_content(message) -> str:
    """Извлечь текст из ответа, с fallback на reasoning (OpenRouter)."""
    content = message.content or ''
    if content:
        return _strip_thinking(content)
    # OpenRouter: content=null, весь текст в reasoning
    reasoning = getattr(message, 'reasoning', None)
    if reasoning:
        logger.debug('LLM content is empty, falling back to reasoning field')
        return _strip_thinking(reasoning)
    return ''


def _build_extra_body(params: GenerationParams) -> dict | None:
    """Собрать extra_body для нестандартных параметров (top_k, repetition_penalty, thinking)."""
    extra: dict = {}
    if params.top_k != 0:
        extra['top_k'] = params.top_k
    if params.repetition_penalty is not None:
        extra['repetition_penalty'] = params.repetition_penalty
    if params.enable_thinking is not None:
        # vLLM / Ollama
        extra['chat_template_kwargs'] = {'enable_thinking': params.enable_thinking}
        # OpenRouter
        extra['reasoning'] = {'enabled': params.enable_thinking}
    return extra or None


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
        max_rpm: int | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
        self._model = model
        self._model_wait_seconds = model_wait_seconds
        self._model_wait_retries = model_wait_retries
        self._rate_limiter = AsyncLimiter(max_rpm, 60) if max_rpm else None

    @asynccontextmanager
    async def _rate_limit(self):
        """Acquire rate limiter if configured, otherwise no-op."""
        if self._rate_limiter:
            t0 = asyncio.get_event_loop().time()
            async with self._rate_limiter:
                waited = asyncio.get_event_loop().time() - t0
                if waited > 0.1:
                    logger.info('Rate limiter: waited %.1fs for token', waited)
                yield
        else:
            yield

    async def _create(self, **kwargs):
        """Обёртка над chat.completions.create с rate limiting и model-wait логикой.

        Rate limiting: если задан max_rpm, ожидает свободный слот перед запросом.

        Обнаруживает два сигнала недоступности модели:
        1. response.choices is None — сервер вернул 200, но пустое тело (перезагрузка)
        2. BadRequestError "Model not found" — модель ещё не загружена

        При обнаружении ждёт model_wait_seconds и повторяет до model_wait_retries раз.
        Любая ошибка во время ожидания считается «модель ещё не готова» — ждём дальше.
        """
        # Первая попытка (без ожидания)
        try:
            async with self._rate_limit():
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

        # Цикл ожидания перезагрузки модели (с jitter для предотвращения thundering herd)
        for wait_attempt in range(1, self._model_wait_retries + 1):
            jitter = random.uniform(0, self._model_wait_seconds * 0.5)
            wait_with_jitter = self._model_wait_seconds + jitter
            logger.warning(
                'LLMClient: model unavailable, waiting %.1fs (attempt %d/%d)...',
                wait_with_jitter, wait_attempt, self._model_wait_retries,
            )
            await asyncio.sleep(wait_with_jitter)

            try:
                async with self._rate_limit():
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

    @staticmethod
    def _penalty_kwargs(params: GenerationParams) -> dict:
        """Добавить penalty-параметры только если отличаются от дефолта (0.0)."""
        kwargs: dict = {}
        if params.frequency_penalty != 0.0:
            kwargs['frequency_penalty'] = params.frequency_penalty
        if params.presence_penalty != 0.0:
            kwargs['presence_penalty'] = params.presence_penalty
        return kwargs

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
            **self._penalty_kwargs(params),
        )
        if params.seed is not None:
            kwargs['seed'] = params.seed
        extra_body = _build_extra_body(params)
        if extra_body:
            kwargs['extra_body'] = extra_body
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        response = await self._create(**kwargs)
        return _extract_content(response.choices[0].message)

    async def complete_vision(
        self,
        prompt: str,
        image_base64: str,
        media_type: str = 'image/png',
        max_tokens: int | None = None,
        params: GenerationParams | None = None,
    ) -> str:
        """Описать изображение через multimodal LLM (Vision).

        Принимает изображение в формате base64 и текстовый запрос.
        Возвращает текстовое описание изображения.
        """
        if params is None:
            params = GenerationParams()
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
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=params.temperature,
            **self._penalty_kwargs(params),
        )
        if params.seed is not None:
            kwargs['seed'] = params.seed
        extra_body = _build_extra_body(params)
        if extra_body:
            kwargs['extra_body'] = extra_body
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        response = await self._create(**kwargs)
        return _extract_content(response.choices[0].message)

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
            **self._penalty_kwargs(params),
        )
        if params.seed is not None:
            kwargs['seed'] = params.seed
        extra_body = _build_extra_body(params)
        if extra_body:
            kwargs['extra_body'] = extra_body
        response = await self._create(**kwargs)
        content = _extract_content(response.choices[0].message) or '{}'
        logger.debug('LLM raw response: %s', content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning('LLM returned invalid JSON: %s\nRaw content: %s', e, content)
            raise ValueError(f'LLM returned invalid JSON: {e}') from e


class _ModelUnavailableError(Exception):
    """Модель временно недоступна (перезагрузка / пустой ответ)."""
