from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Реестр shared in-flight семафоров по (base_url, model). Один и тот же провайдер
# (например vLLM-инстанс, шарится между llm и llm_vision на одной модели) получает
# единый семафор, а не отдельный на каждый LLMClient.
_shared_semaphores: dict[tuple[str, str], tuple[asyncio.Semaphore, int]] = {}
_registry_lock = threading.Lock()


def _get_or_create_semaphore(
    base_url: str, model: str, max_concurrent: int | None,
) -> asyncio.Semaphore | None:
    """Получить shared in-flight Semaphore для (base_url, model), создать при необходимости.

    Если для ключа уже зарегистрирован семафор с другим max_concurrent — оставляем
    существующий и логируем warning. Пользователь должен согласовать значения в конфиге.
    """
    if max_concurrent is None:
        return None
    key = (base_url, model)
    with _registry_lock:
        existing = _shared_semaphores.get(key)
        if existing is not None:
            sem, existing_max = existing
            if max_concurrent != existing_max:
                logger.warning(
                    'LLMClient: shared semaphore for %s already exists with max_concurrent=%d, '
                    'new request asked for %d — keeping existing. Set the same value in config.',
                    key, existing_max, max_concurrent,
                )
            return sem
        sem = asyncio.Semaphore(max_concurrent)
        _shared_semaphores[key] = (sem, max_concurrent)
        logger.info('LLMClient: registered shared in-flight semaphore for %s = %d', key, max_concurrent)
        return sem


def _reset_shared_semaphores() -> None:
    """Очистить registry — для тестов."""
    with _registry_lock:
        _shared_semaphores.clear()


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
        # vLLM
        extra['chat_template_kwargs'] = {'enable_thinking': params.enable_thinking}
        # Ollama (/v1/ OpenAI-compatible API)
        extra['think'] = params.enable_thinking
        extra['options'] = {'think': params.enable_thinking}
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
        max_concurrent: int | None = None,
        enable_thinking: bool | None = None,
        seed: int | None = 42,
        context_window: int = 32768,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
        self._model = model
        self._model_wait_seconds = model_wait_seconds
        self._model_wait_retries = model_wait_retries
        # Shared in-flight cap по (base_url, model). 429-обработку делает SDK retry.
        self._semaphore = _get_or_create_semaphore(base_url, model, max_concurrent)
        self._default_enable_thinking = enable_thinking
        self._default_seed = seed
        self._context_window = context_window

    @property
    def context_window(self) -> int:
        """Контекстное окно модели в токенах. Для сайзинга промптов у consumer'ов."""
        return self._context_window

    def _resolve_params(self, params: GenerationParams | None) -> GenerationParams:
        """Merge client-level defaults (enable_thinking, seed) with per-call params."""
        p = params or GenerationParams()
        if p.enable_thinking is None and self._default_enable_thinking is not None:
            p = replace(p, enable_thinking=self._default_enable_thinking)
        if p.seed is None and self._default_seed is not None:
            p = replace(p, seed=self._default_seed)
        return p

    @asynccontextmanager
    async def _inflight_cap(self):
        """Acquire shared in-flight semaphore if configured, otherwise no-op."""
        if self._semaphore:
            t0 = asyncio.get_event_loop().time()
            async with self._semaphore:
                waited = asyncio.get_event_loop().time() - t0
                if waited > 0.1:
                    logger.info('In-flight cap: waited %.1fs for slot', waited)
                yield
        else:
            yield

    async def _create(self, **kwargs):
        """Обёртка над chat.completions.create с in-flight cap и model-wait логикой.

        In-flight cap: если задан max_concurrent, не больше N одновременных запросов.
        429-обработка — встроенная в SDK (max_retries + Retry-After).

        Обнаруживает два сигнала недоступности модели:
        1. response.choices is None — сервер вернул 200, но пустое тело (перезагрузка)
        2. BadRequestError "Model not found" — модель ещё не загружена

        При обнаружении ждёт model_wait_seconds и повторяет до model_wait_retries раз.
        Любая ошибка во время ожидания считается «модель ещё не готова» — ждём дальше.
        """
        # Первая попытка (без ожидания)
        try:
            async with self._inflight_cap():
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
                async with self._inflight_cap():
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
        params = self._resolve_params(params)
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

    async def stream_complete(
        self,
        messages: list[dict],
        params: GenerationParams | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict]:
        """Stream chat completion chunk-by-chunk.

        Yield'ит словари `{'kind': 'content' | 'reasoning', 'text': str}` —
        reasoning отделён от основного контента для UI-side рендеринга
        (<think>…</think> обрамление).

        Retry на connect-errors и 429/5xx — через AsyncOpenAI SDK (`max_retries`).
        Mid-stream обрыв (после установления connection) не ретраится — LLM
        generation не идемпотентна; потребитель должен обработать исключение.
        """
        params = self._resolve_params(params)
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            stream=True,
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

        async with self._inflight_cap():
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # reasoning_content (vLLM) или reasoning (OpenRouter) — не в типизированной
                # схеме SDK, достаём через getattr. Оба поля может не быть.
                reasoning = (
                    getattr(delta, 'reasoning_content', None)
                    or getattr(delta, 'reasoning', None)
                    or ''
                )
                if reasoning:
                    yield {'kind': 'reasoning', 'text': reasoning}
                if delta.content:
                    yield {'kind': 'content', 'text': delta.content}

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        params: GenerationParams | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Chat completion с tools (function calling). Streaming под капотом.

        Возвращает dict вида ``{'choices': [{'message': {...}, 'finish_reason': '...'}]}`` —
        совместимый с OpenAI-API форматом, чтобы агентский цикл мог итерироваться
        по `message.tool_calls` так же как над raw API JSON.

        ВНИМАНИЕ: используем streaming чтобы обойти строгую серверную валидацию
        ``tool_choice='auto'`` у некоторых vLLM-инсталляций (xAI Grok). Без stream
        сервер требует ``--enable-auto-tool-choice`` на старте и реджектит запрос с tools.
        Возвращаем агрегированный финал — потребитель не видит разницы со streamingом.

        Использование: retrieval-pipeline (`services/pipeline/morag_pipeline.py`) —
        агент выбирает между `search()` / `get_neighbors()` / финальным ответом.
        """
        params = self._resolve_params(params)
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            tools=tools,
            stream=True,
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

        async with self._inflight_cap():
            stream = await self._client.chat.completions.create(**kwargs)
            content_parts: list[str] = []
            tool_calls_by_idx: dict[int, dict] = {}
            finish_reason = ''
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                for tc in (delta.tool_calls or []):
                    idx = tc.index
                    slot = tool_calls_by_idx.setdefault(
                        idx,
                        {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}},
                    )
                    if tc.id:
                        slot['id'] = tc.id
                    if tc.type:
                        slot['type'] = tc.type
                    if tc.function:
                        if tc.function.name:
                            slot['function']['name'] += tc.function.name
                        if tc.function.arguments:
                            slot['function']['arguments'] += tc.function.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        message: dict = {'role': 'assistant', 'content': ''.join(content_parts)}
        if tool_calls_by_idx:
            message['tool_calls'] = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
        return {'choices': [{'message': message, 'finish_reason': finish_reason}]}

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
        params = self._resolve_params(params)
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
        max_tokens: int | None = None,
    ) -> dict:
        """Send a chat completion request expecting a JSON response matching the given schema.

        Passes response_format={"type": "json_schema", ...} to enforce structured output.
        Raises ValueError if the response cannot be parsed.
        """
        params = self._resolve_params(params)
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
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
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
