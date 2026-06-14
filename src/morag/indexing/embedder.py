from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager

from morag.llm.client import _get_or_create_semaphore
from morag.llm.retry import RetryPolicy

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Async sliding window rate limiter для HTTP embedders."""

    def __init__(self, max_rpm: int) -> None:
        self._max = max_rpm
        self._window = 60.0
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                sleep_for = self._window - (now - self._timestamps[0])
            logger.info(
                'AsyncRateLimiter: waiting %.1fs (%d/%d in window)',
                sleep_for, self._max, self._max,
            )
            await asyncio.sleep(sleep_for)


_MD5_MOD = 4_294_967_295  # DO NOT CHANGE


class Embedder(ABC):
    """Интерфейс вычисления эмбеддингов."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Эмбеддинг для хранения документа."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг для поискового запроса."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Батчевый эмбеддинг для хранения документов.

        По умолчанию вызывает embed() по одному. Подклассы могут
        переопределить для более эффективной батчевой обработки.
        """
        return [await self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Размерность вектора."""
        ...


class HttpEmbedder(Embedder):
    """Dense embedder через OpenAI-совместимый /v1/embeddings (AsyncOpenAI SDK).

    Работает с любой моделью (Ollama, vLLM, OpenAI API).
    Формат входа задаётся через document_template / query_template.
    Поле model отправляется в body — Ollama и OpenAI его используют.
    Retry на 429/5xx — из коробки через SDK (max_retries).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        api_key: str = 'ollama',
        document_template: str = '{text}',
        query_template: str = '{text}',
        timeout: int = 30,
        max_retries: int = 3,
        max_rpm: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._dim = dim
        self._doc_tpl = document_template
        self._qry_tpl = query_template
        self._rate_limiter = AsyncRateLimiter(max_rpm) if max_rpm else None
        # Shared in-flight cap по (base_url, model) — общий registry с LLMClient.
        # Для Qwen3-Embedding-4B (~4B params) — критично, инференс тяжёлый.
        self._semaphore = _get_or_create_semaphore(base_url, model, max_concurrent)
        logger.info(
            'HttpEmbedder → %s model=%r dim=%d max_rpm=%s max_concurrent=%s',
            base_url, model, dim, max_rpm, max_concurrent,
        )

    @asynccontextmanager
    async def _inflight_cap(self):
        """Acquire shared in-flight semaphore if configured, otherwise no-op."""
        if self._semaphore:
            async with self._semaphore:
                yield
        else:
            yield

    async def _with_empty_retry(self, fn, attempts: int = 8):
        """OpenRouter иногда отдаёт HTTP 200 с битым/пустым телом embeddings → SDK-парсер кидает
        ('No embedding data received' и т.п.). Это НЕ HTTP-ошибка, SDK-ретрай (429/5xx) её не ловит →
        документ падает на пустом месте. Ретраим с backoff — следующий запрос обычно ок."""
        for i in range(attempts):
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001 — транзиентный битый ответ провайдера
                if i == attempts - 1:
                    raise
                logger.warning('embed retry %d/%d after %s', i + 1, attempts - 1, e)
                await asyncio.sleep(1.0 * (i + 1))

    async def _do_call(self, text: str) -> list[float]:
        async def once():
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            async with self._inflight_cap():
                resp = await self._client.embeddings.create(model=self._model, input=text)
            return list(resp.data[0].embedding)
        return await self._with_empty_retry(once)

    async def _do_call_batch(self, texts: list[str]) -> list[list[float]]:
        async def once():
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            async with self._inflight_cap():
                resp = await self._client.embeddings.create(model=self._model, input=texts)
            data = sorted(resp.data, key=lambda d: d.index)
            return [list(d.embedding) for d in data]
        return await self._with_empty_retry(once)

    async def embed(self, text: str) -> list[float]:
        return await self._do_call(self._doc_tpl.format(text=text))

    async def embed_query(self, text: str) -> list[float]:
        return await self._do_call(self._qry_tpl.format(text=text))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info('HttpEmbedder: embed_batch %d text(s)', len(texts))
        wrapped = [self._doc_tpl.format(text=t) for t in texts]
        return await self._do_call_batch(wrapped)

    @property
    def dim(self) -> int:
        return self._dim


class SparseEmbedder(ABC):
    """Интерфейс вычисления sparse-эмбеддингов."""

    @abstractmethod
    async def embed(self, text: str) -> tuple[list[int], list[float]]:
        """Sparse-вектор для хранения документа.

        Возвращает (indices, values) — пару параллельных списков.
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        """Sparse-вектор для поискового запроса."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Батчевый sparse-эмбеддинг для хранения документов.

        По умолчанию вызывает embed() по одному. Подклассы могут
        переопределить для более эффективной батчевой обработки.
        """
        return [await self.embed(t) for t in texts]


def _word_to_index(word: str) -> int:
    """Хэш токена → индекс sparse-вектора (MD5 % 2^32 - 1)."""
    return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD


class HttpGteSparseEmbedder(SparseEmbedder):
    """Sparse-эмбеддинги через HTTP (POST /encode → {token_weights: [{word: weight}]}).

    Хэширование токенов в индексы выполняется на стороне клиента через _word_to_index —
    так же, как в сервисе embedder-gte. Это необходимо для консистентности индекса в Qdrant.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retry_policy: RetryPolicy | None = None,
        max_rpm: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        import httpx
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._retry = retry_policy or RetryPolicy(max_retries=0)
        self._rate_limiter = AsyncRateLimiter(max_rpm) if max_rpm else None
        # Shared in-flight cap по (base_url, '__gte_sparse__'). GTE-сервис
        # CPU-bound и сериализует запросы — без cap'а burst'ит хост, autoheal
        # рестартит контейнер на healthcheck timeout, in-flight запросы рвутся.
        self._semaphore = _get_or_create_semaphore(
            base_url, '__gte_sparse__', max_concurrent,
        )
        logger.info(
            'HttpGteSparseEmbedder → %s (max_rpm=%s, max_concurrent=%s)',
            base_url, max_rpm, max_concurrent,
        )

    @staticmethod
    def _to_sparse(token_weights: dict[str, float]) -> tuple[list[int], list[float]]:
        index_weight: dict[int, float] = {}
        for word, weight in token_weights.items():
            i = _word_to_index(word)
            if i in index_weight:
                index_weight[i] = max(index_weight[i], weight)
            else:
                index_weight[i] = weight
        return list(index_weight.keys()), list(index_weight.values())

    async def _acquire(self):
        """Acquire shared in-flight semaphore (no-op if max_concurrent=None)."""
        if self._semaphore:
            return self._semaphore
        return None

    async def _do_call(self, text: str) -> tuple[list[int], list[float]]:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        if self._semaphore:
            async with self._semaphore:
                resp = await self._client.post('/encode', json={'text': text})
        else:
            resp = await self._client.post('/encode', json={'text': text})
        resp.raise_for_status()
        token_weights: dict[str, float] = resp.json()['token_weights'][0]
        return self._to_sparse(token_weights)

    async def _do_call_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        if self._semaphore:
            async with self._semaphore:
                resp = await self._client.post('/encode_batch', json={'texts': texts})
        else:
            resp = await self._client.post('/encode_batch', json={'texts': texts})
        resp.raise_for_status()
        all_weights: list[dict[str, float]] = resp.json()['token_weights']
        return [self._to_sparse(tw) for tw in all_weights]

    async def embed(self, text: str) -> tuple[list[int], list[float]]:
        return await self._retry.call(lambda: self._do_call(text))

    async def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return await self.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        logger.info('HttpGteSparseEmbedder: embed_batch %d text(s)', len(texts))
        return await self._retry.call(lambda: self._do_call_batch(texts))
