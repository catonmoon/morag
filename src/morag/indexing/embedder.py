from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque

from morag.llm.retry import RetryPolicy

logger = logging.getLogger(__name__)


class SyncRateLimiter:
    """Синхронный sliding window rate limiter для HTTP embedders."""

    def __init__(self, max_rpm: int) -> None:
        self._max = max_rpm
        self._window = 60.0
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                sleep_for = self._window - (now - self._timestamps[0])
            logger.info('SyncRateLimiter: waiting %.1fs (%d/%d in window)', sleep_for, self._max, self._max)
            time.sleep(sleep_for)


_MD5_MOD = 4_294_967_295  # DO NOT CHANGE


class Embedder(ABC):
    """Интерфейс вычисления эмбеддингов."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Эмбеддинг для хранения документа."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг для поискового запроса."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Батчевый эмбеддинг для хранения документов.

        По умолчанию вызывает embed() по одному. Подклассы могут
        переопределить для более эффективной батчевой обработки.
        """
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Размерность вектора."""
        ...


class HttpEmbedder(Embedder):
    """Dense embedder через OpenAI-совместимый /v1/embeddings.

    Работает с любой моделью (Ollama, vLLM, OpenAI API).
    Формат входа задаётся через document_template / query_template.
    Поле model отправляется в body — Ollama и OpenAI его используют.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        document_template: str = '{text}',
        query_template: str = '{text}',
        timeout: int = 30,
        retry_policy: RetryPolicy | None = None,
        max_rpm: int | None = None,
    ) -> None:
        import httpx
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._model = model
        self._dim = dim
        self._doc_tpl = document_template
        self._qry_tpl = query_template
        self._retry = retry_policy or RetryPolicy(max_retries=0)
        self._rate_limiter = SyncRateLimiter(max_rpm) if max_rpm else None
        logger.info(
            'HttpEmbedder → %s model=%r dim=%d max_rpm=%s',
            base_url, model, dim, max_rpm,
        )

    def _do_call(self, text: str) -> list[float]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post(
            '/v1/embeddings',
            json={'model': self._model, 'input': text},
        )
        resp.raise_for_status()
        return resp.json()['data'][0]['embedding']

    def _do_call_batch(self, texts: list[str]) -> list[list[float]]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post(
            '/v1/embeddings',
            json={'model': self._model, 'input': texts},
        )
        resp.raise_for_status()
        data = resp.json()['data']
        data.sort(key=lambda d: d['index'])
        return [d['embedding'] for d in data]

    def _call(self, text: str) -> list[float]:
        return self._retry.call_sync(lambda: self._do_call(text))

    def _call_batch(self, texts: list[str]) -> list[list[float]]:
        return self._retry.call_sync(lambda: self._do_call_batch(texts))

    def embed(self, text: str) -> list[float]:
        return self._call(self._doc_tpl.format(text=text))

    def embed_query(self, text: str) -> list[float]:
        return self._call(self._qry_tpl.format(text=text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info('HttpEmbedder: embed_batch %d text(s)', len(texts))
        wrapped = [self._doc_tpl.format(text=t) for t in texts]
        return self._call_batch(wrapped)

    @property
    def dim(self) -> int:
        return self._dim


class SparseEmbedder(ABC):
    """Интерфейс вычисления sparse-эмбеддингов."""

    @abstractmethod
    def embed(self, text: str) -> tuple[list[int], list[float]]:
        """Sparse-вектор для хранения документа.

        Возвращает (indices, values) — пару параллельных списков.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        """Sparse-вектор для поискового запроса."""
        ...

    def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Батчевый sparse-эмбеддинг для хранения документов.

        По умолчанию вызывает embed() по одному. Подклассы могут
        переопределить для более эффективной батчевой обработки.
        """
        return [self.embed(t) for t in texts]


def _word_to_index(word: str) -> int:
    """Хэш токена → индекс sparse-вектора (MD5 % 2^32 - 1)."""
    return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD


class HttpGteSparseEmbedder(SparseEmbedder):
    """Sparse-эмбеддинги через HTTP-эндпоинт (POST /encode → {token_weights: [{word: weight}]}).

    Хэширование токенов в индексы выполняется на стороне клиента через _word_to_index —
    так же, как в сервисе embedder-gte. Это необходимо для консистентности индекса в Qdrant.
    """

    def __init__(self, base_url: str, timeout: int = 30,
                 retry_policy: RetryPolicy | None = None,
                 max_rpm: int | None = None) -> None:
        import httpx
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._retry = retry_policy or RetryPolicy(max_retries=0)
        self._rate_limiter = SyncRateLimiter(max_rpm) if max_rpm else None
        logger.info('HttpGteSparseEmbedder → %s (max_rpm=%s)', base_url, max_rpm)

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

    def _do_call(self, text: str) -> tuple[list[int], list[float]]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post('/encode', json={'text': text})
        resp.raise_for_status()
        token_weights: dict[str, float] = resp.json()['token_weights'][0]
        return self._to_sparse(token_weights)

    def _do_call_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post('/encode_batch', json={'texts': texts})
        resp.raise_for_status()
        all_weights: list[dict[str, float]] = resp.json()['token_weights']
        return [self._to_sparse(tw) for tw in all_weights]

    def _call(self, text: str) -> tuple[list[int], list[float]]:
        return self._retry.call_sync(lambda: self._do_call(text))

    def _call_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return self._retry.call_sync(lambda: self._do_call_batch(texts))

    def embed(self, text: str) -> tuple[list[int], list[float]]:
        return self._call(text)

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._call(text)

    def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        logger.info('HttpGteSparseEmbedder: embed_batch %d text(s)', len(texts))
        return self._call_batch(texts)
