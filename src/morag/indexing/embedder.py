from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

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


_DOCUMENT_PREFIX = 'search_document: '
_QUERY_PREFIX = 'search_query: '

_MD5_MOD = 4_294_967_295  # DO NOT CHANGE


class Embedder(ABC):
    """Интерфейс вычисления эмбеддингов."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Эмбеддинг для хранения документа (с префиксом search_document:)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг для поискового запроса (с префиксом search_query:)."""
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


class FridaEmbedder(Embedder):
    """Dense-эмбеддинги через ai-forever/FRIDA.

    Загружает модель один раз при инициализации.
    Для индексации использует префикс 'search_document:',
    для запросов — 'search_query:'.
    """

    def __init__(self, model_name: str = 'ai-forever/FRIDA') -> None:
        import os
        import torch
        from sentence_transformers import SentenceTransformer
        num_cpus = os.cpu_count() or 1
        torch.set_num_threads(num_cpus)
        torch.set_num_interop_threads(num_cpus)
        logger.info('Loading embedding model: %s (threads=%d)', model_name, num_cpus)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info('Embedding model loaded, dim=%d', self._dim)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(
            _DOCUMENT_PREFIX + text, normalize_embeddings=False, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            _QUERY_PREFIX + text, normalize_embeddings=False, show_progress_bar=False,
        ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_DOCUMENT_PREFIX + t for t in texts]
        return self._model.encode(
            prefixed, normalize_embeddings=False, show_progress_bar=False,
        ).tolist()

    @property
    def dim(self) -> int:
        return self._dim


class HttpFridaEmbedder(Embedder):
    """Dense-эмбеддинги через HTTP-эндпоинт (OpenAI-compatible /v1/embeddings).

    Добавляет префиксы search_document: / search_query: перед отправкой — так же, как
    локальный FridaEmbedder. Размерность вектора задаётся явно через параметр dim.
    """

    def __init__(self, base_url: str, dim: int, timeout: int = 30,
                 retry_policy: RetryPolicy | None = None,
                 max_rpm: int | None = None) -> None:
        import httpx
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._dim = dim
        self._retry = retry_policy or RetryPolicy(max_retries=0)
        self._rate_limiter = SyncRateLimiter(max_rpm) if max_rpm else None
        logger.info('HttpFridaEmbedder → %s (dim=%d, max_rpm=%s)', base_url, dim, max_rpm)

    def _do_call(self, text: str) -> list[float]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post('/v1/embeddings', json={'input': text})
        resp.raise_for_status()
        return resp.json()['data'][0]['embedding']

    def _do_call_batch(self, texts: list[str]) -> list[list[float]]:
        if self._rate_limiter:
            self._rate_limiter.acquire()
        resp = self._client.post('/v1/embeddings', json={'input': texts})
        resp.raise_for_status()
        data = resp.json()['data']
        data.sort(key=lambda d: d['index'])
        return [d['embedding'] for d in data]

    def _call(self, text: str) -> list[float]:
        return self._retry.call_sync(lambda: self._do_call(text))

    def _call_batch(self, texts: list[str]) -> list[list[float]]:
        return self._retry.call_sync(lambda: self._do_call_batch(texts))

    def embed(self, text: str) -> list[float]:
        return self._call(_DOCUMENT_PREFIX + text)

    def embed_query(self, text: str) -> list[float]:
        return self._call(_QUERY_PREFIX + text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info('HttpFridaEmbedder: embed_batch %d text(s)', len(texts))
        prefixed = [_DOCUMENT_PREFIX + t for t in texts]
        return self._call_batch(prefixed)

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


def _token_weights_to_sparse(
    token_weights: list[float],
    input_ids: list[int],
    unused_tokens: set[int],
    decode_fn,
) -> tuple[list[int], list[float]]:
    """Преобразовать выходы модели в (indices, values).

    Алгоритм:
    1. Фильтрует спец-токены и нулевые веса.
    2. Декодирует token_id → строку.
    3. Для дублирующихся строк берёт максимальный вес.
    4. Хэширует строку через MD5 → индекс.
    5. При коллизии индексов берёт максимальный вес.
    """
    word_weights: dict[str, float] = defaultdict(float)
    for w, idx in zip(token_weights, input_ids):
        if idx in unused_tokens or w <= 0:
            continue
        tok = decode_fn([int(idx)], skip_special_tokens=True).strip()
        if w > word_weights[tok]:
            word_weights[tok] = float(w)

    index_weight: dict[int, float] = {}
    for word, weight in word_weights.items():
        i = _word_to_index(word)
        if i in index_weight:
            index_weight[i] = max(index_weight[i], weight)
        else:
            index_weight[i] = weight

    indices = list(index_weight.keys())
    values = list(index_weight.values())
    return indices, values


class GteSparseEmbedder(SparseEmbedder):
    """Sparse-эмбеддинги через Alibaba-NLP/gte-multilingual-base.

    Загружает модель один раз при инициализации.
    Не использует префиксы и не меняет регистр текста.
    """

    def __init__(
        self,
        model_name: str = 'Alibaba-NLP/gte-multilingual-base',
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            # MPS causes AcceleratorError in GTE's custom RoPE kernel regardless of dtype;
            # CPU is used as a safe fallback on Apple Silicon and other non-CUDA systems.
            self._device = torch.device('cpu')

        logger.info('Loading sparse embedding model: %s on %s', model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Explicit fp16 on CUDA avoids buffer corruption from "torch_dtype: float16" in model config.
        # On CPU float16 is unsupported for some ops; use float32.
        model_dtype = torch.float16 if self._device.type == 'cuda' else torch.float32
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(self._device).eval()
        self._torch = torch

        self._unused_tokens = {
            t for t in [
                getattr(self._tokenizer, 'cls_token_id', None),
                getattr(self._tokenizer, 'eos_token_id', None),
                getattr(self._tokenizer, 'pad_token_id', None),
                getattr(self._tokenizer, 'unk_token_id', None),
            ] if t is not None
        }
        logger.info('Sparse embedding model loaded')

    def _encode_text(self, text: str) -> tuple[list[int], list[float]]:
        enc = self._tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192,
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        # GTE registers position_ids as a persistent=False buffer; model.to(dtype) converts
        # it to float, corrupting integer index values. Pass explicit int64 position_ids to
        # bypass the corrupted internal buffer entirely.
        enc['position_ids'] = self._torch.arange(
            enc['input_ids'].shape[1], dtype=self._torch.long, device=self._model.device,
        ).unsqueeze(0)
        with self._torch.no_grad():
            out = self._model(**enc, return_dict=True)
        logits = out.logits.detach().cpu()
        token_weights = self._torch.relu(logits).squeeze(-1)
        tw = token_weights[0].numpy().tolist()
        ids = enc['input_ids'][0].cpu().numpy().tolist()
        return _token_weights_to_sparse(tw, ids, self._unused_tokens, self._tokenizer.decode)

    def embed(self, text: str) -> tuple[list[int], list[float]]:
        return self._encode_text(text)

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._encode_text(text)


class HttpGteSparseEmbedder(SparseEmbedder):
    """Sparse-эмбеддинги через HTTP-эндпоинт (POST /encode → {token_weights: [{word: weight}]}).

    Хэширование токенов в индексы выполняется на стороне клиента через _word_to_index —
    так же, как в GteSparseEmbedder. Это необходимо для консистентности индекса в Qdrant.
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
