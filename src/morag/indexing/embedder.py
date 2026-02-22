from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)

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

    DIM = 1024

    def __init__(self, model_name: str = 'ai-forever/FRIDA') -> None:
        from sentence_transformers import SentenceTransformer
        logger.info('Loading embedding model: %s', model_name)
        self._model = SentenceTransformer(model_name)
        logger.info('Embedding model loaded, dim=%d', self.DIM)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(_DOCUMENT_PREFIX + text, normalize_embeddings=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(_QUERY_PREFIX + text, normalize_embeddings=False).tolist()

    @property
    def dim(self) -> int:
        return self.DIM


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

    def __init__(self, model_name: str = 'Alibaba-NLP/gte-multilingual-base') -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        logger.info('Loading sparse embedding model: %s', model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_name, trust_remote_code=True,
        ).eval()
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
            truncation=True,
            max_length=8192,
        )
        with self._torch.no_grad():
            out = self._model(**enc, return_dict=True)
        token_weights = self._torch.relu(out.logits).squeeze(-1)
        tw = token_weights[0].cpu().numpy().tolist()
        ids = enc['input_ids'][0].cpu().numpy().tolist()
        return _token_weights_to_sparse(tw, ids, self._unused_tokens, self._tokenizer.decode)

    def embed(self, text: str) -> tuple[list[int], list[float]]:
        return self._encode_text(text)

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._encode_text(text)
