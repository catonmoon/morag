from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_DOCUMENT_PREFIX = 'search_document: '
_QUERY_PREFIX = 'search_query: '


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
