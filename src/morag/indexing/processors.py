from __future__ import annotations

from abc import ABC, abstractmethod

from morag.indexing.embedder import Embedder, SparseEmbedder
from morag.sources.base import Chunk, Document


class DocumentProcessor(ABC):
    """Интерфейс обработчика документа.

    Применяется до сохранения документа в Qdrant.
    Используется для: конвертации форматов (PDF/DOCX → MD),
    обогащения payload метаданными (автор, теги, ACL и т.д.).
    """

    @abstractmethod
    def process(self, document: Document) -> Document:
        """Обработать документ и вернуть обновлённую версию."""
        ...


class ChunkProcessor(ABC):
    """Интерфейс обработчика чанка.

    Применяется после сборки базового чанка (текст + контекст).
    Используется для: добавления payload-метаданных, построения эмбеддингов.
    Каждый embedding-процессор добавляет именованный вектор в chunk.vectors.
    """

    @abstractmethod
    def process(self, chunk: Chunk, document: Document) -> Chunk:
        """Обработать чанк и вернуть обновлённую версию."""
        ...


class DenseEmbeddingProcessor(ChunkProcessor):
    """Добавляет dense-вектор 'full' в chunk.vectors.

    Вектор строится из конкатенации path + text + context —
    это даёт полное представление чанка в контексте документа.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def process(self, chunk: Chunk, document: Document) -> Chunk:
        full_text = f'{chunk.path}\n{chunk.text}\n{chunk.context}'
        chunk.vectors['full'] = self._embedder.embed(full_text)
        return chunk


class SparseEmbeddingProcessor(ChunkProcessor):
    """Добавляет sparse-вектор 'keywords' в chunk.vectors.

    Вектор строится из основного текста чанка без контекста —
    sparse-поиск ориентирован на точное совпадение ключевых слов.
    Сохраняется в формате {'indices': [...], 'values': [...]}.
    """

    def __init__(self, embedder: SparseEmbedder) -> None:
        self._embedder = embedder

    def process(self, chunk: Chunk, document: Document) -> Chunk:
        indices, values = self._embedder.embed(chunk.text)
        chunk.vectors['keywords'] = {'indices': indices, 'values': values}
        return chunk
