from __future__ import annotations

from abc import ABC, abstractmethod

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
