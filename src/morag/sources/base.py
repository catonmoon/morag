from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class Document:
    """Документ из внешнего источника данных."""

    id: str                                    # стабильный идентификатор: относительный путь или confluence page id
    path: str                                  # путь для отображения (совпадает с id)
    text: str                                  # полный текст в Markdown
    updated_at: datetime                       # дата последнего изменения файла (mtime)
    source_type: str                           # "markdown" | "confluence"
    size: int = 0                              # размер файла в байтах
    indexed_at: datetime | None = None        # дата индексации (выставляется репозиторием при upsert)
    payload: dict = field(default_factory=dict)  # метаданные от DocumentProcessor-ов


@dataclass
class Chunk:
    """Чанк документа, готовый к сохранению в Qdrant."""

    doc_id: str                                       # ссылка на Document.id
    path: str                                         # путь документа (для фильтрации)
    order: int                                        # порядковый номер в документе (0-based)
    total: int                                        # общее количество чанков документа
    text: str                                         # основной текст чанка
    updated_at: datetime                              # наследуется от Document
    context: str = ''                                 # LLM-суммари (пусто если NoopContextGenerator)
    payload: dict = field(default_factory=dict)       # метаданные от ChunkProcessor-ов
    vectors: dict[str, list[float]] = field(default_factory=dict)  # именованные векторы от embedding-процессоров
    id: str = field(default_factory=lambda: str(uuid4()))


class Source(ABC):
    """Абстрактный источник документов."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Загрузить все документы из источника."""
        ...
