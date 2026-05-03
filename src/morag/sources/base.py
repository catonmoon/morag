from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class Document:
    """Документ из внешнего источника данных."""

    id: str                                    # стабильный идентификатор: относительный путь или confluence page id
    path: list[str]                            # пути для отображения; список — один документ может встречаться в нескольких местах
    text: str                                  # полный текст в Markdown
    updated_at: datetime                       # дата последнего изменения файла (mtime)
    source_type: str                           # "markdown" | "confluence" | "attached_jira" | "attached_pdf"
    title: str | None = None                   # название документа (из Confluence / frontmatter / имя файла)
    size: int = 0                              # размер файла в байтах
    url: str | None = None                    # ссылка на источник (для Confluence — URL страницы, для Markdown — None)
    indexed_at: datetime | None = None        # дата индексации (выставляется репозиторием при upsert)
    creator: str | None = None                # автор документа (из frontmatter / git / Confluence history)
    created_at: datetime | None = None        # дата создания документа
    parent_doc_ids: list[str] = field(default_factory=list)  # doc_id родительских документов (для каскадного удаления)
    structural: bool = False                               # структурный (навигационный) документ без полезного тела; не чанкуется
    paged: bool = False                                    # страничный документ (PDF, DOCX, PPTX); чанки обязаны иметь pages
    payload: dict = field(default_factory=dict)  # метаданные от DocumentProcessor-ов
    vectors: dict[str, list[float] | dict] = field(default_factory=dict)  # doc-level именованные векторы (для section-level retrieval через DocVectorProcessor)


@dataclass
class Chunk:
    """Чанк документа, готовый к сохранению в Qdrant."""

    doc_id: str                                       # ссылка на Document.id
    path: list[str]                                   # пути документа (список, как у Document)
    order: int                                        # порядковый номер в документе (0-based)
    total: int                                        # общее количество чанков документа
    text: str                                         # основной текст чанка
    updated_at: datetime                              # наследуется от Document
    source_type: str = ''                             # тип источника (копируется с Document)
    context: str = ''                                 # LLM-суммари (пусто если NoopContextGenerator)
    payload: dict = field(default_factory=dict)       # метаданные от ChunkProcessor-ов
    vectors: dict[str, list[float] | dict] = field(default_factory=dict)  # именованные векторы от embedding-процессоров
    id: str = field(default_factory=lambda: str(uuid4()))


class Source(ABC):
    """Абстрактный источник документов.

    Каждый source знает свой `kind` (тип) и `name` (имя инстанса) — оба берутся
    из config (см. ADR-0012). Используются для генерации prefixed ID документов:
    `<kind>:<name>:<external-id>`. Это гарантирует уникальность ID даже при
    нескольких инстансах одного типа (multi-Confluence, multi-Jira).

    Конкретные реализации обязаны установить self._kind и self._name в __init__.
    """

    _kind: str           # 'local' | 'confluence' | 'jira' (см. config.Source discriminator)
    _name: str           # имя инстанса из config

    @property
    def kind(self) -> str:
        """Тип источника, соответствует discriminator в config (Source.kind)."""
        return self._kind

    @property
    def name(self) -> str:
        """Имя инстанса источника из config (например 'corp', 'vendor')."""
        return self._name

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Категория документа в payload (для UI/retrieval-фильтров).

        Обычно совпадает с kind, но для composite-источников может отличаться:
        ConfluencePdfSource имеет kind='confluence' но source_type='attached_pdf'.
        """
        ...

    def make_id(self, external_id: str) -> str:
        """Построить полный prefixed-ID документа.

        Формат: `<kind>:<name>:<external_id>`. Гарантирует уникальность между
        инстансами одного типа. См. ADR-0012, раздел Document IDs.
        """
        return f'{self._kind}:{self._name}:{external_id}'

    @abstractmethod
    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы документов: только id, updated_at, source_type, size. text=''.

        Document.id ДОЛЖЕН быть prefixed через self.make_id().
        """
        ...

    @abstractmethod
    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить один документ по prefixed-ID."""
        ...

    async def load(self) -> list[Document]:
        """Загрузить все документы (дефолт: get_metadata → load_one для каждого)."""
        stubs = await self.get_metadata()
        docs: list[Document] = []
        for stub in stubs:
            doc = await self.load_one(stub.id)
            if doc is not None:
                docs.append(doc)
        return docs
