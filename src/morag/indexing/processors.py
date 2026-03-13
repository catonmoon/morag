from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from morag.indexing.embedder import Embedder, SparseEmbedder
from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.llm.client import GenerationParams, LLMClient
from morag.sources.base import Chunk, Document

logger = logging.getLogger(__name__)

_LLM_PARAMS = GenerationParams(
    temperature=0.0, top_p=1.0, top_k=0,
    frequency_penalty=0.0, presence_penalty=0.0, seed=42,
)

_SUMMARY_PROMPT_NO_PARENT = """\
Кратко опиши содержание документа — о чём он, какую задачу решает или что описывает. \
Пиши только саммари, без вводных фраз.

Документ:
{doc_text}\
"""

_SUMMARY_PROMPT_WITH_PARENTS = """\
Кратко опиши содержание документа с учётом его места в иерархии. \
Пиши только саммари, без вводных фраз.

Контекст родительских документов:
{parent_context}

Документ:
{doc_text}\
"""


class DocumentProcessor(ABC):
    """Интерфейс обработчика документа.

    Применяется до сохранения документа в Qdrant.
    Используется для: конвертации форматов (PDF/DOCX → MD),
    обогащения payload метаданными (автор, теги, ACL и т.д.).
    """

    @abstractmethod
    async def process(self, document: Document) -> Document:
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

    def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Батчевая обработка чанков. По умолчанию вызывает process() по одному."""
        return [self.process(c, document) for c in chunks]


class DenseEmbeddingProcessor(ChunkProcessor):
    """Добавляет dense-вектор 'full' в chunk.vectors.

    Вектор строится из конкатенации path + text + context —
    это даёт полное представление чанка в контексте документа.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    @staticmethod
    def _full_text(chunk: Chunk) -> str:
        return f'{"\n".join(chunk.path)}\n{chunk.text}\n{chunk.context}'

    def process(self, chunk: Chunk, document: Document) -> Chunk:
        chunk.vectors['full'] = self._embedder.embed(self._full_text(chunk))
        return chunk

    def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Батчевый эмбеддинг всех чанков документа за один вызов."""
        if not chunks:
            return chunks
        texts = [self._full_text(c) for c in chunks]
        vectors = self._embedder.embed_batch(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.vectors['full'] = vec
        return chunks


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


class DocSummaryProcessor(DocumentProcessor):
    """Генерирует контекстуальное саммари документа и сохраняет в payload['doc_summary'].

    Саммари учитывает иерархию: если у документа есть родители, их саммари включается
    в промпт как контекст. Родители гарантированно обработаны раньше (BFS-порядок в pipeline).
    Структурные документы и документы без текста пропускаются.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        doc_repo,
        max_tokens: int = 128,
        token_counter: TokenCounter | None = None,
        context_window: int = 32768,
    ) -> None:
        self._client = llm_client
        self._doc_repo = doc_repo
        self._max_tokens = max_tokens
        self._token_counter = token_counter or TiktokenCounter()
        self._context_window = context_window
        self._overhead_no_parent = self._token_counter.count(
            _SUMMARY_PROMPT_NO_PARENT.format(doc_text='')
        )
        self._overhead_with_parents = self._token_counter.count(
            _SUMMARY_PROMPT_WITH_PARENTS.format(parent_context='', doc_text='')
        )

    async def process(self, document: Document) -> Document:
        """Сгенерировать саммари документа с учётом саммари родителей."""
        if document.structural or not document.text.strip():
            return document

        # Собираем саммари родителей из doc_repo (уже обработаны на предыдущем BFS-уровне)
        parent_summaries: list[str] = []
        for parent_id in document.parent_doc_ids:
            parent_doc = await self._doc_repo.get_by_id(parent_id)
            if parent_doc and 'doc_summary' in parent_doc.payload:
                parent_summaries.append(parent_doc.payload['doc_summary'])

        if parent_summaries:
            parent_context = '\n\n'.join(parent_summaries)
            parent_tokens = self._token_counter.count(parent_context)
            available = self._context_window - self._overhead_with_parents - parent_tokens - self._max_tokens
            doc_text = self._token_counter.truncate(document.text, max(available, 0))
            prompt = _SUMMARY_PROMPT_WITH_PARENTS.format(
                parent_context=parent_context,
                doc_text=doc_text,
            )
        else:
            available = self._context_window - self._overhead_no_parent - self._max_tokens
            doc_text = self._token_counter.truncate(document.text, max(available, 0))
            prompt = _SUMMARY_PROMPT_NO_PARENT.format(doc_text=doc_text)

        messages = [{'role': 'user', 'content': prompt}]
        try:
            summary = await self._client.complete(messages, params=_LLM_PARAMS, max_tokens=self._max_tokens)
            document.payload['doc_summary'] = summary.strip()
            logger.info('DocSummaryProcessor: %s (%d chars)', document.id, len(summary))
        except Exception:
            logger.warning(
                'DocSummaryProcessor: LLM call failed for %s, summary skipped',
                document.id, exc_info=True,
            )

        return document


class MetadataProcessor(ChunkProcessor):
    """Копирует creator и created_at из Document в chunk.payload.

    Позволяет использовать метаданные автора и даты создания
    в результатах поиска без дополнительных запросов к коллекции docs.
    """

    def process(self, chunk: Chunk, document: Document) -> Chunk:
        chunk.payload['source_type'] = document.source_type
        if document.creator is not None:
            chunk.payload['creator'] = document.creator
        if document.created_at is not None:
            chunk.payload['created_at'] = document.created_at.isoformat()
        if document.url is not None:
            chunk.payload['url'] = document.url
        return chunk
