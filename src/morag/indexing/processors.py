from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from morag.indexing.embedder import Embedder, SparseEmbedder
from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.llm.client import LLMClient
from morag.sources.base import Chunk, Document

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT_NO_PARENT = """\
Briefly describe the document's content — what it is about, what problem it solves, or what it describes. \
Write only the summary, without introductory phrases. \
Respond in the same language as the original document.

Document:
{doc_text}"""

_SUMMARY_PROMPT_WITH_PARENTS = """\
Briefly describe the document's content, considering its place in the hierarchy. \
Write only the summary, without introductory phrases. \
Respond in the same language as the original document.

Parent documents context:
{parent_context}

Document:
{doc_text}"""


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
    async def process(self, chunk: Chunk, document: Document) -> Chunk:
        """Обработать чанк и вернуть обновлённую версию."""
        ...

    async def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Батчевая обработка чанков. По умолчанию вызывает process() по одному."""
        return [await self.process(c, document) for c in chunks]


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

    async def process(self, chunk: Chunk, document: Document) -> Chunk:
        chunk.vectors['full'] = await self._embedder.embed(self._full_text(chunk))
        return chunk

    async def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Батчевый эмбеддинг всех чанков документа за один вызов."""
        if not chunks:
            return chunks
        texts = [self._full_text(c) for c in chunks]
        vectors = await self._embedder.embed_batch(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.vectors['full'] = vec
        return chunks


class SparseEmbeddingProcessor(ChunkProcessor):
    """Добавляет sparse-вектор 'keywords' в chunk.vectors.

    Вектор строится из текста чанка (+ опционально doc_summary и chunk.context)
    для улучшения лексического поиска. Сохраняется в формате
    {'indices': [...], 'values': [...]}.

    `include_doc_summary` — добавляет doc_summary документа (метаданные, вроде
    номера дела или сторон).
    `include_chunk_context` — добавляет chunk.context (Anthropic «Contextual BM25»):
    контекстуальная подсказка улучшает попадание на общих терминах.
    """

    def __init__(
        self, embedder: SparseEmbedder,
        include_doc_summary: bool = False,
        include_chunk_context: bool = False,
    ) -> None:
        self._embedder = embedder
        self._include_doc_summary = include_doc_summary
        self._include_chunk_context = include_chunk_context

    def _sparse_text(self, chunk: Chunk, document: Document) -> str:
        parts = [chunk.text]
        if self._include_chunk_context and chunk.context:
            parts.append(chunk.context)
        if self._include_doc_summary:
            doc_summary = document.payload.get('doc_summary', '')
            if doc_summary:
                parts.append(doc_summary)
        return '\n'.join(parts)

    async def process(self, chunk: Chunk, document: Document) -> Chunk:
        indices, values = await self._embedder.embed(self._sparse_text(chunk, document))
        chunk.vectors['keywords'] = {'indices': indices, 'values': values}
        return chunk

    async def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Батчевый sparse-эмбеддинг всех чанков документа за один вызов."""
        if not chunks:
            return chunks
        texts = [self._sparse_text(c, document) for c in chunks]
        results = await self._embedder.embed_batch(texts)
        for chunk, (indices, values) in zip(chunks, results):
            chunk.vectors['keywords'] = {'indices': indices, 'values': values}
        return chunks


class DocSummaryProcessor(DocumentProcessor):
    """Генерирует контекстуальное саммари документа и сохраняет в payload['doc_summary'].

    Саммари учитывает иерархию: если у документа есть родители, их саммари включается
    в промпт как контекст. Родители гарантированно обработаны раньше (BFS-порядок в pipeline).
    Структурные документы и документы без текста пропускаются.

    Подклассы могут переопределить _prompt_no_parent / _prompt_with_parents для
    кастомных промптов (например LegalDocSummaryProcessor).
    """

    _prompt_no_parent: str = _SUMMARY_PROMPT_NO_PARENT
    _prompt_with_parents: str = _SUMMARY_PROMPT_WITH_PARENTS

    def __init__(
        self,
        llm_client: LLMClient,
        doc_repo,
        max_tokens: int = 128,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._client = llm_client
        self._doc_repo = doc_repo
        self._max_tokens = max_tokens
        self._token_counter = token_counter or TiktokenCounter()
        self._overhead_no_parent = self._token_counter.count(
            self._prompt_no_parent.format(doc_text='')
        )
        self._overhead_with_parents = self._token_counter.count(
            self._prompt_with_parents.format(parent_context='', doc_text='')
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

        context_window = self._client.context_window
        if parent_summaries:
            parent_context = '\n\n'.join(parent_summaries)
            parent_tokens = self._token_counter.count(parent_context)
            available = context_window - self._overhead_with_parents - parent_tokens - self._max_tokens
            doc_text = self._token_counter.truncate(document.text, max(available, 0))
            prompt = self._prompt_with_parents.format(
                parent_context=parent_context,
                doc_text=doc_text,
            )
        else:
            available = context_window - self._overhead_no_parent - self._max_tokens
            doc_text = self._token_counter.truncate(document.text, max(available, 0))
            prompt = self._prompt_no_parent.format(doc_text=doc_text)

        messages = [{'role': 'user', 'content': prompt}]
        summary = await self._client.complete(messages, max_tokens=self._max_tokens)
        document.payload['doc_summary'] = summary.strip()
        logger.info('DocSummaryProcessor: %s (%d chars)', document.id, len(summary))

        return document


_LEGAL_SUMMARY_PROMPT = """\
Analyze the following document and produce a structured summary in English.

If this is a COURT CASE / LEGAL ORDER / JUDGMENT, extract:
- Document type (e.g. Court Order, Judgment, Application, Appeal)
- Case number (e.g. SCT 295/2025, CFI 057/2025)
- Court/Tribunal name
- Date of Issue
- Judge(s) name(s)
- Claimant(s) / Applicant(s) — full names
- Defendant(s) / Respondent(s) — full names
- Brief outcome (1-2 sentences)
- Key monetary amounts if any

If this is a LAW / REGULATION / STATUTE, extract:
- Document type (Law, Regulation, Amendment)
- Official title and law number (e.g. DIFC Law No. 7 of 2018)
- Subject matter (1 sentence)
- Date of enactment / consolidation
- Key articles mentioned

Format as a structured block, one field per line. Example:

Type: Court Order
Case: SCT 295/2025
Court: DIFC Small Claims Tribunal
Date of Issue: 10 December 2025
Judge: H.E. Justice Shamlan Al Sawalehi
Claimant: OLEXA
Defendant: ODON
Outcome: Permission to Appeal Application refused. No order as to costs.

Document:
{doc_text}"""

_LEGAL_SUMMARY_PROMPT_WITH_PARENTS = """\
Analyze the following document and produce a structured summary in English. \
Consider the parent documents context for additional information.

If this is a COURT CASE / LEGAL ORDER / JUDGMENT, extract:
- Document type (e.g. Court Order, Judgment, Application, Appeal)
- Case number (e.g. SCT 295/2025, CFI 057/2025)
- Court/Tribunal name
- Date of Issue
- Judge(s) name(s)
- Claimant(s) / Applicant(s) — full names
- Defendant(s) / Respondent(s) — full names
- Brief outcome (1-2 sentences)
- Key monetary amounts if any

If this is a LAW / REGULATION / STATUTE, extract:
- Document type (Law, Regulation, Amendment)
- Official title and law number (e.g. DIFC Law No. 7 of 2018)
- Subject matter (1 sentence)
- Date of enactment / consolidation
- Key articles mentioned

Format as a structured block, one field per line. Example:

Type: Court Order
Case: SCT 295/2025
Court: DIFC Small Claims Tribunal
Date of Issue: 10 December 2025
Judge: H.E. Justice Shamlan Al Sawalehi
Claimant: OLEXA
Defendant: ODON
Outcome: Permission to Appeal Application refused. No order as to costs.

Parent documents context:
{parent_context}

Document:
{doc_text}"""


class LegalDocSummaryProcessor(DocSummaryProcessor):
    """Извлекает структурированные метаданные из юридических документов.

    Для судебных актов: тип, номер дела, суд, дата, судья, стороны, исход, суммы.
    Для законов: тип, название, номер, предмет, дата, ключевые статьи.
    """

    _prompt_no_parent: str = _LEGAL_SUMMARY_PROMPT
    _prompt_with_parents: str = _LEGAL_SUMMARY_PROMPT_WITH_PARENTS


_TITLE_PROMPT = """\
Determine the document title from its content. \
If the text contains an explicit title or title page, use it. \
If there is no explicit title, compose a short title that reflects the document's subject.

Requirements:
- Return ONLY the title, without quotes, explanations, or introductory phrases.
- The title must be in the same language as the original document.
- The title must be concise (up to 10 words).

Document:
{doc_text}\
"""


_SCAN_PAGES_RE = re.compile(r'<!-- page:\d+ -->')


class DocTitleProcessor(DocumentProcessor):
    """Определяет название документа через LLM и сохраняет в payload['title'].

    Два режима определения фрагмента для анализа:
    - scan_pages: если задано и в тексте есть маркеры <!-- page:N -->,
      берёт первые N страниц (текст до маркера страницы N+1).
    - scan_tokens: fallback — первые N токенов от начала документа.
    Структурные документы и документы без текста пропускаются.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 64,
        scan_tokens: int = 32768,
        scan_pages: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._client = llm_client
        self._max_tokens = max_tokens
        self._scan_tokens = scan_tokens
        self._scan_pages = scan_pages
        self._token_counter = token_counter or TiktokenCounter()
        self._prompt_overhead = self._token_counter.count(
            _TITLE_PROMPT.format(doc_text='')
        )

    @staticmethod
    def _extract_pages(text: str, n_pages: int) -> str | None:
        """Извлечь первые n_pages страниц по маркерам <!-- page:N -->."""
        markers = list(_SCAN_PAGES_RE.finditer(text))
        if not markers:
            return None
        # Берём текст от начала до маркера страницы n_pages+1
        if len(markers) > n_pages:
            return text[:markers[n_pages].start()].rstrip()
        return text

    # Source types где title уже задан людьми (Confluence page tree, Jira tasks)
    _SKIP_SOURCE_TYPES = frozenset({'confluence', 'attached_jira'})

    async def process(self, document: Document) -> Document:
        if document.structural or not document.text.strip():
            return document

        if document.source_type in self._SKIP_SOURCE_TYPES:
            logger.debug('DocTitleProcessor: skipping %s (source_type=%s, title from source)',
                         document.id, document.source_type)
            return document

        # Попробовать взять первые N страниц
        doc_text: str | None = None
        if self._scan_pages is not None:
            doc_text = self._extract_pages(document.text, self._scan_pages)

        context_window = self._client.context_window
        # Fallback на scan_tokens
        if doc_text is None:
            scan_limit = min(
                self._scan_tokens,
                context_window - self._prompt_overhead - self._max_tokens,
            )
            if scan_limit <= 0:
                logger.warning('DocTitleProcessor: no room for doc text, skipping %s', document.id)
                return document
            doc_text = self._token_counter.truncate(document.text, scan_limit)

        # Обрезать по контекстному окну в любом случае
        max_doc_tokens = context_window - self._prompt_overhead - self._max_tokens
        if max_doc_tokens <= 0:
            logger.warning('DocTitleProcessor: no room for doc text, skipping %s', document.id)
            return document
        doc_text = self._token_counter.truncate(doc_text, max_doc_tokens)

        prompt = _TITLE_PROMPT.format(doc_text=doc_text)
        messages = [{'role': 'user', 'content': prompt}]
        try:
            title = await self._client.complete(
                messages, max_tokens=self._max_tokens,
            )
            document.title = title.strip()
            document.path = [title.strip()]
            logger.info('DocTitleProcessor: %s → %s', document.id, title.strip())
        except Exception:
            logger.warning(
                'DocTitleProcessor: LLM call failed for %s, title skipped',
                document.id, exc_info=True,
            )

        return document


_PAGE_MARKER_RE = re.compile(r'<!-- page:(\d+) -->\n?')


class PageMarkerProcessor(ChunkProcessor):
    """Извлекает номера страниц из маркеров <!-- page:N --> в тексте чанка.

    Сохраняет список страниц в chunk.payload['pages'] и удаляет маркеры из текста,
    чтобы они не попали в эмбеддинги. Должен выполняться до embedding-процессоров.

    При батчевой обработке отслеживает «текущую страницу»: если чанк не содержит
    маркера, он наследует последнюю известную страницу от предыдущих чанков.
    """

    async def process(self, chunk: Chunk, document: Document) -> Chunk:
        markers = _PAGE_MARKER_RE.findall(chunk.text)
        if markers:
            chunk.payload['pages'] = sorted({int(m) for m in markers})
        chunk.text = _PAGE_MARKER_RE.sub('', chunk.text)
        return chunk

    async def process_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        last_page: int | None = None
        for chunk in chunks:
            markers = _PAGE_MARKER_RE.findall(chunk.text)
            if markers:
                pages = sorted({int(m) for m in markers})
                chunk.payload['pages'] = pages
                last_page = pages[-1]
            elif last_page is not None:
                chunk.payload['pages'] = [last_page]
            chunk.text = _PAGE_MARKER_RE.sub('', chunk.text)
        return chunks


class MetadataProcessor(ChunkProcessor):
    """Копирует creator и created_at из Document в chunk.payload.

    Позволяет использовать метаданные автора и даты создания
    в результатах поиска без дополнительных запросов к коллекции docs.
    """

    async def process(self, chunk: Chunk, document: Document) -> Chunk:
        chunk.payload['source_type'] = document.source_type
        if document.creator is not None:
            chunk.payload['creator'] = document.creator
        if document.created_at is not None:
            chunk.payload['created_at'] = document.created_at.isoformat()
        if document.url is not None:
            chunk.payload['url'] = document.url
        return chunk
