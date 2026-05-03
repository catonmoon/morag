"""Источник PDF-вложений из Confluence.

Обнаруживает PDF-файлы, прикреплённые к страницам Confluence, скачивает их
и конвертирует в Markdown через PdfConverter. Каждое вложение — отдельный Document
с parent_doc_ids=[page_id].

Запускается после ConfluenceSource, чтобы родительские страницы уже были в базе.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

from atlassian import Confluence

from morag.config import ConfluenceSourceConfig
from morag.sources.base import Document, Source
from morag.sources.pdf_converter import PdfConverter
from morag.storage.repository import DocRepository

logger = logging.getLogger(__name__)

# Лимит вложений за один запрос (Confluence API)
_ATTACHMENT_PAGE_SIZE = 50


class ConfluencePdfSource(Source):
    """Источник PDF-вложений из Confluence.

    Для каждой страницы Confluence запрашивает вложения, фильтрует по mime_types,
    скачивает PDF и конвертирует через PdfConverter.
    """

    @property
    def source_type(self) -> str:
        return 'attached_pdf'

    def __init__(
        self,
        config: ConfluenceSourceConfig,
        converter: PdfConverter,
        doc_repo: DocRepository,
    ) -> None:
        # password / api_token проверяется в Pydantic-валидаторе ConfluenceSourceConfig
        credential = config.api_token or config.password

        # PDF-вложения принадлежат тому же ConfluenceSource (тот же config.name)
        self._kind = 'confluence'
        self._name = config.name

        self._client = Confluence(
            url=config.url,
            username=config.username,
            password=credential,
            cloud=config.api_token is not None,
            timeout=config.timeout,
            backoff_and_retry=config.max_retries > 0,
            max_backoff_retries=config.max_retries,
        )
        self._base_url = config.url.rstrip('/')
        self._mime_types = set(config.attachments.mime_types)
        self._skip_ancestor_ids = set(config.attachments.skip_ancestor_ids)
        self._url_mode = config.attachments.url_mode
        self._timeout = config.timeout
        self._converter = converter
        self._doc_repo = doc_repo

        # Кеш: attachment_id → (page_id, download_url, filename)
        self._attachment_meta: dict[str, tuple[str, str, str]] = {}

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы PDF-вложений: обход всех проиндексированных страниц Confluence."""
        # Получаем все page_id из doc_repo (source_type='confluence').
        # IDs приходят prefixed: 'confluence:<name>:<page_id>'
        page_ids = await self._doc_repo.get_ids_by_source_type('confluence')
        if not page_ids:
            logger.info('No Confluence pages found, skipping PDF attachments')
            return []

        # Префиксим skip_ancestor_ids для совпадения с prefixed-IDs из repo
        skip_prefixed = {self.make_id(pid) for pid in self._skip_ancestor_ids}
        pages_to_skip = await self._collect_descendants(skip_prefixed)
        if pages_to_skip:
            before = len(page_ids)
            page_ids = page_ids - pages_to_skip
            logger.info(
                'Skipping attachments for %d page(s) (skip_ancestor_ids descendants), %d page(s) remaining',
                before - len(page_ids), len(page_ids),
            )

        if not page_ids:
            logger.info('All pages excluded by skip_ancestor_ids, no attachments to scan')
            return []

        logger.info('Scanning %d Confluence page(s) for PDF attachments...', len(page_ids))

        stubs: list[Document] = []
        for page_id in sorted(page_ids):
            try:
                page_stubs = await self._fetch_page_attachments(page_id)
                stubs.extend(page_stubs)
            except Exception:
                logger.exception(
                    'Failed to fetch attachments for page %s', page_id,
                )

        logger.info('Found %d PDF attachment(s) across %d page(s)', len(stubs), len(page_ids))
        return stubs

    async def _collect_descendants(self, ancestor_ids: set[str]) -> set[str]:
        """BFS от ancestor_ids вниз по иерархии, собрать всех потомков (включая сами ancestor_ids)."""
        if not ancestor_ids:
            return set()

        result: set[str] = set(ancestor_ids)
        queue: deque[str] = deque(ancestor_ids)

        while queue:
            parent_id = queue.popleft()
            children = await self._doc_repo.find_children(parent_id)
            for child in children:
                if child.id not in result:
                    result.add(child.id)
                    queue.append(child.id)

        return result

    def _strip_prefix(self, doc_id: str) -> str:
        prefix = f'{self._kind}:{self._name}:'
        return doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id

    async def load_one(self, doc_id: str) -> Document | None:
        """Скачать PDF-вложение и конвертировать через PdfConverter.

        doc_id приходит prefixed; ищем по нему в кеше (ключи кеша тоже prefixed).
        """
        meta = self._attachment_meta.get(doc_id)
        if meta is None:
            logger.warning('Attachment metadata not found for %s', doc_id)
            return None

        page_id, download_url, filename = meta

        try:
            pdf_bytes = await asyncio.to_thread(
                self._download_attachment, download_url,
            )
            if pdf_bytes is None:
                return None

            markdown = await self._converter.convert(pdf_bytes, filename)
            if markdown is None:
                logger.error('Failed to convert PDF attachment: %s', doc_id)
                return None

            parent_doc = await self._doc_repo.get_by_id(page_id)
            if parent_doc is not None and parent_doc.path:
                att_path = [f'{parent_doc.path[0]}/{filename}']
            else:
                att_path = [f'{page_id}/{filename}']

            # updated_at наследуется от родительской страницы: вложение переиндексируется
            # только при изменении страницы
            updated_at = parent_doc.updated_at if parent_doc is not None else datetime.now(tz=timezone.utc)

            url = await self._build_attachment_url(
                page_id, download_url, filename,
            )

            return Document(
                id=doc_id,                # уже prefixed (получили его prefixed на входе)
                path=att_path,
                text=markdown,
                updated_at=updated_at,
                source_type='attached_pdf',
                title=filename,
                size=len(pdf_bytes),
                url=url,
                parent_doc_ids=[page_id],  # page_id уже prefixed (из get_ids_by_source_type)
                paged=True,
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except Exception:
            logger.exception('Failed to load PDF attachment %s', doc_id)
            return None

    async def _build_attachment_url(
        self, page_id: str, download_url: str, filename: str,
    ) -> str | None:
        """Сформировать URL вложения в зависимости от url_mode."""
        if self._url_mode == 'download':
            if download_url.startswith('http'):
                return download_url
            return f'{self._base_url}{download_url}'

        if self._url_mode == 'parent_page':
            parent_doc = await self._doc_repo.get_by_id(page_id)
            if parent_doc is not None and parent_doc.url:
                return parent_doc.url
            return None

        # preview (default) — on-premise Confluence
        return (
            f'{self._base_url}/pages/viewpageattachments.action'
            f'?pageId={page_id}&preview=/{page_id}/{filename}'
        )

    async def _fetch_page_attachments(self, page_id: str) -> list[Document]:
        """Получить PDF-вложения страницы через Confluence API.

        page_id приходит prefixed (из get_ids_by_source_type). Confluence API
        ничего не знает про префиксы — нужен raw external_id.
        """
        external_page_id = self._strip_prefix(page_id)
        attachments = await asyncio.to_thread(
            self._get_attachments_sync, external_page_id,
        )

        # parent lookup в repo использует prefixed id (id в Qdrant tоже prefixed)
        parent_doc = await self._doc_repo.get_by_id(page_id)
        parent_path = parent_doc.path[0] if parent_doc is not None and parent_doc.path else external_page_id
        updated_at = parent_doc.updated_at if parent_doc is not None else datetime.now(tz=timezone.utc)

        stubs: list[Document] = []
        for att in attachments:
            media_type = att.get('metadata', {}).get('mediaType', '')
            if media_type not in self._mime_types:
                continue

            att_id = att['id'].removeprefix('att')
            # ID composite: префикс конфлюенс-инстанса + att-id (att_id уникален в рамках инстанса)
            doc_id = self.make_id(f'att:{att_id}')
            filename = att.get('title', f'attachment_{att_id}.pdf')

            download_url = att.get('_links', {}).get('download', '')

            # Кешируем метаданные для load_one (ключ — prefixed doc_id)
            self._attachment_meta[doc_id] = (page_id, download_url, filename)

            stubs.append(Document(
                id=doc_id,
                path=[f'{parent_path}/{filename}'],
                text='',
                updated_at=updated_at,
                source_type='attached_pdf',
                title=filename,
                size=0,
                parent_doc_ids=[page_id],  # page_id уже prefixed
                paged=True,
                payload={'source_name': self._name, 'source_kind': self._kind},
            ))

        return stubs

    def _get_attachments_sync(self, page_id: str) -> list[dict]:
        """Sync-запрос вложений страницы через Confluence API с пагинацией."""
        all_attachments: list[dict] = []
        start = 0

        while True:
            result = self._client.get_attachments_from_content(
                page_id, start=start, limit=_ATTACHMENT_PAGE_SIZE,
                expand='version',
            )
            batch = result.get('results', [])
            all_attachments.extend(batch)

            if len(batch) < _ATTACHMENT_PAGE_SIZE:
                break
            start += _ATTACHMENT_PAGE_SIZE

        return all_attachments

    def _download_attachment(self, download_url: str) -> bytes | None:
        """Скачать вложение через сессию Confluence (с авторизацией)."""
        try:
            url = download_url if download_url.startswith('http') else f'{self._base_url}{download_url}'
            response = self._client._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            return response.content
        except Exception:
            logger.exception('Failed to download attachment: %s', download_url)
            return None


