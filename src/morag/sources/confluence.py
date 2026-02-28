from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from atlassian import Confluence
from bs4 import BeautifulSoup
from markdownify import markdownify

from morag.config import ConfluenceConfig
from morag.sources.base import Document, Source

logger = logging.getLogger(__name__)

# Максимальное количество страниц за один CQL-запрос (ограничение Confluence API)
_CQL_PAGE_SIZE = 200

_IMAGE_PROMPT = (
    'Опиши это изображение кратко и точно на русском языке. '
    'Если это диаграмма, схема или таблица — опиши её структуру и содержание. '
    'Если это скриншот интерфейса — опиши что на нём показано. '
    'Ответь одним абзацем без вступительных фраз.'
)


class ConfluenceSource(Source):
    """Источник страниц из Confluence.

    Загружает страницы через Confluence REST API, конвертирует HTML в Markdown.
    Поддерживает on-premise (username + password) и Cloud (username + api_token).

    Если передан vision_client, изображения на страницах скачиваются и описываются
    через multimodal LLM, заменяя <img> тег текстовым описанием в Markdown.

    Фильтрация (опционально):
    - spaces: список space key — загружать только из указанных пространств
    - ancestor_ids: список page id — загружать только потомков указанных страниц
    """

    def __init__(self, config: ConfluenceConfig, vision_client=None) -> None:
        credential = config.api_token or config.password
        if not credential:
            raise ValueError('Confluence config requires either api_token or password')

        self._client = Confluence(
            url=config.url,
            username=config.username,
            password=credential,
            cloud=config.api_token is not None,
        )
        self._base_url = config.url.rstrip('/')
        self._spaces = config.spaces
        self._ancestor_ids = config.ancestor_ids
        self._skip_ancestor_ids = config.skip_ancestor_ids
        self._vision_client = vision_client

    async def load(self) -> list[Document]:
        """Загрузить все страницы из Confluence согласно конфигу."""
        pages = self._fetch_pages()
        logger.info('Fetched %d pages from Confluence', len(pages))

        docs: list[Document] = []
        for page in pages:
            doc = await self._page_to_document(page)
            if doc is not None:
                docs.append(doc)

        logger.info('Converted %d pages to documents', len(docs))
        return docs

    def _fetch_pages(self) -> list[dict]:
        """Получить все страницы через CQL с пагинацией."""
        cql = self._build_cql()
        logger.info('Confluence CQL: %s', cql)

        pages: list[dict] = []
        start = 0

        while True:
            result = self._client.cql(
                cql,
                start=start,
                limit=_CQL_PAGE_SIZE,
                expand='content.body.view,content.history.lastUpdated,content.history.createdBy,content.history.createdDate,content.space',
            )
            batch = result.get('results', [])
            pages.extend(batch)

            if len(batch) < _CQL_PAGE_SIZE:
                break
            start += _CQL_PAGE_SIZE
            logger.debug('Fetched %d pages so far, continuing...', len(pages))

        return pages

    def _build_cql(self) -> str:
        """Построить CQL-запрос с учётом фильтров."""
        parts = ['type = page']

        if self._ancestor_ids:
            ids = ', '.join(f'"{pid}"' for pid in self._ancestor_ids)
            parts.append(f'(ancestor IN ({ids}) OR id IN ({ids}))')
        elif self._spaces:
            keys = ', '.join(f'"{s}"' for s in self._spaces)
            parts.append(f'space IN ({keys})')

        if self._skip_ancestor_ids:
            ids = ', '.join(f'"{pid}"' for pid in self._skip_ancestor_ids)
            # исключаем потомков и сами корневые страницы
            parts.append(f'ancestor NOT IN ({ids})')
            parts.append(f'id NOT IN ({ids})')

        parts.append('ORDER BY lastmodified DESC')
        return ' AND '.join(parts[:-1]) + ' ' + parts[-1]

    async def _page_to_document(self, page: dict) -> Document | None:
        """Конвертировать страницу Confluence в Document."""
        try:
            page_id = page['content']['id'] if 'content' in page else page['id']
            title = page['content']['title'] if 'content' in page else page['title']
            space_key = (
                page['content'].get('space', {}).get('key', 'UNKNOWN')
                if 'content' in page
                else page.get('space', {}).get('key', 'UNKNOWN')
            )

            if 'content' in page:
                history = page['content'].get('history', {})
                html = page['content'].get('body', {}).get('view', {}).get('value', '')
                last_updated = history.get('lastUpdated', {}).get('when', '')
            else:
                history = page.get('history', {})
                html = page.get('body', {}).get('view', {}).get('value', '')
                last_updated = history.get('lastUpdated', {}).get('when', '')

            creator = history.get('createdBy', {}).get('displayName') or None
            created_date_raw = history.get('createdDate') or history.get('createdAt')
            created_at = _parse_confluence_date(created_date_raw) if created_date_raw else None

            markdown = await self._process_html(html, page_id)
            text = f'# {title}\n\n{markdown}'.strip()

            updated_at = _parse_confluence_date(last_updated)
            path = f'{space_key}/{title}'

            return Document(
                id=page_id,
                path=path,
                text=text,
                updated_at=updated_at,
                source_type='confluence',
                size=len(text.encode('utf-8')),
                creator=creator,
                created_at=created_at,
            )
        except Exception:
            page_id_for_log = (
                page.get('content', {}).get('id') or page.get('id', '?')
            )
            logger.exception('Failed to convert page id=%s', page_id_for_log)
            return None

    async def _process_html(self, html: str, page_id: str) -> str:
        """Конвертировать HTML Confluence в Markdown, опционально описывая изображения."""
        if not html:
            return ''
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['script', 'style']):
            tag.decompose()

        if self._vision_client:
            await self._replace_images_with_descriptions(soup, page_id)

        return markdownify(str(soup), heading_style='ATX', bullets='-').strip()

    async def _replace_images_with_descriptions(self, soup: BeautifulSoup, page_id: str) -> None:
        """Скачать изображения и заменить <img> тегами с описанием от vision LLM."""
        imgs = soup.find_all('img')
        if not imgs:
            return

        logger.debug('Page %s: processing %d image(s) with vision LLM', page_id, len(imgs))

        tasks = [self._describe_image(img.get('src', ''), img.get('alt', '')) for img in imgs]
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        for img, description in zip(imgs, descriptions):
            if isinstance(description, Exception) or not description:
                img.decompose()
            else:
                p = soup.new_tag('p')
                p.string = f'[Изображение: {description}]'
                img.replace_with(p)

    async def _describe_image(self, src: str, alt: str) -> str | None:
        """Скачать изображение по URL и получить описание от vision LLM."""
        if not src:
            return alt or None

        image_bytes = await asyncio.to_thread(self._download_image, src)
        if not image_bytes:
            return alt or None

        media_type = _guess_media_type(src)
        image_b64 = base64.b64encode(image_bytes).decode('ascii')

        try:
            description = await self._vision_client.complete_vision(
                _IMAGE_PROMPT, image_b64, media_type,
            )
            logger.debug('Image described: %s -> %s...', src[:60], description[:80])
            return description.strip() or None
        except Exception:
            logger.warning('Vision LLM failed for image: %s', src[:60])
            return alt or None

    def _download_image(self, src: str) -> bytes | None:
        """Синхронно скачать изображение, используя сессию Confluence (с авторизацией)."""
        try:
            url = src if src.startswith('http') else urljoin(self._base_url, src)
            # Используем сессию atlassian клиента — она уже содержит авторизацию
            response = self._client._session.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception:
            logger.warning('Failed to download image: %s', src[:60])
            return None


def _html_to_markdown(html: str) -> str:
    """Конвертировать HTML в Markdown (без обработки изображений)."""
    if not html:
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    return markdownify(str(soup), heading_style='ATX', bullets='-').strip()


def _guess_media_type(src: str) -> str:
    """Угадать MIME-тип изображения по URL."""
    path = urlparse(src).path
    mime, _ = mimetypes.guess_type(path)
    return mime or 'image/png'


def _parse_confluence_date(date_str: str) -> datetime:
    """Парсить дату из Confluence API (ISO 8601) в datetime с UTC."""
    if not date_str:
        return datetime.now(tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc)
    except ValueError:
        logger.warning('Cannot parse Confluence date: %r', date_str)
        return datetime.now(tz=timezone.utc)
