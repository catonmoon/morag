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
from morag.llm.client import GenerationParams
from morag.sources.base import Document, Source

logger = logging.getLogger(__name__)

# Максимальное количество страниц за один CQL-запрос (ограничение Confluence API)
_CQL_PAGE_SIZE = 200

_IMAGE_PROMPT = (
    "Твоя задача проанализировать и представить в текстовом markdown виде переданное изображение которое встретилось в пользовательской документации.\n"
    "Если изображение это скриншот текстового терминала, то необходимо выдать заголовок «скриншот терминала» и вывести его текстовое содержимое.\n"
    "Если изображение это диаграмма или векторная фигура, представимая в виде plantuml, то это должен быть plantuml код и ничего больше.\n"
    "Если изображение это скриншот программной формы, то необходимо выдать заголовок «скриншот формы» и далее сжатое текстовое описание этой формы, ее назначение и важные поля.\n"
    "Если изображение это скриншот HTML, то необходимо выдать заголовок «скриншот HTML» и далее сжатое текстовое описание, назначение и важные элементы.\n"
    "Во всех остальных случаях нужно вывести заголовок «изображение» и краткое описание изображения."
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

    @property
    def source_type(self) -> str:
        return 'confluence'

    def __init__(self, config: ConfluenceConfig, vision_client=None, vision_max_tokens: int | None = None) -> None:
        credential = config.api_token or config.password
        if not credential:
            raise ValueError('Confluence config requires either api_token or password')

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
        self._spaces = config.spaces
        self._ancestor_ids = config.ancestor_ids
        self._skip_ancestor_ids = config.skip_ancestor_ids
        self._vision_client = vision_client
        self._vision_max_tokens = vision_max_tokens
        self._min_image_size_bytes = config.min_image_size_bytes
        self._timeout = config.timeout

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы страниц Confluence: только метаданные, без тела и изображений."""
        pages = self._fetch_pages_metadata()
        logger.info('Fetched metadata for %d pages from Confluence', len(pages))

        stubs: list[Document] = []
        for page in pages:
            try:
                stub = self._page_to_stub(page)
                if stub is not None:
                    stubs.append(stub)
            except Exception:
                page_id = page.get('content', {}).get('id') or page.get('id', '?')
                logger.exception('Failed to get metadata for page id=%s', page_id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить одну страницу Confluence целиком по page_id."""
        try:
            page = await asyncio.to_thread(
                self._client.get_page_by_id,
                doc_id,
                expand='body.view,history.lastUpdated,history.createdBy,history.createdDate,space,ancestors',
            )
            return await self._page_to_document(page)
        except Exception:
            logger.exception('Failed to load page id=%s', doc_id)
            return None

    def _fetch_pages_metadata(self) -> list[dict]:
        """Получить метаданные страниц через CQL без тела (быстрый запрос)."""
        cql = self._build_cql()
        logger.info('Confluence CQL: %s', cql)

        pages: list[dict] = []
        start = 0

        while True:
            result = self._client.cql(
                cql,
                start=start,
                limit=_CQL_PAGE_SIZE,
                expand='content.history.lastUpdated,content.space,content.ancestors',
            )
            batch = result.get('results', [])
            pages.extend(batch)

            if len(batch) < _CQL_PAGE_SIZE:
                break
            start += _CQL_PAGE_SIZE
            logger.debug('Fetched metadata for %d pages so far, continuing...', len(pages))

        return pages

    def _page_to_stub(self, page: dict) -> Document | None:
        """Создать стаб Document из метаданных страницы (без тела)."""
        content = page.get('content', page)
        page_id = content['id']
        title = content['title']
        space = content.get('space', {})
        space_name = space.get('name') or space.get('key', 'UNKNOWN')
        ancestor_objects = content.get('ancestors', [])
        ancestors = [a['title'] for a in ancestor_objects if a.get('title')]
        parent_doc_ids = [ancestor_objects[-1]['id']] if ancestor_objects else []
        history = content.get('history', {})

        last_updated = history.get('lastUpdated', {}).get('when', '')
        updated_at = _parse_confluence_date(last_updated)
        path = _build_page_path(space_name, ancestors, title)

        links = content.get('_links', {})
        webui = links.get('webui', '')
        url = f'{links.get("base", self._base_url)}{webui}' if webui else None

        return Document(
            id=page_id,
            path=[path],
            text='',
            updated_at=updated_at,
            source_type='confluence',
            size=0,
            url=url,
            parent_doc_ids=parent_doc_ids,
        )

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
            content = page.get('content', page)
            page_id = content['id']
            title = content['title']
            space = content.get('space', {})
            space_name = space.get('name') or space.get('key', 'UNKNOWN')
            ancestor_objects = content.get('ancestors', [])
            ancestors = [a['title'] for a in ancestor_objects if a.get('title')]
            parent_doc_ids = [ancestor_objects[-1]['id']] if ancestor_objects else []
            history = content.get('history', {})
            html = content.get('body', {}).get('view', {}).get('value', '')
            last_updated = history.get('lastUpdated', {}).get('when', '')

            creator = history.get('createdBy', {}).get('displayName') or None
            created_date_raw = history.get('createdDate') or history.get('createdAt')
            created_at = _parse_confluence_date(created_date_raw) if created_date_raw else None

            markdown = await self._process_html(html, page_id)
            text = f'# {title}\n\n{markdown}'.strip()
            structural = not html.strip()

            updated_at = _parse_confluence_date(last_updated)
            path = _build_page_path(space_name, ancestors, title)

            links = content.get('_links', {})
            webui = links.get('webui', '')
            url = f'{links.get("base", self._base_url)}{webui}' if webui else None

            return Document(
                id=page_id,
                path=[path],
                text=text,
                updated_at=updated_at,
                source_type='confluence',
                size=len(text.encode('utf-8')),
                url=url,
                creator=creator,
                created_at=created_at,
                parent_doc_ids=parent_doc_ids,
                structural=structural,
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

        logger.debug('Page %s HTML (before cleanup):\n%s', page_id, soup)

        _clean_jira_macros(soup)
        _remove_vendor_ui_blocks(soup)

        if self._vision_client:
            await self._replace_images_with_descriptions(soup, page_id)

        return markdownify(str(soup), heading_style='ATX', bullets='-').strip()

    async def _replace_images_with_descriptions(self, soup: BeautifulSoup, page_id: str) -> None:
        """Скачать изображения и заменить <img> тегами с описанием от vision LLM."""
        imgs = soup.find_all('img')
        if not imgs:
            return

        logger.info('Page %s: processing %d image(s) with vision LLM', page_id, len(imgs))

        tasks = [self._describe_image(img.get('src', ''), img.get('alt', ''), page_id) for img in imgs]
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        for img, description in zip(imgs, descriptions):
            if isinstance(description, Exception) or not description:
                img.decompose()
            else:
                p = soup.new_tag('p')
                p.string = f'[Изображение: {description}]'
                img.replace_with(p)

    async def _describe_image(self, src: str, alt: str, page_id: str) -> str | None:
        """Скачать изображение по URL и получить описание от vision LLM."""
        if not src:
            return alt or None

        media_type = _guess_media_type(src)
        if media_type == 'image/svg+xml':
            logger.info('Page %s: skipping SVG image: %s', page_id, src)
            return alt or None

        image_bytes = await asyncio.to_thread(self._download_image, src, page_id)
        if not image_bytes:
            return alt or None

        if self._min_image_size_bytes is not None and len(image_bytes) < self._min_image_size_bytes:
            logger.info(
                'Page %s: skipping small image (%d bytes < %d): %s',
                page_id, len(image_bytes), self._min_image_size_bytes, src,
            )
            return alt or None

        image_b64 = base64.b64encode(image_bytes).decode('ascii')

        try:
            description = await self._vision_client.complete_vision(
                _IMAGE_PROMPT, image_b64, media_type,
                max_tokens=self._vision_max_tokens,
                params=GenerationParams(seed=42),
            )
            logger.info('Page %s: image described: %s -> %s...', page_id, src[:80], description[:80].replace('\n', '\\n'))
            return description.strip() or None
        except Exception:
            logger.warning('Page %s: vision LLM failed for image %s', page_id, src, exc_info=True)
            return alt or None

    def _download_image(self, src: str, page_id: str) -> bytes | None:
        """Синхронно скачать изображение, используя сессию Confluence (с авторизацией)."""
        try:
            url = src if src.startswith('http') else urljoin(self._base_url, src)
            # Используем сессию atlassian клиента — она уже содержит авторизацию
            response = self._client._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            return response.content
        except Exception:
            logger.warning('Page %s: failed to download image: %s', page_id, src, exc_info=True)
            return None


# Домены, характерные для UI-артефактов плагинов Confluence (Table Filter and Charts и т.п.)
_VENDOR_UI_DOMAINS = ('docs.stiltsoft.com', 'survey.alchemer.com')
# Теги, при достижении которых прекращаем подъём по DOM (структурные контейнеры)
_BLOCK_STOP_TAGS = frozenset({'body', 'html', 'table', 'tr', 'td', 'th'})
# Теги блочных контейнеров — кандидаты на удаление как корень макроса
_BLOCK_CONTAINER_TAGS = frozenset({'div', 'section', 'aside', 'nav', 'header', 'footer', 'ul', 'ol'})


def _clean_jira_macros(soup: BeautifulSoup) -> None:
    """Очистить Jira-макросы Confluence: оставить только ссылку на задачу.

    Confluence рендерит Jira-макрос как:
      <span data-macro-name="jira">
        <a class="jira-issue-key" href="..."><span class="aui-icon"> </span>MODP-12345</a>
        - <span class="summary">Getting issue details...</span>
        <span class="aui-lozenge ...">STATUS</span>
      </span>

    Заменяем весь span на чистый <a href="...">MODP-12345</a>.
    """
    for macro in soup.find_all('span', attrs={'data-macro-name': 'jira'}):
        link = macro.find('a', class_='jira-issue-key')
        if link:
            # убираем иконку-placeholder внутри ссылки
            for icon in link.find_all('span', class_='aui-icon'):
                icon.decompose()
            macro.replace_with(link)
        else:
            macro.decompose()


def _remove_vendor_ui_blocks(soup: BeautifulSoup) -> None:
    """Удалить блоки UI-артефактов плагинов по ссылкам на домены вендоров.

    Ищет <a> теги с href на известные домены (stiltsoft, alchemer),
    поднимается по DOM до наибольшего блочного предка и удаляет его целиком.
    Это убирает мусор от макросов типа Table Filter and Charts из Confluence.
    """
    to_remove: set = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href', '')
        if not any(domain in href for domain in _VENDOR_UI_DOMAINS):
            continue
        target = a_tag
        for parent in a_tag.parents:
            if parent.name in _BLOCK_STOP_TAGS:
                break
            if parent.name in _BLOCK_CONTAINER_TAGS:
                target = parent
        to_remove.add(target)
    for el in to_remove:
        el.decompose()


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


def _build_page_path(space_name: str, ancestors: list[str], title: str) -> str:
    """Построить путь страницы в виде RootPage/Ancestor/Title.

    Имя пространства в путь не включается — путь всегда начинается с корневой страницы.
    Если предков нет, путь содержит только заголовок страницы.
    """
    parts = ancestors + [title]
    return '/'.join(parts)


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
