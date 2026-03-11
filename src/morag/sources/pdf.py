from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import fitz
import httpx

from morag.llm.client import LLMClient
from morag.sources.base import Document, Source

logger = logging.getLogger(__name__)

# Регулярное выражение для inline base64-изображений в markdown
_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\(data:(?P<media>image/[^;]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)\)'
)

# Регулярное выражение для плейсхолдеров изображений
_IMAGE_PLACEHOLDER_RE = re.compile(r'<!--\s*image\s*-->')

# Регулярное выражение для плейсхолдеров формул
_FORMULA_PLACEHOLDER_RE = re.compile(r'<!--\s*formula-not-decoded\s*-->')

# Промпт для описания изображений
_IMAGE_PROMPT = (
    'Опиши подробно, что изображено на изображении. Если что-то не видно (низкая вероятность распознавания), '
    'такие данные не приводи.'
)

# Промпт для распознавания формул
_FORMULA_PROMPT = (
    'Обрати внимание на формулы, переведи формулу в markdown, т.е. начни и закончи с $$'
)

# Zoom для рендеринга изображений из PDF (2x = 144 DPI)
_IMAGE_ZOOM = 2.0


@dataclass
class _PictureInfo:
    """Метаданные картинки из docling JSON."""

    page_no: int
    bbox_l: float
    bbox_t: float
    bbox_r: float
    bbox_b: float
    coord_origin: str


class PdfSource(Source):
    """Источник локальных PDF-файлов.

    Рекурсивно сканирует директорию, конвертирует PDF → Markdown через docling-serve,
    обрабатывает изображения и формулы через Vision LLM.
    parent_doc_ids ссылается на структурные документы директорий (создаются DirectorySource).
    """

    @property
    def source_type(self) -> str:
        return 'pdf'

    def __init__(
        self,
        root: Path | str,
        docling_base_url: str,
        docling_timeout: int = 300,
        vision_client: LLMClient | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._docling_base_url = docling_base_url.rstrip('/')
        self._docling_timeout = docling_timeout
        self._vision_client = vision_client

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы PDF-файлов (без конвертации)."""
        all_pdf_files = sorted(self._root.rglob('*.pdf'))

        stubs: list[Document] = []
        for path in all_pdf_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить PDF: конвертировать через docling-serve, обработать изображения."""
        path = self._root / doc_id
        try:
            stat = path.stat()
        except OSError:
            return None

        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        # Конвертация через docling-serve
        result = await self._convert_pdf(path)
        if result is None:
            logger.error('Failed to convert PDF: %s', doc_id)
            return None

        markdown, pictures, formulas = result

        # Обработка изображений через Vision LLM (вырезаем из PDF по координатам)
        if self._vision_client is not None and pictures:
            markdown = await self._process_image_placeholders(markdown, pictures, path)

        # Обработка формул через Vision LLM (вырезаем из PDF по координатам)
        if self._vision_client is not None and formulas:
            markdown = await self._process_formula_placeholders(markdown, formulas, path)
        else:
            # Без vision_client или без координат — просто удаляем плейсхолдеры
            markdown = _FORMULA_PLACEHOLDER_RE.sub('', markdown)

        return Document(
            id=doc_id,
            path=[doc_id],
            text=markdown,
            updated_at=updated_at,
            source_type='pdf',
            size=stat.st_size,
            url=path.as_uri(),
            parent_doc_ids=self._parent_doc_ids(path),
        )

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Вычислить parent_doc_ids для файла."""
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [str(parent_dir.relative_to(self._root)) + '/']

    def _get_file_metadata(self, path: Path) -> Document | None:
        """Получить метаданные PDF-файла без конвертации."""
        try:
            stat = path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(path.relative_to(self._root))
            return Document(
                id=doc_id,
                path=[doc_id],
                text='',
                updated_at=updated_at,
                source_type='pdf',
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
            )
        except OSError:
            return None

    async def _convert_pdf(
        self, path: Path,
    ) -> tuple[str, list[_PictureInfo], list[_PictureInfo]] | None:
        """Отправить PDF в docling-serve как base64 и получить markdown + метаданные.

        Возвращает (markdown, pictures, formulas).
        Использует sync httpx в thread — docling-serve нестабильно работает с async-запросами.
        """
        fn = partial(self._convert_pdf_sync, path)
        return await asyncio.get_event_loop().run_in_executor(None, fn)

    def _convert_pdf_sync(
        self, path: Path,
    ) -> tuple[str, list[_PictureInfo], list[_PictureInfo]] | None:
        """Sync-вызов docling-serve."""
        url = f'{self._docling_base_url}/v1/convert/source'

        pdf_bytes = path.read_bytes()
        b64 = base64.b64encode(pdf_bytes).decode()

        to_formats = ['md']
        if self._vision_client is not None:
            to_formats.append('json')

        payload = {
            'sources': [{'kind': 'file', 'base64_string': b64, 'filename': path.name}],
            'options': {
                'to_formats': to_formats,
                'image_export_mode': 'placeholder',
                'do_ocr': False,
            },
        }

        try:
            response = httpx.post(url, json=payload, timeout=self._docling_timeout)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                return str(data), []

            doc = data.get('document', {})
            md = doc.get('md_content') or data.get('md_content') or data.get('content')
            if not md:
                logger.warning(
                    'Unexpected docling-serve response format for %s: keys=%s',
                    path.name, list(data.keys()),
                )
                return None

            # Извлечь координаты картинок и формул из JSON
            json_content = doc.get('json_content')
            pictures = self._parse_elements(json_content, 'pictures', path.name)
            formulas = self._parse_elements(json_content, 'formulas', path.name)

            return md, pictures, formulas
        except httpx.HTTPStatusError as exc:
            logger.error(
                'docling-serve HTTP error for %s: %s %s',
                path.name, exc.response.status_code, exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error('docling-serve request failed for %s: %s', path.name, exc)
            return None

    @staticmethod
    def _parse_json_content(json_content: str | dict | None, filename: str) -> dict | None:
        """Распарсить JSON-контент docling."""
        if json_content is None:
            return None
        try:
            if isinstance(json_content, str):
                return json.loads(json_content)
            return json_content
        except (json.JSONDecodeError, TypeError):
            logger.warning('Failed to parse docling JSON for %s', filename)
            return None

    @staticmethod
    def _parse_elements(
        json_content: str | dict | None,
        kind: str,
        filename: str,
    ) -> list[_PictureInfo]:
        """Извлечь координаты элементов из docling JSON.

        kind='pictures' — картинки (из jc['pictures']).
        kind='formulas' — формулы (из jc['texts'] где label='formula').
        """
        if json_content is None:
            return []

        jc = PdfSource._parse_json_content(json_content, filename)
        if jc is None:
            return []

        if kind == 'pictures':
            elements = jc.get('pictures', [])
        elif kind == 'formulas':
            elements = [t for t in jc.get('texts', []) if t.get('label') == 'formula']
        else:
            return []

        result: list[_PictureInfo] = []
        for elem in elements:
            provs = elem.get('prov', [])
            if not provs:
                continue
            prov = provs[0]
            bbox = prov.get('bbox', {})
            if not all(k in bbox for k in ('l', 't', 'r', 'b')):
                continue
            result.append(_PictureInfo(
                page_no=prov.get('page_no', 0),
                bbox_l=bbox['l'],
                bbox_t=bbox['t'],
                bbox_r=bbox['r'],
                bbox_b=bbox['b'],
                coord_origin=bbox.get('coord_origin', 'BOTTOMLEFT'),
            ))

        return result

    @staticmethod
    def _crop_image(pdf_path: Path, pic: _PictureInfo) -> str | None:
        """Вырезать изображение из PDF по координатам и вернуть base64 PNG."""
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[pic.page_no - 1]  # 0-based
            page_height = page.rect.height

            # Конвертация координат: BOTTOMLEFT → TOP-LEFT (PyMuPDF)
            if pic.coord_origin == 'BOTTOMLEFT':
                x0 = pic.bbox_l
                y0 = page_height - pic.bbox_t
                x1 = pic.bbox_r
                y1 = page_height - pic.bbox_b
            else:
                x0 = pic.bbox_l
                y0 = pic.bbox_t
                x1 = pic.bbox_r
                y1 = pic.bbox_b

            clip = fitz.Rect(x0, y0, x1, y1)
            mat = fitz.Matrix(_IMAGE_ZOOM, _IMAGE_ZOOM)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes('png')
            doc.close()

            return base64.b64encode(img_bytes).decode()
        except Exception as exc:
            logger.warning('Failed to crop image from page %d: %s', pic.page_no, exc)
            return None

    async def _process_image_placeholders(
        self,
        markdown: str,
        pictures: list[_PictureInfo],
        pdf_path: Path,
    ) -> str:
        """Заменить <!-- image --> плейсхолдеры описаниями от Vision LLM.

        Картинки вырезаются из PDF по координатам из docling JSON.
        """
        placeholders = list(_IMAGE_PLACEHOLDER_RE.finditer(markdown))
        if not placeholders:
            return markdown

        # Количество плейсхолдеров и картинок может не совпадать
        n = min(len(placeholders), len(pictures))
        if len(placeholders) != len(pictures):
            logger.warning(
                'Image placeholder count (%d) != pictures count (%d), processing %d',
                len(placeholders), len(pictures), n,
            )

        logger.info('Processing %d image(s) from PDF via Vision LLM...', n)

        # Вырезаем картинки в thread pool
        loop = asyncio.get_event_loop()
        crop_tasks = [
            loop.run_in_executor(None, self._crop_image, pdf_path, pictures[i])
            for i in range(n)
        ]
        cropped_images = await asyncio.gather(*crop_tasks)

        # Описываем через Vision LLM
        descriptions: list[str | None] = [None] * n
        for i, img_b64 in enumerate(cropped_images):
            if img_b64 is None:
                continue
            try:
                desc = await self._vision_client.complete_vision(
                    _IMAGE_PROMPT, img_b64, media_type='image/png',
                )
                descriptions[i] = desc
            except Exception as exc:
                logger.warning('Failed to describe image %d: %s', i + 1, exc)

        # Заменяем плейсхолдеры в обратном порядке
        for i in range(n - 1, -1, -1):
            match = placeholders[i]
            img_num = i + 1
            if descriptions[i]:
                replacement = (
                    f'\n\n**[Изображение {img_num}]**\n\n'
                    f'*Описание:* {descriptions[i]}\n\n'
                )
            else:
                replacement = f'\n\n**[Изображение {img_num}]**\n\n'
            markdown = markdown[:match.start()] + replacement + markdown[match.end():]

        return markdown

    async def _process_formula_placeholders(
        self,
        markdown: str,
        formulas: list[_PictureInfo],
        pdf_path: Path,
    ) -> str:
        """Заменить <!-- formula-not-decoded --> плейсхолдеры на LaTeX через Vision LLM.

        Формулы вырезаются из PDF по координатам из docling JSON.
        """
        placeholders = list(_FORMULA_PLACEHOLDER_RE.finditer(markdown))
        if not placeholders:
            return markdown

        n = min(len(placeholders), len(formulas))
        if len(placeholders) != len(formulas):
            logger.warning(
                'Formula placeholder count (%d) != formulas count (%d), processing %d',
                len(placeholders), len(formulas), n,
            )

        logger.info('Processing %d formula(s) from PDF via Vision LLM...', n)

        # Вырезаем формулы в thread pool
        loop = asyncio.get_event_loop()
        crop_tasks = [
            loop.run_in_executor(None, self._crop_image, pdf_path, formulas[i])
            for i in range(n)
        ]
        cropped_images = await asyncio.gather(*crop_tasks)

        # Распознаём через Vision LLM
        descriptions: list[str | None] = [None] * n
        for i, img_b64 in enumerate(cropped_images):
            if img_b64 is None:
                continue
            try:
                desc = await self._vision_client.complete_vision(
                    _FORMULA_PROMPT, img_b64, media_type='image/png',
                )
                descriptions[i] = desc
            except Exception as exc:
                logger.warning('Failed to recognize formula %d: %s', i + 1, exc)

        # Заменяем плейсхолдеры в обратном порядке
        for i in range(n - 1, -1, -1):
            match = placeholders[i]
            if descriptions[i]:
                replacement = f'\n\n{descriptions[i]}\n\n'
            else:
                replacement = ''
            markdown = markdown[:match.start()] + replacement + markdown[match.end():]

        # Удалить оставшиеся необработанные плейсхолдеры
        markdown = _FORMULA_PLACEHOLDER_RE.sub('', markdown)

        return markdown
