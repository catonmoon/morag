"""Интерфейс и реализации конвертации PDF → Markdown.

PdfConverter — абстрактный интерфейс для конвертации PDF-файлов в Markdown.
DoclingPdfConverter — реализация через docling-serve + Vision LLM для изображений и формул.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz
import httpx

from morag.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Регулярное выражение для плейсхолдеров изображений
_IMAGE_PLACEHOLDER_RE = re.compile(r'<!--\s*image\s*-->')

# Регулярное выражение для плейсхолдеров формул
_FORMULA_PLACEHOLDER_RE = re.compile(r'<!--\s*formula-not-decoded\s*-->')

# Промпт для описания изображений
_IMAGE_PROMPT = (
    'Опиши подробно, что изображено на изображении. Если что-то не видно (низкая вероятность '
    'распознавания), такие данные не приводи.'
)

# Промпт для распознавания формул
_FORMULA_PROMPT = (
    'Обрати внимание на формулы, переведи формулу в markdown, т.е. начни и закончи с $$'
)

# Zoom для рендеринга изображений из PDF (2x = 144 DPI)
_IMAGE_ZOOM = 2.0


@dataclass
class PictureInfo:
    """Метаданные картинки/формулы из docling JSON."""

    page_no: int
    bbox_l: float
    bbox_t: float
    bbox_r: float
    bbox_b: float
    coord_origin: str


class PdfConverter(ABC):
    """Абстрактный интерфейс конвертации PDF → Markdown."""

    @abstractmethod
    async def convert(self, pdf_bytes: bytes, filename: str) -> str | None:
        """Конвертировать PDF в Markdown.

        Возвращает markdown-текст или None при ошибке.
        """
        ...


class DoclingPdfConverter(PdfConverter):
    """Конвертация PDF через docling-serve + Vision LLM для изображений и формул."""

    def __init__(
        self,
        docling_base_url: str,
        docling_timeout: int = 300,
        vision_client: LLMClient | None = None,
        vision_max_tokens: int | None = None,
    ) -> None:
        self._docling_base_url = docling_base_url.rstrip('/')
        self._docling_timeout = docling_timeout
        self._vision_client = vision_client
        self._vision_max_tokens = vision_max_tokens

    async def convert(self, pdf_bytes: bytes, filename: str) -> str | None:
        """Конвертировать PDF в Markdown через docling-serve."""
        result = await self._convert_via_docling(pdf_bytes, filename)
        if result is None:
            return None

        markdown, pictures, formulas = result

        # Для обработки изображений и формул нужен временный файл (PyMuPDF работает с Path)
        if self._vision_client is not None and (pictures or formulas):
            with NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
                tmp.write(pdf_bytes)
                tmp.flush()
                pdf_path = Path(tmp.name)

                if pictures:
                    markdown = await self._process_image_placeholders(
                        markdown, pictures, pdf_path,
                    )
                if formulas:
                    markdown = await self._process_formula_placeholders(
                        markdown, formulas, pdf_path,
                    )

        if not self._vision_client or not formulas:
            # Без vision_client или без формул — удаляем плейсхолдеры
            markdown = _FORMULA_PLACEHOLDER_RE.sub('', markdown)

        return markdown

    async def _convert_via_docling(
        self, pdf_bytes: bytes, filename: str,
    ) -> tuple[str, list[PictureInfo], list[PictureInfo]] | None:
        """Отправить PDF в docling-serve и получить markdown + метаданные."""
        fn = partial(self._convert_sync, pdf_bytes, filename)
        return await asyncio.get_event_loop().run_in_executor(None, fn)

    def _convert_sync(
        self, pdf_bytes: bytes, filename: str,
    ) -> tuple[str, list[PictureInfo], list[PictureInfo]] | None:
        """Sync-вызов docling-serve."""
        url = f'{self._docling_base_url}/v1/convert/source'

        b64 = base64.b64encode(pdf_bytes).decode()

        to_formats = ['md']
        if self._vision_client is not None:
            to_formats.append('json')

        payload = {
            'sources': [{'kind': 'file', 'base64_string': b64, 'filename': filename}],
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
                return str(data), [], []

            doc = data.get('document', {})
            md = doc.get('md_content') or data.get('md_content') or data.get('content')
            if not md:
                logger.warning(
                    'Unexpected docling-serve response format for %s: keys=%s',
                    filename, list(data.keys()),
                )
                return None

            json_content = doc.get('json_content')
            pictures = _parse_elements(json_content, 'pictures', filename)
            formulas = _parse_elements(json_content, 'formulas', filename)

            return md, pictures, formulas
        except httpx.HTTPStatusError as exc:
            logger.error(
                'docling-serve HTTP error for %s: %s %s',
                filename, exc.response.status_code, exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error('docling-serve request failed for %s: %s', filename, exc)
            return None

    async def _process_image_placeholders(
        self,
        markdown: str,
        pictures: list[PictureInfo],
        pdf_path: Path,
    ) -> str:
        """Заменить <!-- image --> плейсхолдеры описаниями от Vision LLM."""
        placeholders = list(_IMAGE_PLACEHOLDER_RE.finditer(markdown))
        if not placeholders:
            return markdown

        n = min(len(placeholders), len(pictures))
        if len(placeholders) != len(pictures):
            logger.warning(
                'Image placeholder count (%d) != pictures count (%d), processing %d',
                len(placeholders), len(pictures), n,
            )

        logger.info('Processing %d image(s) from PDF via Vision LLM...', n)

        loop = asyncio.get_event_loop()
        crop_tasks = [
            loop.run_in_executor(None, _crop_image, pdf_path, pictures[i])
            for i in range(n)
        ]
        cropped_images = await asyncio.gather(*crop_tasks)

        descriptions: list[str | None] = [None] * n
        for i, img_b64 in enumerate(cropped_images):
            if img_b64 is None:
                continue
            try:
                desc = await self._vision_client.complete_vision(
                    _IMAGE_PROMPT, img_b64, media_type='image/png',
                    max_tokens=self._vision_max_tokens,
                )
                descriptions[i] = desc
            except Exception as exc:
                logger.warning('Failed to describe image %d: %s', i + 1, exc)

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
        formulas: list[PictureInfo],
        pdf_path: Path,
    ) -> str:
        """Заменить <!-- formula-not-decoded --> плейсхолдеры на LaTeX через Vision LLM."""
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

        loop = asyncio.get_event_loop()
        crop_tasks = [
            loop.run_in_executor(None, _crop_image, pdf_path, formulas[i])
            for i in range(n)
        ]
        cropped_images = await asyncio.gather(*crop_tasks)

        descriptions: list[str | None] = [None] * n
        for i, img_b64 in enumerate(cropped_images):
            if img_b64 is None:
                continue
            try:
                desc = await self._vision_client.complete_vision(
                    _FORMULA_PROMPT, img_b64, media_type='image/png',
                    max_tokens=self._vision_max_tokens,
                )
                descriptions[i] = desc
            except Exception as exc:
                logger.warning('Failed to recognize formula %d: %s', i + 1, exc)

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


def _crop_image(pdf_path: Path, pic: PictureInfo) -> str | None:
    """Вырезать изображение из PDF по координатам и вернуть base64 PNG."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[pic.page_no - 1]
        page_height = page.rect.height

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


def _parse_elements(
    json_content: str | dict | None,
    kind: str,
    filename: str,
) -> list[PictureInfo]:
    """Извлечь координаты элементов из docling JSON.

    kind='pictures' — картинки (из jc['pictures']).
    kind='formulas' — формулы (из jc['texts'] где label='formula').
    """
    if json_content is None:
        return []

    jc = _parse_json_content(json_content, filename)
    if jc is None:
        return []

    if kind == 'pictures':
        elements = jc.get('pictures', [])
    elif kind == 'formulas':
        elements = [t for t in jc.get('texts', []) if t.get('label') == 'formula']
    else:
        return []

    result: list[PictureInfo] = []
    for elem in elements:
        provs = elem.get('prov', [])
        if not provs:
            continue
        prov = provs[0]
        bbox = prov.get('bbox', {})
        if not all(k in bbox for k in ('l', 't', 'r', 'b')):
            continue
        result.append(PictureInfo(
            page_no=prov.get('page_no', 0),
            bbox_l=bbox['l'],
            bbox_t=bbox['t'],
            bbox_r=bbox['r'],
            bbox_b=bbox['b'],
            coord_origin=bbox.get('coord_origin', 'BOTTOMLEFT'),
        ))

    return result
