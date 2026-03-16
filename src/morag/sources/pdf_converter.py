"""Интерфейс и реализации конвертации PDF → Markdown.

PdfConverter — абстрактный интерфейс для конвертации PDF-файлов в Markdown.
DoclingPdfConverter — реализация через docling-serve + Vision LLM для изображений и формул.
VisionPdfConverter — реализация только через Vision LLM (постраничный рендеринг).
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

from morag.llm.client import GenerationParams, LLMClient
from morag.sources.pdf_postprocess import PdfPostProcessor

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
                    params=GenerationParams(seed=42),
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
                    params=GenerationParams(seed=42),
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


# Регулярное выражение для удаления обёрток ```markdown ... ```
_CODE_FENCE_WRAP_RE = re.compile(
    r'^```(?:markdown)?\s*\n(.*?)```\s*$',
    re.DOTALL,
)

_VISION_PAGE_PROMPT = (
    'Перед тобой скан страницы документа. Преобразуй её содержимое в Markdown.\n\n'
    'Правила:\n'
    '- Сохраняй структуру: заголовки (##), списки, абзацы.\n'
    '- Таблицы оформляй как markdown-таблицы (| ... | ... |).\n'
    '- Формулы записывай в LaTeX: инлайн $...$ или блочные $$...$$.\n'
    '- Диаграммы и графики: опиши текстом в блоке вида\n'
    '  **[Диаграмма]** *Описание: ...*\n'
    '  Укажи тип диаграммы, оси, основные данные и тренды.\n'
    '- Изображения (фото, скриншоты, иллюстрации): опиши кратко в блоке\n'
    '  **[Изображение]** *Описание: ...*\n'
    '- Не добавляй информацию, которой нет на странице.\n'
    '- Не добавляй колонтитулы и номера страниц.\n'
    '- НЕ оборачивай ответ в ```markdown``` или другие code fences.\n'
    '- Отвечай ТОЛЬКО markdown-содержимым страницы, без вступлений и пояснений.'
)

_VISION_PAGE_PROMPT_WITH_CONTEXT = (
    'Перед тобой скан страницы документа. Преобразуй её содержимое в Markdown.\n\n'
    'Для справки — конец предыдущей страницы:\n'
    '---\n'
    '{prev_tail}\n'
    '---\n\n'
    'Правила:\n'
    '- Преобразуй ВСЁ содержимое текущей страницы. Не пропускай текст, '
    'даже если он частично совпадает с концом предыдущей страницы.\n'
    '- Если предложение начато на предыдущей странице и продолжается на текущей, '
    'начни с продолжения этого предложения.\n'
    '- Сохраняй структуру: заголовки (##), списки, абзацы.\n'
    '- Таблицы оформляй как markdown-таблицы (| ... | ... |).\n'
    '- Формулы записывай в LaTeX: инлайн $...$ или блочные $$...$$.\n'
    '- Диаграммы и графики: опиши текстом в блоке вида\n'
    '  **[Диаграмма]** *Описание: ...*\n'
    '  Укажи тип диаграммы, оси, основные данные и тренды.\n'
    '- Изображения (фото, скриншоты, иллюстрации): опиши кратко в блоке\n'
    '  **[Изображение]** *Описание: ...*\n'
    '- Не добавляй информацию, которой нет на странице.\n'
    '- Не добавляй колонтитулы и номера страниц.\n'
    '- НЕ оборачивай ответ в ```markdown``` или другие code fences.\n'
    '- Отвечай ТОЛЬКО markdown-содержимым страницы, без вступлений и пояснений.'
)

_CONTEXT_TAIL_LINES = 15  # количество строк хвоста предыдущей страницы


def _strip_code_fences(text: str) -> str:
    """Убрать обёртку ```markdown ... ``` из ответа LLM."""
    stripped = text.strip()
    m = _CODE_FENCE_WRAP_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


class VisionPdfConverter(PdfConverter):
    """Конвертация PDF через Vision LLM: постраничный рендеринг → markdown."""

    def __init__(
        self,
        vision_client: LLMClient,
        max_tokens: int = 4096,
        dpi: int = 144,
        concurrency: int = 1,
        generation_params: GenerationParams | None = None,
        context_tail_lines: int = 0,
        postprocessors: list[PdfPostProcessor] | None = None,
    ) -> None:
        self._vision_client = vision_client
        self._max_tokens = max_tokens
        self._zoom = dpi / 72.0  # PDF default is 72 DPI
        self._semaphore = asyncio.Semaphore(concurrency)
        self._gen_params = generation_params
        self._context_tail_lines = context_tail_lines
        self._postprocessors = postprocessors or []

    async def convert(self, pdf_bytes: bytes, filename: str) -> str | None:
        """Конвертировать PDF постранично через Vision LLM."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        except Exception as exc:
            logger.error('Failed to open PDF %s: %s', filename, exc)
            return None

        page_count = len(doc)
        logger.info(
            'Processing %d page(s) of %s via Vision LLM (concurrency=%d, context_tail=%d)...',
            page_count, filename, self._semaphore._value, self._context_tail_lines,
        )

        loop = asyncio.get_event_loop()
        mat = fitz.Matrix(self._zoom, self._zoom)

        # Рендерим все страницы параллельно в executor
        render_tasks = [
            loop.run_in_executor(None, self._render_page, doc, i, mat)
            for i in range(page_count)
        ]
        page_images = await asyncio.gather(*render_tasks)
        doc.close()

        use_sliding = self._context_tail_lines > 0

        if use_sliding:
            # Последовательная обработка с передачей хвоста предыдущей страницы
            results: list[str | None] = []
            prev_tail: str | None = None
            for idx, img_b64 in enumerate(page_images):
                md = await self._process_page(
                    idx, img_b64, page_count, filename, prev_tail,
                )
                results.append(md)
                if md:
                    lines = md.splitlines()
                    prev_tail = '\n'.join(lines[-self._context_tail_lines:])
                else:
                    prev_tail = None
        else:
            # Параллельная обработка (без контекста)
            results = list(await asyncio.gather(*[
                self._process_page(i, img, page_count, filename, None)
                for i, img in enumerate(page_images)
            ]))

        # Проверяем что все страницы обработаны успешно
        failed_pages = [i + 1 for i, md in enumerate(results) if not md]
        if failed_pages:
            logger.error(
                'Vision LLM failed for %d page(s) of %s: %s',
                len(failed_pages), filename, failed_pages,
            )
            return None

        page_markdowns = [
            f'<!-- page:{i + 1} -->\n{md}'
            for i, md in enumerate(results)
        ]

        result = '\n\n'.join(page_markdowns)

        for pp in self._postprocessors:
            result = pp.process(result)

        return result

    async def _process_page(
        self,
        idx: int,
        img_b64: str | None,
        page_count: int,
        filename: str,
        prev_tail: str | None,
    ) -> str | None:
        """Обработать одну страницу через Vision LLM."""
        if img_b64 is None:
            logger.warning('Skipping page %d of %s: render failed', idx + 1, filename)
            return None
        if prev_tail:
            prompt = _VISION_PAGE_PROMPT_WITH_CONTEXT.format(prev_tail=prev_tail)
        else:
            prompt = _VISION_PAGE_PROMPT
        async with self._semaphore:
            try:
                raw = await self._vision_client.complete_vision(
                    prompt, img_b64,
                    media_type='image/png',
                    max_tokens=self._max_tokens,
                    params=self._gen_params,
                )
                md = _strip_code_fences(raw) if raw else None
                logger.info('Page %d/%d of %s done', idx + 1, page_count, filename)
                return md
            except Exception as exc:
                logger.warning(
                    'Vision LLM failed for page %d of %s: %s', idx + 1, filename, exc,
                )
                return None

    @staticmethod
    def _render_page(doc: fitz.Document, page_idx: int, mat: fitz.Matrix) -> str | None:
        """Отрендерить страницу PDF в base64 PNG."""
        try:
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes('png')
            return base64.b64encode(img_bytes).decode()
        except Exception as exc:
            logger.warning('Failed to render page %d: %s', page_idx + 1, exc)
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
