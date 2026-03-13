from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from markdown_it import MarkdownIt

from morag.indexing.token_counter import TokenCounter

logger = logging.getLogger(__name__)


# CommonMark-парсер с поддержкой GFM-таблиц.
# Инстанс stateless (parse создаёт свежий State) — безопасно переиспользовать.
_md_parser = MarkdownIt().enable('table')

# Строка вида ```...] в начале строки — часть конструкции [Изображение: ```plantuml...```].
# CommonMark трактует как fence (info string ']'). Добавляем backtick в конец →
# info string содержит backtick → fence невалиден по спеке CommonMark §4.5.
_BRACKET_FENCE_RE = re.compile(r'^([ \t]*`{3,}[^`]*\])[ \t]*$', re.MULTILINE)


def _parse_md(text: str) -> list:
    """Парсит Markdown с предобработкой Confluence-специфичных конструкций.

    Предобработка не меняет количество строк — номера строк в token.map
    соответствуют исходному тексту.
    """
    processed = _BRACKET_FENCE_RE.sub(r'\1`', text)
    return _md_parser.parse(processed)


class BlockSplitter(ABC):
    """Интерфейс разделителя блоков текста."""

    @abstractmethod
    def can_split(self, text: str) -> bool:
        """Вернуть True если разделитель применим к данному тексту."""
        ...

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Разбить текст на части. Каждая часть должна быть меньше исходной."""
        ...


class MarkdownHeaderSplitter(BlockSplitter):
    """Разделяет текст по заголовкам Markdown (# ## ###...).

    Использует CommonMark-парсер: заголовки внутри code fences не считаются заголовками.
    """

    def can_split(self, text: str) -> bool:
        tokens = _parse_md(text)
        heading_count = sum(1 for t in tokens if t.type == 'heading_open')
        return heading_count > 1

    def split(self, text: str) -> list[str]:
        sections = _split_by_headers(text)
        return [s.strip() for s in sections if s.strip()]


class TableRowSplitter(BlockSplitter):
    """Разделяет таблицы Markdown по строкам с дублированием шапки.

    Набирает строки жадно, пока помещаются в лимит токенов (шапка + строки данных).
    """

    _TABLE_ROW_RE = re.compile(r'^\s*\|')
    _SEPARATOR_RE = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

    def __init__(self, counter: TokenCounter, limit: int) -> None:
        self._counter = counter
        self._limit = limit

    def can_split(self, text: str) -> bool:
        """Применим если таблица не влезает в лимит целиком и имеет > 1 строки данных."""
        if self._counter.fits(text, self._limit):
            return False
        return len(self._extract_data_rows(text)) > 1

    def split(self, text: str) -> list[str]:
        lines = text.split('\n')

        pre_lines: list[str] = []
        table_header: list[str] = []
        separator: str | None = None
        data_rows: list[str] = []
        post_lines: list[str] = []
        phase = 'pre'

        for line in lines:
            if phase == 'pre':
                if self._TABLE_ROW_RE.match(line):
                    phase = 'header'
                    table_header.append(line)
                else:
                    pre_lines.append(line)
            elif phase == 'header':
                if self._SEPARATOR_RE.match(line):
                    separator = line
                    phase = 'data'
                elif self._TABLE_ROW_RE.match(line):
                    table_header.append(line)
                else:
                    phase = 'post'
                    post_lines.append(line)
            elif phase == 'data':
                if self._TABLE_ROW_RE.match(line):
                    data_rows.append(line)
                else:
                    phase = 'post'
                    post_lines.append(line)
            else:
                post_lines.append(line)

        if not data_rows:
            return [text]

        header_parts = table_header + ([separator] if separator else [])
        header_text = '\n'.join(header_parts)
        header_tokens = self._counter.count(header_text)

        chunks: list[str] = []
        current_rows: list[str] = []
        current_tokens = header_tokens

        for idx, row in enumerate(data_rows):
            row_tokens = self._counter.count(row)
            if current_rows and current_tokens + row_tokens > self._limit:
                parts: list[str] = []
                if not chunks and pre_lines:
                    parts.extend(pre_lines)
                parts.append(header_text)
                parts.extend(current_rows)
                chunks.append('\n'.join(parts).strip())
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        if current_rows:
            parts = []
            if not chunks and pre_lines:
                parts.extend(pre_lines)
            parts.append(header_text)
            parts.extend(current_rows)
            if post_lines:
                parts.extend(post_lines)
            chunks.append('\n'.join(parts).strip())

        return chunks if len(chunks) > 1 else [text]

    def _extract_data_rows(self, text: str) -> list[str]:
        lines = text.split('\n')
        separator_found = False
        data_rows: list[str] = []

        for line in lines:
            if not separator_found:
                if self._SEPARATOR_RE.match(line):
                    separator_found = True
            else:
                if self._TABLE_ROW_RE.match(line):
                    data_rows.append(line)
                else:
                    break

        return data_rows


def split_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения.

    Использует razdel для кириллического текста и nltk для латиницы.
    """
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    latin = sum(1 for c in text if 'A' <= c <= 'z')
    if cyrillic >= latin:
        from razdel import sentenize
        return [s.text.strip() for s in sentenize(text) if s.text.strip()]
    else:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _top_level_blocks(text: str) -> list[tuple[str, int, int]]:
    """Извлекает top-level блоки из Markdown-текста через CommonMark-парсер.

    Возвращает список (token_type, start_line, end_line) для каждого
    top-level блок-элемента. Строки 0-based, end_line exclusive.
    """
    tokens = _parse_md(text)
    blocks: list[tuple[str, int, int]] = []
    level = 0
    for token in tokens:
        if token.nesting == 1:
            if level == 0 and token.map:
                blocks.append((token.type, token.map[0], token.map[1]))
            level += 1
        elif token.nesting == -1:
            level -= 1
        elif token.nesting == 0 and level == 0 and token.map:
            blocks.append((token.type, token.map[0], token.map[1]))
    return blocks


def split_into_units(text: str) -> list[str]:
    """Разбивает текст на атомарные единицы для семантического чанкинга.

    Атомарные блоки (не разрезаются):
    - Code fences (``` ... ```) — включая plantuml, mermaid, ASCII-диаграммы
    - Таблицы Markdown (последовательные строки с |)

    Всё остальное разбивается на предложения через split_sentences().
    """
    source_lines = text.split('\n')
    blocks = _top_level_blocks(text)
    units: list[str] = []

    for block_type, start, end in blocks:
        block_text = '\n'.join(source_lines[start:end]).strip()
        if not block_text:
            continue
        if block_type in ('fence', 'code_block', 'table_open'):
            units.append(block_text)
        else:
            sentences = split_sentences(block_text)
            units.extend(sentences)

    return [u for u in units if u.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Разбивает текст на блоки по структуре Markdown.

    Использует CommonMark-парсер для корректного определения блоков:
    code fences, таблицы, заголовки, параграфы, списки и т.д.
    """
    source_lines = text.split('\n')
    blocks = _top_level_blocks(text)

    if not blocks:
        stripped = text.strip()
        return [stripped] if stripped else []

    paragraphs: list[str] = []
    for _, start, end in blocks:
        block_text = '\n'.join(source_lines[start:end]).strip()
        if block_text:
            paragraphs.append(block_text)

    return paragraphs


def split_into_semantic_units(
    text: str,
    counter: TokenCounter,
    max_tokens: int,
) -> list[str]:
    """Иерархическая нарезка текста на единицы для SemanticChunker.

    1. По заголовкам Markdown → секции (заголовок + контент до следующего)
    2. По абзацам (двойной перенос строки; code fences и таблицы — цельные абзацы)
    3. Для каждого абзаца > max_tokens:
       а) таблица     → разрезка по строкам с дублированием шапки (≤ max_tokens)
       б) code fence  → оставить как есть + WARNING
       в) текст       → разбить на предложения (split_sentences)
    """
    sections = _split_by_headers(text)
    table_splitter = TableRowSplitter(counter, max_tokens)

    units: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if counter.count(section) <= max_tokens:
            units.append(section)
            continue

        paragraphs = _split_paragraphs(section)
        for para in paragraphs:
            if counter.count(para) <= max_tokens:
                units.append(para)
            elif table_splitter.can_split(para):
                units.extend(table_splitter.split(para))
            elif _is_code_fence(para):
                logger.warning(
                    'Code fence exceeds max_tokens (%d > %d), keeping as-is',
                    counter.count(para), max_tokens,
                )
                units.append(para)
            else:
                sentences = split_sentences(para)
                units.extend(s for s in sentences if s.strip())

    return [u for u in units if u.strip()]


def _is_code_fence(text: str) -> bool:
    """Проверяет, является ли текст code fence блоком (``` ... ```)."""
    return text.lstrip().startswith('```')


def _split_by_headers(text: str) -> list[str]:
    """Разбивает текст по заголовкам Markdown. Каждый заголовок начинает новую секцию.

    Использует CommonMark-парсер: заголовки внутри code fences корректно игнорируются.
    """
    tokens = _parse_md(text)
    lines = text.split('\n')

    heading_starts = [t.map[0] for t in tokens if t.type == 'heading_open' and t.map]

    if not heading_starts:
        return [text]

    # Первый заголовок на первой строке не создаёт разбиение
    # (совместимость с прежним поведением: split только если есть контент до заголовка)
    if heading_starts[0] == 0:
        split_points = heading_starts[1:]
    else:
        split_points = heading_starts

    if not split_points:
        return [text]

    sections: list[str] = []
    prev = 0
    for sp in split_points:
        sections.append('\n'.join(lines[prev:sp]))
        prev = sp

    sections.append('\n'.join(lines[prev:]))
    return sections


class SemanticSplitter(BlockSplitter):
    """Разделяет текст по семантическим границам через эмбеддинги.

    Находит точки разрыва там где косинусное расстояние между соседними
    предложениями максимально — то есть где тема меняется сильнее всего.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        breakpoint_percentile: int = 95,
        min_sentences: int = 3,
    ) -> None:
        self._embed_fn = embed_fn
        self._breakpoint_percentile = breakpoint_percentile
        self._min_sentences = min_sentences

    def can_split(self, text: str) -> bool:
        return len(split_sentences(text)) >= self._min_sentences

    def split(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) < self._min_sentences:
            return [text]

        embeddings = [self._embed_fn(s) for s in sentences]
        distances = self._cosine_distances(embeddings)

        threshold = float(np.percentile(distances, self._breakpoint_percentile))
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        if not breakpoints:
            return [text]

        return self._join_by_breakpoints(sentences, breakpoints)

    @staticmethod
    def _cosine_distances(embeddings: list[list[float]]) -> list[float]:
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i], dtype=np.float32)
            b = np.array(embeddings[i + 1], dtype=np.float32)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            similarity = float(np.dot(a, b) / (norm + 1e-8))
            distances.append(1.0 - similarity)
        return distances

    @staticmethod
    def _join_by_breakpoints(sentences: list[str], breakpoints: list[int]) -> list[str]:
        chunks: list[str] = []
        prev = 0
        for bp in breakpoints:
            chunk = ' '.join(sentences[prev:bp])
            if chunk:
                chunks.append(chunk)
            prev = bp
        tail = ' '.join(sentences[prev:])
        if tail:
            chunks.append(tail)
        return chunks if chunks else [' '.join(sentences)]



class FixedSizeSplitter(BlockSplitter):
    """Последний резерв: разбивает по абзацам, предложениям, словам и символам."""

    _TABLE_ROW_RE = re.compile(r'^\s*\|')
    _TABLE_SEP_RE = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

    def __init__(self, counter: TokenCounter, limit: int) -> None:
        self._counter = counter
        self._limit = limit

    def can_split(self, text: str) -> bool:
        return bool(text.strip())

    def split(self, text: str) -> list[str]:
        paragraphs = _split_paragraphs(text)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._counter.count(para)

            if para_tokens > self._limit:
                if current_parts:
                    chunks.append('\n\n'.join(current_parts))
                    current_parts = []
                    current_tokens = 0
                chunks.extend(self._split_oversized(para))
            elif current_tokens + para_tokens > self._limit and current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = [para]
                current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunks.append('\n\n'.join(current_parts))

        return chunks if chunks else [text]

    @classmethod
    def _is_table(cls, text: str) -> bool:
        """Проверяет, является ли текст Markdown-таблицей с заголовком и разделителем."""
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False
        return (
            bool(cls._TABLE_ROW_RE.match(lines[0]))
            and any(cls._TABLE_SEP_RE.match(line) for line in lines[:3])
        )

    def _split_table_rows(self, text: str) -> list[str]:
        """Разбивает Markdown-таблицу на чанки по строкам с дублированием шапки.

        Возвращает [text] если таблица содержит только одну строку данных и не может
        быть разбита дальше (сигнал для RecursiveSplitter о том, что прогресса нет).
        """
        lines = text.split('\n')
        header_lines: list[str] = []
        separator: str | None = None
        data_rows: list[str] = []
        phase = 'header'

        for line in lines:
            if phase == 'header':
                if self._TABLE_SEP_RE.match(line):
                    separator = line
                    phase = 'data'
                elif self._TABLE_ROW_RE.match(line):
                    header_lines.append(line)
            elif phase == 'data':
                if self._TABLE_ROW_RE.match(line):
                    data_rows.append(line)
                else:
                    break

        if not data_rows:
            return [text]

        header_parts = header_lines + ([separator] if separator else [])
        header_text = '\n'.join(header_parts)
        header_tokens = self._counter.count(header_text)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for row in data_rows:
            row_tokens = self._counter.count(row)
            if current and current_tokens + row_tokens + header_tokens > self._limit:
                chunks.append(header_text + '\n' + '\n'.join(current))
                current = [row]
                current_tokens = row_tokens
            else:
                current.append(row)
                current_tokens += row_tokens

        if current:
            chunks.append(header_text + '\n' + '\n'.join(current))

        return chunks if len(chunks) > 1 else [text]

    def _split_oversized(self, text: str) -> list[str]:
        """Последовательно пробует: таблица → предложения → слова → символы."""
        if self._is_table(text):
            result = self._split_table_rows(text)
            if len(result) > 1:
                return result
        sentences = split_sentences(text)
        if len(sentences) > 1:
            return self._pack_and_recurse(sentences, ' ', self._split_by_words)
        return self._split_by_words(text)

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        if len(words) > 1:
            return self._pack_and_recurse(words, ' ', self._split_by_chars)
        return self._split_by_chars(text)

    def _split_by_chars(self, text: str) -> list[str]:
        chars_per_chunk = max(1, self._limit * 4)
        return [text[i : i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]

    def _pack_and_recurse(
        self, units: list[str], sep: str, fallback: Callable[[str], list[str]]
    ) -> list[str]:
        """Жадно упаковывает единицы в чанки; всё что не влезает отдаёт в fallback."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._counter.count(unit)
            if unit_tokens > self._limit:
                if current:
                    chunks.append(sep.join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(fallback(unit))
            elif current_tokens + unit_tokens > self._limit and current:
                chunks.append(sep.join(current))
                current = [unit]
                current_tokens = unit_tokens
            else:
                current.append(unit)
                current_tokens += unit_tokens

        if current:
            chunks.append(sep.join(current))

        return chunks or [sep.join(units)]


class RecursiveSplitter:
    """Рекурсивно применяет цепочку сплиттеров до тех пор, пока блоки не влезут в лимит.

    Порядок сплиттеров определяет приоритет стратегий. FixedSizeSplitter
    рекомендуется ставить последним — он гарантирует завершение рекурсии.
    """

    def __init__(
        self,
        counter: TokenCounter,
        limit: int,
        splitters: list[BlockSplitter],
    ) -> None:
        self._counter = counter
        self._limit = limit
        self._splitters = splitters

    def split(self, text: str) -> list[str]:
        return self._recurse(text)

    def _recurse(self, text: str) -> list[str]:
        if self._counter.fits(text, self._limit):
            return [text]

        for splitter in self._splitters:
            if not splitter.can_split(text):
                continue

            parts = splitter.split(text)
            if not parts or (len(parts) == 1 and parts[0] == text):
                continue  # сплиттер не дал прогресса

            result: list[str] = []
            for part in parts:
                result.extend(self._recurse(part))
            return result

        return [text]  # ни один сплиттер не помог


def pack_blocks(blocks: list[str], counter: TokenCounter, limit: int) -> list[list[str]]:
    """Жадная упаковка блоков в пачки до заполнения лимита токенов.

    Каждый вызов LLM получает максимально возможный контекст.
    """
    if not blocks:
        return []

    packs: list[list[str]] = []
    current_pack: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = counter.count(block)
        if current_tokens + block_tokens > limit and current_pack:
            packs.append(current_pack)
            current_pack = [block]
            current_tokens = block_tokens
        else:
            current_pack.append(block)
            current_tokens += block_tokens

    if current_pack:
        packs.append(current_pack)

    return packs
