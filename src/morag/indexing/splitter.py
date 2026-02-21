from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from morag.indexing.token_counter import TokenCounter


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
    """Разделяет текст по заголовкам Markdown (# ## ###...)."""

    _HEADER_RE = re.compile(r'^#{1,6}\s', re.MULTILINE)

    def can_split(self, text: str) -> bool:
        return len(self._HEADER_RE.findall(text)) > 1

    def split(self, text: str) -> list[str]:
        lines = text.split('\n')
        blocks: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            if self._HEADER_RE.match(line) and current:
                blocks.append('\n'.join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            blocks.append('\n'.join(current).strip())

        return [b for b in blocks if b]


class TableRowSplitter(BlockSplitter):
    """Разделяет таблицы Markdown по N строк с дублированием шапки."""

    _TABLE_ROW_RE = re.compile(r'^\s*\|')
    _SEPARATOR_RE = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

    def __init__(self, rows_per_chunk: int = 20) -> None:
        self.rows_per_chunk = rows_per_chunk

    def can_split(self, text: str) -> bool:
        return len(self._extract_data_rows(text)) > self.rows_per_chunk

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

        chunks: list[str] = []
        total = len(data_rows)

        for i in range(0, total, self.rows_per_chunk):
            batch = data_rows[i : i + self.rows_per_chunk]
            parts: list[str] = []

            if i == 0 and pre_lines:
                parts.extend(pre_lines)

            parts.extend(table_header)
            if separator:
                parts.append(separator)
            parts.extend(batch)

            if i + self.rows_per_chunk >= total and post_lines:
                parts.extend(post_lines)

            chunk = '\n'.join(parts).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

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
        return len(self._split_sentences(text)) >= self._min_sentences

    def split(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
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
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

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

    def __init__(self, counter: TokenCounter, limit: int) -> None:
        self._counter = counter
        self._limit = limit

    def can_split(self, text: str) -> bool:
        return bool(text.strip())

    def split(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

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

    def _split_oversized(self, text: str) -> list[str]:
        """Последовательно пробует: предложения → слова → символы."""
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
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
