"""Постпроцессоры для Vision PDF конвертации.

Подключаемая система: каждый постпроцессор реализует PdfPostProcessor
и применяется последовательно к результату конвертации.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PdfPostProcessor(ABC):
    """Базовый класс постпроцессора PDF-конвертации."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Обработать текст и вернуть результат."""
        ...


class CodeFencePostProcessor(PdfPostProcessor):
    """Удаление orphan code fences из текста.

    Vision LLM иногда оборачивает ответ в ```markdown ... ``` или оставляет
    незакрытые code fences. Этот постпроцессор удаляет такие артефакты.
    """

    # Полная обёртка: ```markdown\n...\n```
    _WRAP_RE = re.compile(
        r'```(?:markdown|md|)\s*\n(.*?)\n\s*```',
        re.DOTALL,
    )
    # Orphan открывающий fence без закрывающего (до конца текста или до следующего fence)
    _ORPHAN_OPEN_RE = re.compile(r'^```(?:markdown|md|)\s*$', re.MULTILINE)

    def process(self, text: str) -> str:
        before_len = len(text)
        # Шаг 1: развернуть полные обёртки ```markdown ... ```
        text = self._WRAP_RE.sub(r'\1', text)
        # Шаг 2: удалить orphan открывающие fences
        text = self._ORPHAN_OPEN_RE.sub('', text)
        # Шаг 3: удалить одиночные закрывающие ``` на отдельной строке (без открывающего)
        text = re.sub(r'^\s*```\s*$', '', text, flags=re.MULTILINE)
        # Чистим пустые строки, оставшиеся после удаления
        text = re.sub(r'\n{3,}', '\n\n', text)
        after_len = len(text)
        if before_len != after_len:
            logger.info(
                'CodeFencePostProcessor: %d → %d chars (removed %d)',
                before_len, after_len, before_len - after_len,
            )
        return text


class DeduplicatePostProcessor(PdfPostProcessor):
    """Удаление дублирующихся абзацев и зацикленных фраз внутри абзацев.

    Два этапа:
    1. Межабзацная дедупликация: fuzzy-сравнение каждого абзаца
       с предыдущими N абзацами (скользящее окно).
    2. Внутриабзацная дедупликация: поиск повторяющихся фраз
       внутри одного абзаца (зацикливание модели).
    """

    def __init__(
        self,
        threshold: float = 0.7,
        window: int = 5,
        min_phrase_len: int = 20,
    ) -> None:
        """
        Args:
            threshold: порог fuzzy-сходства для межабзацной дедупликации (0..1).
            window: количество предыдущих абзацев для сравнения.
            min_phrase_len: мин. длина повторяющейся фразы для внутриабзацной дедупликации.
        """
        self._threshold = threshold
        self._window = window
        self._min_phrase_len = min_phrase_len

    def process(self, text: str) -> str:
        before_len = len(text)
        text = self._dedup_paragraphs(text)
        text = self._dedup_intra_paragraph(text)
        after_len = len(text)
        if before_len != after_len:
            removed_pct = (1 - after_len / before_len) * 100
            logger.info(
                'DeduplicatePostProcessor: %d → %d chars (removed %.1f%%)',
                before_len, after_len, removed_pct,
            )
        return text

    def _dedup_paragraphs(self, text: str) -> str:
        """Удалить дублирующиеся абзацы (fuzzy-сравнение со скользящим окном)."""
        paragraphs = re.split(r'\n{2,}', text)
        result: list[str] = []

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue

            # Короткие абзацы (заголовки, разделители) не дедуплицируем
            if len(stripped) < self._min_phrase_len:
                result.append(para)
                continue

            # Сравниваем с последними N абзацами
            is_dup = False
            window_start = max(0, len(result) - self._window)
            for prev in result[window_start:]:
                prev_stripped = prev.strip()
                if len(prev_stripped) < self._min_phrase_len:
                    continue
                ratio = SequenceMatcher(None, stripped, prev_stripped).ratio()
                if ratio >= self._threshold:
                    is_dup = True
                    break

            if not is_dup:
                result.append(para)

        return '\n\n'.join(result)

    def _dedup_intra_paragraph(self, text: str) -> str:
        """Удалить повторяющиеся фразы внутри абзацев (зацикливание модели).

        Ищет подстроку длиной >= min_phrase_len, которая встречается 2+ раз подряд
        (возможно с небольшими вариациями), и оставляет одно вхождение.
        """
        paragraphs = text.split('\n\n')
        result: list[str] = []

        for para in paragraphs:
            if len(para) < self._min_phrase_len * 2:
                result.append(para)
                continue
            result.append(self._remove_repeated_phrases(para))

        return '\n\n'.join(result)

    def _remove_repeated_phrases(self, text: str) -> str:
        """Убрать повторяющиеся фразы из текста (один абзац).

        Два этапа:
        1. Sentence-level: дедупликация предложений по первым 30 символам + fuzzy.
        2. Loop detection: поиск повторяющихся подстрок скользящим окном.
        """
        min_len = self._min_phrase_len
        text = self._dedup_sentences(text, min_len)
        text = self._remove_loops(text, min_len)
        return text

    def _dedup_sentences(self, text: str, min_len: int) -> str:
        """Удалить повторяющиеся предложения (fuzzy-сравнение)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text

        seen: dict[str, int] = {}
        dedup_indices: set[int] = set()

        for i, sent in enumerate(sentences):
            normalized = sent.strip().lower()
            if len(normalized) < min_len:
                continue
            key = normalized[:30]
            if key in seen:
                prev_idx = seen[key]
                prev_sent = sentences[prev_idx].strip()
                ratio = SequenceMatcher(None, sent.strip(), prev_sent).ratio()
                if ratio >= self._threshold:
                    dedup_indices.add(i)
                    continue
            seen[key] = i

        if not dedup_indices:
            return text

        kept = [s for i, s in enumerate(sentences) if i not in dedup_indices]
        return ' '.join(kept)

    def _remove_loops(self, text: str, min_len: int) -> str:
        """Удалить зацикливания — повторяющиеся подстроки внутри текста.

        Ищет подстроку длиной >= min_len, которая уже встречалась ранее
        в тексте (в пределах окна поиска), и удаляет повторное вхождение.
        """
        text_lower = text.lower()
        text_len = len(text)
        if text_len < min_len * 2:
            return text

        search_back = max(min_len * 10, 500)
        remove_ranges: list[tuple[int, int]] = []

        i = min_len
        while i <= text_len - min_len:
            fragment = text_lower[i:i + min_len]
            search_start = max(0, i - search_back)
            pos = text_lower.find(fragment, search_start, i)
            if pos == -1:
                i += 1
                continue

            # Расширяем совпадение до полной длины повтора
            match_len = min_len
            while (i + match_len < text_len
                   and pos + match_len < i
                   and text_lower[pos + match_len] == text_lower[i + match_len]):
                match_len += 1

            # Фильтр: повтор должен быть рядом с оригиналом (gap < match * 3)
            gap = i - (pos + match_len)
            if match_len >= min_len and gap < match_len * 3:
                remove_ranges.append((i, i + match_len))
                i += match_len
            else:
                i += 1

        if not remove_ranges:
            return text

        parts: list[str] = []
        prev_end = 0
        for start, end in remove_ranges:
            if start > prev_end:
                parts.append(text[prev_end:start])
            prev_end = max(prev_end, end)
        parts.append(text[prev_end:])

        result = ''.join(parts)
        result = re.sub(r' {2,}', ' ', result)
        return result
