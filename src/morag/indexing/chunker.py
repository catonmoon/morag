from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Ты — интеллектуальный анализатор документации. Твоя задача — разбить переданный текст \
на смысловые чанки, пригодные для последующего семантического поиска.

Каждый чанк должен:
- Содержать логически завершённую мысль: правило, описание функциональности, \
часть алгоритма, пример или другую осмысленную информацию.
- Не дели текст слишком мелко — не должно быть соседних чанков на одну и ту же тему.
- Таблицы копируй как есть, без анализа содержимого.

Не включай в чанки: оглавление, футеры, навигацию, шаблонные и повторяющиеся элементы.

Формат ответа — строго JSON: {"chunks": ["текст чанка 1", "текст чанка 2", ...]}
"""


class Chunker(ABC):
    """Интерфейс разбивки текстового блока на чанки."""

    @abstractmethod
    async def chunk(self, block: str) -> list[str]:
        """Разбить блок на список текстов чанков."""
        ...


class PassthroughChunker(Chunker):
    """Возвращает блок как есть — один блок равен одному чанку."""

    async def chunk(self, block: str) -> list[str]:
        return [block]


class LLMChunker(Chunker):
    """Разбивает блок на семантические чанки через LLM со structured output.

    При каждой неудачной попытке (невалидный JSON, пустой список, неверная структура)
    повторяет запрос до max_retries раз. Если все попытки провалились — возвращает
    блок целиком как один чанк (passthrough-fallback) и логирует ERROR.
    """

    def __init__(self, client, max_retries: int = 3) -> None:
        self._client = client
        self._max_retries = max_retries

    async def chunk(self, block: str) -> list[str]:
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': block},
        ]

        for attempt in range(1, self._max_retries + 1):
            result = await self._try_chunk(messages, attempt)
            if result is not None:
                return result

        logger.error(
            'LLMChunker: all %d attempts failed for block (%d chars), falling back to passthrough',
            self._max_retries, len(block),
        )
        return [block]

    async def _try_chunk(self, messages: list[dict], attempt: int) -> list[str] | None:
        """Одна попытка чанкинга. Возвращает список чанков или None при неудаче."""
        try:
            data = await self._client.complete_json(messages)
        except ValueError:
            logger.warning('LLMChunker: attempt %d — invalid JSON response', attempt)
            return None

        chunks = data.get('chunks')
        if not chunks or not isinstance(chunks, list):
            logger.warning(
                'LLMChunker: attempt %d — unexpected response structure (got %r)', attempt, data
            )
            return None

        result = [c for c in chunks if isinstance(c, str) and c.strip()]
        if not result:
            logger.warning(
                'LLMChunker: attempt %d — empty chunks list after filtering (raw chunks=%r)',
                attempt, chunks,
            )
            return None

        return result
