from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from morag.llm.client import GenerationParams

logger = logging.getLogger(__name__)

_LLM_PARAMS = GenerationParams(
    temperature=0.0, top_p=1.0, top_k=0,
    frequency_penalty=0.0, presence_penalty=0.0, seed=42,
)

_CHUNKS_SCHEMA = {
    'type': 'object',
    'properties': {
        'chunks': {
            'type': 'array',
            'items': {'type': 'string'},
        },
    },
    'required': ['chunks'],
    'additionalProperties': False,
}

_SYSTEM_PROMPT = """\
Ты — анализатор технической документации для RAG-системы.
Твоя задача — разбить текст на смысловые чанки для семантического поиска.
Чанк — это минимальная самостоятельная единица знания, которая имеет смысл без чтения соседних чанков.

Хороший чанк должен:
- Описывать одну тему или один аспект темы
- Быть понятным сам по себе
- Содержать завершённую мысль
- Быть полезным для поиска и ответа на вопрос

Разбивай текст по смыслу, а не по размеру.

Не дроби текст без необходимости:
Если несколько абзацев описывают одну тему — это один чанк.

Разделяй текст если:
- начинается новая функция или возможность
- начинается новый алгоритм
- начинается новый процесс
- меняется тема
- начинается новый раздел документации

Каждый чанк должен содержать достаточно контекста чтобы быть понятным отдельно.
Таблицы копируй полностью без изменений.

Не включай:
- оглавление
- навигацию
- повторяющиеся элементы
- номера страниц
- шаблонные подписи

Очень важно:

НЕ пересказывай текст.
НЕ интерпретируй текст.
НЕ сокращай текст.
НЕ добавляй новый текст.

Каждый чанк должен быть точной копией соответствующего участка исходного текста.
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
            data = await self._client.complete_json(messages, schema=_CHUNKS_SCHEMA, schema_name='chunks', params=_LLM_PARAMS)
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
