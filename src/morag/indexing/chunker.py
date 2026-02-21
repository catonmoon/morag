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

    При любой ошибке (невалидный JSON, пустой список, неверная структура)
    возвращает блок целиком как один чанк.
    """

    def __init__(self, client) -> None:
        self._client = client

    async def chunk(self, block: str) -> list[str]:
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': block},
        ]
        try:
            data = await self._client.complete_json(messages)
        except ValueError:
            logger.warning('LLMChunker: invalid JSON response, falling back to passthrough')
            return [block]

        chunks = data.get('chunks')
        if not chunks or not isinstance(chunks, list):
            logger.warning('LLMChunker: unexpected response structure, falling back to passthrough')
            return [block]

        result = [c for c in chunks if isinstance(c, str) and c.strip()]
        if not result:
            logger.warning('LLMChunker: empty chunks list, falling back to passthrough')
            return [block]

        return result
