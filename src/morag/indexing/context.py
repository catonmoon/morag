from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Ты — ассистент, который помогает дать контекст смысловым фрагментам документов.

Вот полный текст документа, с которого был взят смысловой фрагмент:
{doc_text}

Вот сам смысловой фрагмент (чанк), для которого нужно сформулировать контекст:
{chunk_text}

Теперь сформулируй короткое обобщение содержания всего документа, которое даёт необходимый \
контекст для понимания этого фрагмента.

Требования:
- Обобщение должно быть коротким (2–3 предложения).
- Не нужно повторять сам текст чанка.
- Сконцентрируйся на окружении, в котором находится этот фрагмент.\
"""


class ContextGenerator(ABC):
    """Интерфейс генерации контекстуального суммари для чанка."""

    @abstractmethod
    async def generate(self, doc_text: str, chunk_text: str) -> str:
        """Сгенерировать суммари чанка в контексте всего документа.

        Возвращает строку: краткое содержание документа + роль данного чанка.
        Пустая строка означает отсутствие суммари.
        """
        ...


class NoopContextGenerator(ContextGenerator):
    """Не генерирует суммари — возвращает пустую строку."""

    async def generate(self, doc_text: str, chunk_text: str) -> str:
        return ''


class LLMContextGenerator(ContextGenerator):
    """Генерирует суммари чанка через вызов LLM.

    При любой ошибке возвращает пустую строку — индексация продолжается без контекста.
    """

    def __init__(self, client) -> None:
        self._client = client

    async def generate(self, doc_text: str, chunk_text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(doc_text=doc_text, chunk_text=chunk_text)
        messages = [{'role': 'user', 'content': prompt}]
        try:
            return await self._client.complete(messages, temperature=0.3)
        except Exception:
            logger.warning('LLMContextGenerator: failed to generate context, returning empty string')
            return ''
