from __future__ import annotations

from abc import ABC, abstractmethod


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
    """Генерирует суммари через вызов LLM."""

    async def generate(self, doc_text: str, chunk_text: str) -> str:
        raise NotImplementedError('LLMContextGenerator требует llm/client.py — будет реализован позже')
