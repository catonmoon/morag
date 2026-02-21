from __future__ import annotations

from abc import ABC, abstractmethod


class Chunker(ABC):
    """Интерфейс разбивки текстового блока на чанки."""

    @abstractmethod
    def chunk(self, block: str) -> list[str]:
        """Разбить блок на список текстов чанков."""
        ...


class PassthroughChunker(Chunker):
    """Возвращает блок как есть — один блок равен одному чанку."""

    def chunk(self, block: str) -> list[str]:
        return [block]


class LLMChunker(Chunker):
    """Разбивает блок на семантические чанки через LLM со structured output."""

    def chunk(self, block: str) -> list[str]:
        raise NotImplementedError('LLMChunker требует llm/client.py — будет реализован позже')
