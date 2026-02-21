from abc import ABC, abstractmethod

import tiktoken


class TokenCounter(ABC):
    """Интерфейс для подсчёта токенов."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Вернуть количество токенов в тексте."""
        ...

    def fits(self, text: str, limit: int) -> bool:
        """Проверить, вписывается ли текст в лимит токенов."""
        return self.count(text) <= limit


class TiktokenCounter(TokenCounter):
    """Реализация TokenCounter на базе tiktoken."""

    def __init__(self, encoding: str = 'cl100k_base') -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))
