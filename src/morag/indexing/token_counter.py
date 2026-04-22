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

    @abstractmethod
    def truncate(self, text: str, limit: int) -> str:
        """Обрезать текст до limit токенов (с конца)."""
        ...


class TiktokenCounter(TokenCounter):
    """Реализация TokenCounter на базе tiktoken (приближение для LLM)."""

    def __init__(self, encoding: str = 'cl100k_base') -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        # disallowed_special=() — специальные токены вроде <|endoftext|> встречаются
        # в сырых текстах (Confluence, код), tiktoken по умолчанию raise'ает.
        # Треатим их как обычный текст.
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate(self, text: str, limit: int) -> str:
        tokens = self._enc.encode(text, disallowed_special=())
        if len(tokens) <= limit:
            return text
        return self._enc.decode(tokens[:limit])


class HuggingFaceTokenCounter(TokenCounter):
    """Реализация TokenCounter на базе HuggingFace tokenizer.

    Используется для точного подсчёта токенов embedder-модели
    (например, Qwen/Qwen3-Embedding-4B). Для русского текста разница с
    TikToken ~33-40% (TikToken переоценивает на смешанном контенте меньше).
    """

    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-4B') -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, limit: int) -> str:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= limit:
            return text
        return self._tokenizer.decode(tokens[:limit])
