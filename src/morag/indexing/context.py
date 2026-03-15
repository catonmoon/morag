from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.llm.client import GenerationParams

logger = logging.getLogger(__name__)

def _llm_params(enable_thinking: bool | None = None) -> GenerationParams:
    return GenerationParams(
        temperature=0.0, top_p=1.0, top_k=0,
        frequency_penalty=0.0, presence_penalty=0.0, seed=42,
        enable_thinking=enable_thinking,
    )

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
- Сконцентрируйся на окружении, в котором находится этот фрагмент.
- Отвечай на языке оригинального документа.\
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

    Перед вызовом обрезает doc_text так, чтобы prompt + chunk + doc влезали
    в контекстное окно модели. При любой ошибке возвращает пустую строку.
    """

    def __init__(
        self,
        client,
        token_counter: TokenCounter | None = None,
        context_window: int = 32768,
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self._client = client
        self._token_counter = token_counter or TiktokenCounter()
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._params = _llm_params(enable_thinking)
        self._prompt_overhead = self._token_counter.count(
            _PROMPT_TEMPLATE.format(doc_text='', chunk_text='')
        )

    async def generate(self, doc_text: str, chunk_text: str) -> str:
        output_reserve = self._max_output_tokens if self._max_output_tokens is not None else 0
        chunk_tokens = self._token_counter.count(chunk_text)
        available_for_doc = self._context_window - self._prompt_overhead - chunk_tokens - output_reserve

        if available_for_doc <= 0:
            logger.warning(
                'LLMContextGenerator: chunk (%d tokens) + prompt overhead (%d) + output reserve (%d) '
                'exceeds context window (%d), skipping context',
                chunk_tokens, self._prompt_overhead, output_reserve, self._context_window,
            )
            return ''

        doc_token_count = self._token_counter.count(doc_text)
        if doc_token_count > available_for_doc:
            logger.info(
                'LLMContextGenerator: doc_text truncated from %d to %d tokens',
                doc_token_count, available_for_doc,
            )
            doc_text = self._token_counter.truncate(doc_text, available_for_doc)

        prompt = _PROMPT_TEMPLATE.format(doc_text=doc_text, chunk_text=chunk_text)
        messages = [{'role': 'user', 'content': prompt}]
        try:
            result = await self._client.complete(messages, params=self._params, max_tokens=self._max_output_tokens)
            logger.info(
                'LLMContextGenerator: generated context (%d chars) for chunk (%d chars)',
                len(result), len(chunk_text),
            )
            return result
        except Exception:
            logger.warning('LLMContextGenerator: failed to generate context, returning empty string')
            return ''
