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
You are an assistant that provides context for semantic chunks of documents.

Document summary:
{doc_summary}

Surrounding text of the document:
{doc_text}

Here is the chunk that needs context:
{chunk_text}

Now write a short summary that provides the necessary context for understanding this chunk.

Requirements:
- The summary must be short (2–3 sentences).
- Do not repeat the chunk text itself.
- Focus on the surrounding context in which this chunk appears.
- Respond in the same language as the original document.\
"""

def _extract_window(doc_text: str, char_offset: int, window_tokens: int,
                     counter: TokenCounter) -> str:
    """Извлечь окно текста вокруг позиции чанка в документе.

    Args:
        doc_text: полный текст документа.
        char_offset: позиция начала чанка в документе (символы).
        window_tokens: размер окна в токенах.
        counter: счётчик токенов.
    """
    if char_offset <= 0:
        return counter.truncate(doc_text, window_tokens)

    half_window = window_tokens // 2

    # Текст до позиции чанка
    before_text = doc_text[:char_offset]
    before_tokens = counter.count(before_text)
    if before_tokens > half_window:
        excess = before_tokens - half_window
        start_pos = len(counter.truncate(before_text, excess))
    else:
        start_pos = 0

    # Текст от позиции чанка
    after_text = doc_text[char_offset:]
    after_truncated = counter.truncate(after_text, half_window)
    end_pos = char_offset + len(after_truncated)

    return doc_text[start_pos:end_pos]


class ContextGenerator(ABC):
    """Интерфейс генерации контекстуального суммари для чанка."""

    @abstractmethod
    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '', *,
                       char_offset: int = 0,
                       path: list[str] | None = None) -> str:
        """Сгенерировать суммари чанка в контексте всего документа.

        Args:
            doc_text: полный текст документа.
            chunk_text: текст чанка.
            doc_summary: суммари документа (если есть).
            char_offset: позиция начала чанка в документе (символы).
            path: путь документа (для точного расчёта embedder budget).
        """
        ...


class NoopContextGenerator(ContextGenerator):
    """Не генерирует суммари — возвращает пустую строку."""

    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '', *,
                       char_offset: int = 0,
                       path: list[str] | None = None) -> str:
        return ''


class LLMContextGenerator(ContextGenerator):
    """Генерирует суммари чанка через вызов LLM.

    Использует doc_summary + окно текста вокруг позиции чанка (window_tokens)
    вместо полного текста документа. Если window_tokens не задан — отправляет
    весь документ с обрезкой по context_window.

    Адаптивный max_output_tokens: если задан chunk_max_tokens (512 для FRIDA),
    лимит ответа рассчитывается как chunk_max_tokens - chunk_tokens - path_overhead.
    Маленький чанк → больше контекста, большой → меньше. Это оптимизирует и качество
    (контекст заполняет всё доступное место в embedder), и скорость (LLM не генерирует
    лишнего).
    """

    _PATH_OVERHEAD_FALLBACK = 20  # запас если path не передан

    def __init__(
        self,
        client,
        token_counter: TokenCounter | None = None,
        embed_counter: TokenCounter | None = None,
        context_window: int = 32768,
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
        window_tokens: int | None = None,
        chunk_max_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._token_counter = token_counter or TiktokenCounter()  # для LLM context window
        self._embed_counter = embed_counter or self._token_counter  # для chunk_max_tokens
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._params = _llm_params(enable_thinking)
        self._window_tokens = window_tokens
        self._chunk_max_tokens = chunk_max_tokens
        self._prompt_overhead = self._token_counter.count(
            _PROMPT_TEMPLATE.format(doc_text='', chunk_text='', doc_summary='')
        )

    def _adaptive_max_tokens(self, chunk_tokens: int, path_tokens: int) -> int | None:
        """Вычислить адаптивный лимит ответа: сколько влезет в embedder."""
        if self._chunk_max_tokens is not None:
            adaptive = self._chunk_max_tokens - chunk_tokens - path_tokens
            adaptive = max(32, adaptive)  # минимум 32 токена контекста
            if self._max_output_tokens is not None:
                return min(adaptive, self._max_output_tokens)
            return adaptive
        return self._max_output_tokens

    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '', *,
                       char_offset: int = 0,
                       path: list[str] | None = None) -> str:
        # embed_counter — для адаптивного лимита (сколько влезет в embedder)
        embed_chunk_tokens = self._embed_counter.count(chunk_text)
        if path is not None:
            # +1 за separator \n между path и text
            path_tokens = self._embed_counter.count('\n'.join(path)) + 1
        else:
            path_tokens = self._PATH_OVERHEAD_FALLBACK
        max_tokens = self._adaptive_max_tokens(embed_chunk_tokens, path_tokens)
        output_reserve = max_tokens if max_tokens is not None else 0

        # llm_counter — для расчёта промпта LLM
        chunk_tokens = self._token_counter.count(chunk_text)
        summary_tokens = self._token_counter.count(doc_summary)
        available_for_doc = (
            self._context_window - self._prompt_overhead
            - chunk_tokens - summary_tokens - output_reserve
        )

        if available_for_doc <= 0:
            logger.warning(
                'LLMContextGenerator: chunk (%d) + summary (%d) + overhead (%d) + reserve (%d) '
                'exceeds context window (%d), skipping context',
                chunk_tokens, summary_tokens, self._prompt_overhead,
                output_reserve, self._context_window,
            )
            return ''

        # Окно вокруг позиции чанка или полный документ
        if self._window_tokens:
            window_limit = min(self._window_tokens, available_for_doc)
            doc_text = _extract_window(doc_text, char_offset, window_limit, self._token_counter)
        else:
            doc_token_count = self._token_counter.count(doc_text)
            if doc_token_count > available_for_doc:
                logger.info(
                    'LLMContextGenerator: doc_text truncated from %d to %d tokens',
                    doc_token_count, available_for_doc,
                )
                doc_text = self._token_counter.truncate(doc_text, available_for_doc)

        prompt = _PROMPT_TEMPLATE.format(
            doc_summary=doc_summary, doc_text=doc_text, chunk_text=chunk_text,
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            result = await self._client.complete(
                messages, params=self._params, max_tokens=max_tokens,
            )
            logger.info(
                'LLMContextGenerator: context %d tok (max %s) for chunk %d tok',
                self._token_counter.count(result),
                max_tokens, chunk_tokens,
            )
            return result
        except Exception:
            logger.error(
                'LLMContextGenerator: failed to generate context for chunk (%d chars)',
                len(chunk_text),
            )
            raise
