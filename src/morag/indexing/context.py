from __future__ import annotations

import logging
import re
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

_PAGE_RE = re.compile(r'<!-- page:(\d+) -->')


def _extract_window(doc_text: str, chunk_text: str, window_tokens: int,
                     counter: TokenCounter) -> str:
    """Извлечь окно текста вокруг страницы чанка."""
    # Определить номер страницы из чанка
    page_match = _PAGE_RE.search(chunk_text)
    if not page_match:
        # Нет маркера страницы — обрезать от начала
        return counter.truncate(doc_text, window_tokens)

    page_num = int(page_match.group(1))

    # Найти позицию этой страницы в документе
    page_positions = [(m.start(), int(m.group(1))) for m in _PAGE_RE.finditer(doc_text)]
    if not page_positions:
        return counter.truncate(doc_text, window_tokens)

    # Найти позицию целевой страницы
    target_pos = 0
    for pos, num in page_positions:
        if num == page_num:
            target_pos = pos
            break

    # Берём текст вокруг целевой позиции
    # Считаем половину окна в каждую сторону
    half_window = window_tokens // 2

    # Ищем начало: отступаем назад от target_pos
    before_text = doc_text[:target_pos]
    before_tokens = counter.count(before_text)
    if before_tokens > half_window:
        # Обрезаем начало, оставляя half_window токенов перед target_pos
        # truncate берёт от начала, нам нужно от конца — берём suffix
        excess = before_tokens - half_window
        # Грубая оценка: пропускаем excess токенов от начала
        start_pos = len(counter.truncate(before_text, excess))
    else:
        start_pos = 0

    # Ищем конец: берём от target_pos + half_window токенов
    after_text = doc_text[target_pos:]
    after_truncated = counter.truncate(after_text, half_window)
    end_pos = target_pos + len(after_truncated)

    return doc_text[start_pos:end_pos]


class ContextGenerator(ABC):
    """Интерфейс генерации контекстуального суммари для чанка."""

    @abstractmethod
    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '') -> str:
        """Сгенерировать суммари чанка в контексте всего документа.

        Возвращает строку: краткое содержание документа + роль данного чанка.
        Пустая строка означает отсутствие суммари.
        """
        ...


class NoopContextGenerator(ContextGenerator):
    """Не генерирует суммари — возвращает пустую строку."""

    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '') -> str:
        return ''


class LLMContextGenerator(ContextGenerator):
    """Генерирует суммари чанка через вызов LLM.

    Использует doc_summary + окно текста вокруг страницы чанка (window_tokens)
    вместо полного текста документа. Если window_tokens не задан — отправляет
    весь документ с обрезкой по context_window.
    """

    def __init__(
        self,
        client,
        token_counter: TokenCounter | None = None,
        context_window: int = 32768,
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
        window_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._token_counter = token_counter or TiktokenCounter()
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._params = _llm_params(enable_thinking)
        self._window_tokens = window_tokens
        self._prompt_overhead = self._token_counter.count(
            _PROMPT_TEMPLATE.format(doc_text='', chunk_text='', doc_summary='')
        )

    async def generate(self, doc_text: str, chunk_text: str,
                       doc_summary: str = '') -> str:
        output_reserve = self._max_output_tokens if self._max_output_tokens is not None else 0
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

        # Окно вокруг страницы чанка или полный документ
        if self._window_tokens:
            window_limit = min(self._window_tokens, available_for_doc)
            doc_text = _extract_window(doc_text, chunk_text, window_limit, self._token_counter)
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
                messages, params=self._params, max_tokens=self._max_output_tokens,
            )
            logger.info(
                'LLMContextGenerator: generated context (%d chars) for chunk (%d chars)',
                len(result), len(chunk_text),
            )
            return result
        except Exception:
            logger.error(
                'LLMContextGenerator: failed to generate context for chunk (%d chars)',
                len(chunk_text),
            )
            raise
