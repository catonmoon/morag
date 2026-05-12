"""LLM-реранкер для retrieval pipeline.

Принимает query + список чанков, возвращает чанки в порядке релевантности
(первый — наиболее полезный). Inclusive-стратегия: включает связанные по смыслу
чанки, даже если формулировка отличается от запроса.

Бюджет input'а считается ТОЧНО по токенам через TokenCounter (как в DocReranker).
Если кандидатов больше чем влезает в окно — отрезаем хвост (greedy top-down по
порядку прихода из RRF). Это лучше чем глобальный `limit=20` — для маленьких
чанков влезает 40+, для больших 5-10. Bandwidth адаптивный.

OWUI/pipelines-независимо — чистый async API.
"""

from __future__ import annotations

import logging
import re

from morag.indexing.token_counter import TokenCounter
from morag.llm.client import GenerationParams, LLMClient

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    'Вопрос: "{query}"\n\n'
    'Чанки:\n{items}\n\n'
    'Какие из этих чанков могут быть полезны для ответа на вопрос? '
    'Если чанк содержит прямое определение термина, связанное с запросом — '
    'обязательно включай его, даже если формулировка отличается от запроса. '
    'Для оценки смысла предпочитай более свежие чанки (по дате «Обновлён»).\n'
    'Верни ТОЛЬКО номера чанков через запятую, '
    'В ПОРЯДКЕ РЕЛЕВАНТНОСТИ — более полезные первыми. '
    'Например: 3, 0, 5\n'
    'Если ни один не релевантен — верни: none'
)

# Накладные сверх skeleton+items: chat-template + safety buffer.
# Skeleton (template + query) и output reserve (= max_tokens) считаем точно.
_CHAT_OVERHEAD_TOKENS = 100
_SAFETY_BUFFER_TOKENS = 100
# Минимальный бюджет на input — даже на крошечном context_window вмещаем что-то.
_MIN_BUDGET_TOKENS = 2000
# Приближение токенов сепаратора '\n---\n' между chunk-items.
_SEPARATOR_TOKENS = 5


def _format_chunk_item(i: int, c: dict) -> str:
    """Формат одного chunk-item в rerank-промпте. Используется и для подсчёта токенов."""
    path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
    context = c.get('context', '')
    updated_at = c.get('updated_at', '')
    lines = [f'[{i}] {path_display}']
    if updated_at:
        lines.append(f'Обновлён: {updated_at}')
    if context:
        lines.append(f'Контекст: {context}')
    lines.append(c['text'])
    return '\n'.join(lines)


class LLMReranker:
    """Фильтр/ранжирование чанков через LLM с токен-бюджетом."""

    def __init__(
        self,
        llm_client: LLMClient,
        token_counter: TokenCounter,
        max_tokens: int = 100,
        enable_thinking: bool | None = False,
        max_input_tokens: int = 0,
    ) -> None:
        """
        :param token_counter: TokenCounter для подсчёта токенов skeleton + items.
        :param max_tokens: лимит на ответ LLM (только номера через запятую).
        :param enable_thinking: reasoning-флаг (None = не отправлять — для xAI Grok).
        :param max_input_tokens: override бюджета на input. 0 = auto от
            `llm.context_window - точные накладные`. >0 = ручной потолок.
        """
        self._llm = llm_client
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._max_input_tokens_override = max_input_tokens

    def _compute_budget(self, query: str) -> int:
        """Бюджет токенов на items (chunks) в одном rerank-вызове."""
        skeleton = _PROMPT_TEMPLATE.format(query=query, items='')
        skeleton_tokens = self._token_counter.count(skeleton)
        overhead = (
            skeleton_tokens
            + _CHAT_OVERHEAD_TOKENS
            + self._max_tokens
            + _SAFETY_BUFFER_TOKENS
        )
        available = self._llm.context_window - overhead
        if self._max_input_tokens_override > 0:
            available = min(available, self._max_input_tokens_override)
        return max(_MIN_BUDGET_TOKENS, available)

    def _fit_to_budget(
        self, chunks: list[dict], budget: int,
    ) -> tuple[list[dict], int]:
        """Greedy top-down: набираем чанки до исчерпания бюджета.

        Возвращает (fitted, dropped_count). Если ОДИН чанк превышает бюджет —
        кладём его всё равно (хоть что-то отдать reranker'у; LLM обработает,
        возможно с truncation).
        """
        fitted: list[dict] = []
        used = 0
        for i, c in enumerate(chunks):
            item_text = _format_chunk_item(i, c)
            t = self._token_counter.count(item_text)
            extra = t + (_SEPARATOR_TOKENS if fitted else 0)
            if fitted and used + extra > budget:
                break
            fitted.append(c)
            used += extra
        return fitted, len(chunks) - len(fitted)

    async def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        budget = self._compute_budget(query)
        fitted, dropped = self._fit_to_budget(chunks, budget)
        logger.info(
            '[rerank] candidates=%d → fits=%d (dropped %d, budget=%d tokens)',
            len(chunks), len(fitted), dropped, budget,
        )
        items = [_format_chunk_item(i, c) for i, c in enumerate(fitted)]
        prompt = _PROMPT_TEMPLATE.format(query=query, items='\n---\n'.join(items))
        try:
            answer = (await self._llm.complete(
                [{'role': 'user', 'content': prompt}],
                params=GenerationParams(temperature=0.0, enable_thinking=self._enable_thinking),
                max_tokens=self._max_tokens,
            )).strip()
        except Exception as exc:
            logger.warning('rerank failed, returning fitted chunks unranked: %s', exc)
            return fitted
        if 'none' in answer.lower():
            return []
        indices = [int(x) for x in re.findall(r'\d+', answer)]
        filtered = [fitted[i] for i in indices if 0 <= i < len(fitted)]
        # Fallback если парсинг сломался: вернуть fitted набор (не терять recall)
        return filtered or fitted
