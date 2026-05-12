"""LLM-реранкер для одного документа в режиме get_doc.

Отличается от LLMReranker:
1. Все фрагменты принадлежат ОДНОМУ документу — путь `doc.path` подаётся
   ОДИН раз как «глобальный контекст», а не дублируется per-chunk.
2. На входе lite-чанки `[{order, text}]` — нет context/path/прочего (rerank это
   не использует, цель — выбор номеров; полные чанки fetch'атся после).
3. Батчинг по ТОЧНОМУ бюджету токенов: считаем skeleton-промпт точно
   (через TokenCounter), оставляем чёткий резерв под output + chat-format
   overhead. Никаких магических констант.
4. Результат — list[int] из `order`-ов (не chunk-объекты).

OWUI/pipelines-независимо — чистый async API.
"""
from __future__ import annotations

import logging
import re

from morag.indexing.token_counter import TokenCounter
from morag.llm.client import GenerationParams, LLMClient

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    'Ниже — фрагменты ОДНОГО документа по пути: {doc_path}\n\n'
    'Вопрос пользователя: "{query}"\n\n'
    'Фрагменты:\n{items}\n\n'
    'Выбери НОМЕРА фрагментов которые могут содержать ответ на вопрос. '
    'Включай и связанные по смыслу фрагменты, даже если формулировка '
    'отличается от запроса.\n'
    'Верни ТОЛЬКО номера через запятую (без объяснений). '
    'Пример: 3, 7, 12\n'
    'Если ни один фрагмент не релевантен — верни: none'
)

# Накладные сверх skeleton+chunks:
#   - chat-template (role markers, BOS/EOS): ~100 токенов
#   - safety buffer: 100
# Skeleton (template + path + query) и output reserve (= max_tokens) считаем точно.
_CHAT_OVERHEAD_TOKENS = 100
_SAFETY_BUFFER_TOKENS = 100
# Минимальный бюджет на батч — даже при крошечном context_window вмещаем хоть что-то.
_MIN_BATCH_BUDGET_TOKENS = 2000


class DocReranker:
    """Фильтр чанков одного документа через LLM с батчингом по токенам."""

    def __init__(
        self,
        llm_client: LLMClient,
        token_counter: TokenCounter,
        max_tokens: int = 200,
        enable_thinking: bool | None = False,
        max_input_tokens: int = 0,
    ) -> None:
        """
        :param token_counter: реализация TokenCounter для подсчёта токенов
            (TiktokenCounter в проде; можно подменить на HF Qwen tokenizer
            если нужна точность для русского).
        :param max_tokens: лимит на ответ LLM (короткий — только номера).
        :param enable_thinking: reasoning-флаг (False для скорости).
        :param max_input_tokens: override бюджета на input одного батча.
            0 = auto от `llm.context_window - точные накладные`.
            >0 = вручную ограничить (например, для «иголка в стоге сена» —
            форсировать мелкие батчи чтобы LLM не пропускал редкий чанк).
        """
        self._llm = llm_client
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._max_input_tokens_override = max_input_tokens

    def _compute_budget(self, doc_path_str: str, query: str) -> int:
        """Сколько токенов отводим под чанки в одном батче.

        Точное вычисление:
            available = context_window
                        - skeleton_tokens (template + path + query)
                        - chat_overhead
                        - output_reserve (= max_tokens)
                        - safety
        Если override задан — берём min(available, override).
        """
        skeleton = _PROMPT_TEMPLATE.format(
            doc_path=doc_path_str, query=query, items='',
        )
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
        return max(_MIN_BATCH_BUDGET_TOKENS, available)

    def _make_batches(self, chunks: list[dict], budget: int) -> list[list[dict]]:
        """Гриди-разбиение чанков на батчи по бюджету токенов."""
        batches: list[list[dict]] = []
        current: list[dict] = []
        current_tokens = 0
        for c in chunks:
            t = self._token_counter.count(c['text']) + 16  # +16 на префикс «[N] »
            if current and (current_tokens + t > budget):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(c)
            current_tokens += t
        if current:
            batches.append(current)
        return batches

    async def _rerank_batch(
        self, query: str, doc_path_str: str, batch: list[dict],
    ) -> list[int]:
        """Один LLM-вызов для одного батча. Возвращает order'ы релевантных чанков."""
        items: list[str] = []
        for i, c in enumerate(batch):
            items.append(f'[{i}] {c["text"]}')
        prompt = _PROMPT_TEMPLATE.format(
            query=query, doc_path=doc_path_str,
            items='\n---\n'.join(items),
        )
        try:
            answer = (await self._llm.complete(
                [{'role': 'user', 'content': prompt}],
                params=GenerationParams(
                    temperature=0.0,
                    enable_thinking=self._enable_thinking,
                ),
                max_tokens=self._max_tokens,
            )).strip()
        except Exception as exc:
            logger.warning('doc_rerank batch failed, keeping all in batch: %s', exc)
            return [c['order'] for c in batch]
        if 'none' in answer.lower():
            return []
        indices = [int(x) for x in re.findall(r'\d+', answer)]
        # Маппим i (позиция в батче) → order (позиция в документе)
        keep: list[int] = []
        for i in indices:
            if 0 <= i < len(batch):
                keep.append(batch[i]['order'])
        return keep

    async def rerank(
        self, query: str, doc_path: list[str], chunks: list[dict],
    ) -> list[int]:
        """Вернуть order'ы релевантных чанков. Чанки должны иметь поля 'order' + 'text'.

        Порядок выходного списка — по позиции в документе (order asc).
        """
        if not chunks:
            return []
        doc_path_str = ' / '.join(doc_path) if doc_path else '(без пути)'
        budget = self._compute_budget(doc_path_str, query)
        batches = self._make_batches(chunks, budget)
        logger.info(
            '[doc_rerank] q=%r chunks=%d batches=%d budget=%d tokens',
            query[:80], len(chunks), len(batches), budget,
        )
        useful_orders: set[int] = set()
        for bi, batch in enumerate(batches):
            orders = await self._rerank_batch(query, doc_path_str, batch)
            logger.info(
                '[doc_rerank] batch %d/%d: %d chunks → kept %d',
                bi + 1, len(batches), len(batch), len(orders),
            )
            useful_orders.update(orders)
        return sorted(useful_orders)
