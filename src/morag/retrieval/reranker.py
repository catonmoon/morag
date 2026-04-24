"""LLM-реранкер для retrieval pipeline.

Принимает query + список чанков, возвращает чанки в порядке релевантности
(первый — наиболее полезный). Inclusive-стратегия: включает связанные по смыслу
чанки, даже если формулировка отличается от запроса.

OWUI/pipelines-независимо — чистый async API.
"""

from __future__ import annotations

import logging
import re

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


class LLMReranker:
    """Фильтр/ранжирование чанков через LLM.

    Консьюмер передаёт query + chunks; получает отфильтрованные и
    отсортированные по релевантности. При сбое LLM возвращает исходные чанки
    (fallback — не теряем recall).
    """

    def __init__(self, llm_client: LLMClient, max_tokens: int = 100) -> None:
        self._llm = llm_client
        self._max_tokens = max_tokens

    async def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        items: list[str] = []
        for i, c in enumerate(chunks):
            path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
            context = c.get('context', '')
            updated_at = c.get('updated_at', '')
            lines = [f'[{i}] {path_display}']
            if updated_at:
                lines.append(f'Обновлён: {updated_at}')
            if context:
                lines.append(f'Контекст: {context}')
            lines.append(c['text'])
            items.append('\n'.join(lines))
        prompt = _PROMPT_TEMPLATE.format(query=query, items='\n---\n'.join(items))
        try:
            answer = (await self._llm.complete(
                [{'role': 'user', 'content': prompt}],
                params=GenerationParams(temperature=0.0, enable_thinking=False),
                max_tokens=self._max_tokens,
            )).strip()
        except Exception as exc:
            logger.warning('rerank failed, returning all chunks: %s', exc)
            return chunks
        if 'none' in answer.lower():
            return []
        indices = [int(x) for x in re.findall(r'\d+', answer)]
        filtered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        # Fallback если парсинг сломался: вернуть исходный набор (не терять recall)
        return filtered or chunks
