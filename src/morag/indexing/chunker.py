from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from morag.indexing.splitter import (
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    pack_blocks,
    split_into_semantic_units,
)
from morag.indexing.token_counter import TokenCounter
from morag.llm.client import GenerationParams

logger = logging.getLogger(__name__)

def _llm_params(enable_thinking: bool | None = None) -> GenerationParams:
    return GenerationParams(
        temperature=0.0, top_p=1.0, top_k=0,
        frequency_penalty=0.0, presence_penalty=0.0, seed=42,
        enable_thinking=enable_thinking,
    )

_CHUNKS_SCHEMA = {
    'type': 'object',
    'properties': {
        'chunks': {
            'type': 'array',
            'items': {'type': 'string'},
        },
    },
    'required': ['chunks'],
    'additionalProperties': False,
}

_SYSTEM_PROMPT = """\
You are a technical documentation analyzer for a RAG system.
Your task is to split the provided text into semantic chunks for semantic search.
A chunk is the smallest self-contained unit of knowledge that makes sense without reading neighboring chunks.

A good chunk should:
- Cover one topic or one aspect of a topic
- Be understandable on its own
- Contain a complete thought
- Be useful for search and answering questions
- Be formatted in Markdown
- Preserve original Markdown formatting where possible
- Preserve all links exactly as they are (URLs and Markdown links)

Split by meaning, not by size.
Do not over-split: if several paragraphs describe one topic, they form a single chunk.

Split the text when:
- A new feature or capability begins
- A new algorithm begins
- A new process begins
- The topic changes
- A new documentation section begins

Each chunk must contain enough context to be understandable on its own.

The input text may be:
- Part of a document
- An entire section of a document
- An entire document

If the input text already looks like a single logically complete block (e.g. a list):
- Return one chunk equal to the input text
- Do not split it further

=====================
TABLE HANDLING
=====================

If the input text contains a table:
- Convert each table row (except the header row) into a meaningful sentence
- Each table row should become a separate chunk
- The sentence must fully convey the meaning of the row
- The sentence must be understandable without viewing the table
- Use column names as semantic parts of the sentence
- Use natural sentences in the language of the original document
- Do not list values separated by spaces or commas
- If the table contains links, preserve them in Markdown format
- It is acceptable to combine sentences from multiple rows into one chunk

Пример преобразования таблицы:
Вход:
| Parameter | Default | Description |
|----------|---------|-------------|
| timeout  | 30      | Request timeout in seconds |
| retries  | 3       | Number of retry attempts |

Выход:
{
  "chunks": [
    "Параметр **timeout** имеет значение по умолчанию 30 секунд и задаёт время ожидания запроса.",
    "Параметр **retries** имеет значение по умолчанию 3 и задаёт количество повторных попыток."
  ]
}


Пример 2:
Вход:
| Model | Documentation |
|------|---------------|
| resnet50 | https://example.com/resnet |
| bert-base | https://example.com/bert |

Выход:
{
  "chunks": [
    "Документация модели **resnet50** доступна по ссылке https://example.com/resnet. Документация модели **bert-base** доступна по ссылке https://example.com/bert."
  ]
}

=====================
EXCLUDE
=====================
- Table of contents
- Navigation elements
- Repeated elements
- Page numbers
- Boilerplate signatures

=====================
IMPORTANT (for everything except tables)
=====================
DO NOT paraphrase the text.
DO NOT interpret the text.
DO NOT shorten the text.
DO NOT add new text.

Each chunk must be an exact copy of the corresponding section of the source text (except tables).
All Markdown elements except tables must be preserved:
- Links
- Lists
- Code
- Headings
- Formatting

Respond in the same language as the original document.
"""



class Chunker(ABC):
    """Интерфейс разбивки текстового блока на чанки."""

    @abstractmethod
    async def chunk(self, block: str) -> list[str]:
        """Разбить блок на список текстов чанков."""
        ...


class PassthroughChunker(Chunker):
    """Возвращает блок как есть — один блок равен одному чанку."""

    async def chunk(self, block: str) -> list[str]:
        return [block]


class ChunkingError(Exception):
    """Ошибка чанкинга LLM — все попытки исчерпаны, fallback отключён."""


class _LLMError:
    """Классификация ошибки LLM-вызова."""
    TIMEOUT = 'timeout'              # блок слишком большой → halving
    INVALID_JSON = 'invalid_json'    # модели не хватило контекста → halving
    OTHER = 'other'


def _classify_error(exc: Exception) -> str:
    """Определить тип ошибки LLM для выбора стратегии halving."""
    cls_name = type(exc).__name__
    msg = str(exc).lower()

    # openai.APITimeoutError или httpx.ReadTimeout
    if 'timeout' in cls_name.lower() or 'timeout' in msg:
        return _LLMError.TIMEOUT

    return _LLMError.OTHER


class LLMChunker(Chunker):
    """Разбивает блок на семантические чанки через LLM со structured output.

    Двухуровневая обработка ошибок:
    1. Таймаут / невалидный JSON → адаптивное деление блока пополам (halving_retries раз).
    2. Прочие ошибки → повторные попытки (max_retries).

    Ожидание перезагрузки модели вынесено в LLMClient (model_wait_seconds / model_wait_retries).

    После исчерпания всех попыток:
    - fallback_enabled=True → семантический fallback (RecursiveSplitter).
    - fallback_enabled=False → ChunkingError (документ пропускается, переиндексация при следующем запуске).
    """

    _FALLBACK_TOKEN_LIMIT = 512

    def __init__(
        self,
        client,
        max_retries: int = 3,
        token_counter: TokenCounter | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        fallback_token_limit: int = _FALLBACK_TOKEN_LIMIT,
        halving_retries: int = 0,
        fallback_enabled: bool = False,
        enable_thinking: bool | None = None,
    ) -> None:
        self._client = client
        self._max_retries = max_retries
        self._token_counter = token_counter
        self._embed_fn = embed_fn
        self._fallback_token_limit = fallback_token_limit
        self._halving_retries = halving_retries
        self._fallback_enabled = fallback_enabled
        self._params = _llm_params(enable_thinking)

    async def chunk(self, block: str) -> list[str]:
        return await self._chunk_with_halving(block, halving_left=self._halving_retries)

    async def _chunk_with_halving(
        self, block: str, halving_left: int,
    ) -> list[str]:
        """Попытка LLM-чанкинга с halving при таймауте / невалидном JSON.

        TIMEOUT → openai уже сделал свои retry внутри, повторять бессмысленно → halving.
        INVALID_JSON → блок слишком большой для контекста → halving.
        OTHER → повторные попытки (max_retries).
        """
        seen_errors: set[str] = set()

        for attempt in range(1, self._max_retries + 1):
            result, error_type = await self._try_chunk(block, attempt)
            if result is not None:
                return result
            seen_errors.add(error_type)

            # Транспортные ошибки: openai уже сделал свои retry внутри —
            # повторять бессмысленно, сразу переходим к halving.
            if error_type == _LLMError.TIMEOUT:
                break

        # Timeout/invalid_json → блок слишком большой → halving
        _HALVABLE = {_LLMError.TIMEOUT, _LLMError.INVALID_JSON}
        if seen_errors & _HALVABLE and halving_left > 0:
            return await self._halve_and_retry(block, halving_left)

        # Последний рубеж: семантический fallback или исключение
        return self._fallback_or_raise(block)

    async def _halve_and_retry(
        self, block: str, halving_left: int,
    ) -> list[str]:
        """Разбить блок пополам и рекурсивно прогнать каждый подблок."""
        if self._token_counter is None:
            logger.warning(
                'LLMChunker: timeout on block (%d chars), '
                'no token_counter for halving, using fallback/raise',
                len(block),
            )
            return self._fallback_or_raise(block)

        block_tokens = self._token_counter.count(block)
        half_limit = block_tokens // 2

        # Используем RecursiveSplitter чтобы разбить блок на подблоки ≤ half_limit
        splitter = RecursiveSplitter(
            self._token_counter, half_limit,
            splitters=[
                MarkdownHeaderSplitter(),
                FixedSizeSplitter(self._token_counter, half_limit),
            ],
        )
        sub_blocks_raw = splitter.split(block)
        sub_blocks = pack_blocks(sub_blocks_raw, self._token_counter, half_limit)

        logger.info(
            'LLMChunker: timeout → halving block (%d chars, ~%d tokens) '
            'into %d sub-block(s) of ≤%d tokens (halving_left=%d)',
            len(block), block_tokens, len(sub_blocks), half_limit, halving_left - 1,
        )

        all_chunks: list[str] = []
        for i, pack in enumerate(sub_blocks):
            sub_text = '\n\n'.join(pack)
            logger.info(
                'LLMChunker: processing sub-block %d/%d (%d chars)',
                i + 1, len(sub_blocks), len(sub_text),
            )
            chunks = await self._chunk_with_halving(sub_text, halving_left - 1)
            all_chunks.extend(chunks)

        return all_chunks

    def _fallback_or_raise(self, block: str) -> list[str]:
        """Семантический fallback если включён, иначе — исключение."""
        if self._fallback_enabled:
            return self._fallback_split(block)

        raise ChunkingError(
            f'LLMChunker: all attempts failed for block ({len(block)} chars), '
            f'fallback disabled — document will be re-indexed on next run'
        )

    def _fallback_split(self, block: str) -> list[str]:
        """Семантический fallback: RecursiveSplitter с цепочкой сплиттеров."""
        if self._token_counter is None:
            logger.error(
                'LLMChunker: all attempts failed for block (%d chars), '
                'no token_counter for fallback, returning block as-is',
                len(block),
            )
            return [block]

        limit = self._fallback_token_limit
        splitters = [MarkdownHeaderSplitter()]
        if self._embed_fn is not None:
            splitters.append(SemanticSplitter(
                embed_fn=self._embed_fn, breakpoint_percentile=90, min_sentences=3,
            ))
        splitters.append(FixedSizeSplitter(self._token_counter, limit))

        recursive = RecursiveSplitter(self._token_counter, limit, splitters)
        chunks = recursive.split(block)

        mode = 'semantic' if self._embed_fn is not None else 'fixed-size'
        logger.warning(
            'LLMChunker: all attempts failed for block (%d chars), '
            '%s fallback split into %d chunk(s) of ≤%d tokens',
            len(block), mode, len(chunks), limit,
        )
        return chunks

    async def _try_chunk(
        self, block: str, attempt: int,
    ) -> tuple[list[str] | None, str]:
        """Одна попытка чанкинга. Возвращает (чанки, тип_ошибки)."""
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': block},
        ]
        logger.debug(
            'LLMChunker: attempt %d — input block (%d chars): %s',
            attempt, len(block), block.replace('\n', '\\n'),
        )
        try:
            data = await self._client.complete_json(
                messages, schema=_CHUNKS_SCHEMA,
                schema_name='chunks', params=self._params,
            )
        except ValueError:
            logger.warning(
                'LLMChunker: attempt %d — invalid JSON response [%s] for block (%d chars)',
                attempt, _LLMError.INVALID_JSON, len(block), exc_info=True,
            )
            return None, _LLMError.INVALID_JSON
        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                'LLMChunker: attempt %d — LLM call failed [%s] for block (%d chars)',
                attempt, error_type, len(block), exc_info=True,
            )
            return None, error_type

        chunks = data.get('chunks')
        if not chunks or not isinstance(chunks, list):
            logger.warning(
                'LLMChunker: attempt %d — unexpected response structure (got %r)',
                attempt, data,
            )
            return None, _LLMError.OTHER

        result = [c for c in chunks if isinstance(c, str) and c.strip()]
        if not result:
            logger.warning(
                'LLMChunker: attempt %d — empty chunks list after filtering (raw chunks=%r)',
                attempt, chunks,
            )
            return None, _LLMError.OTHER

        return result, ''


class SemanticChunker(Chunker):
    """Чанкер на основе семантических границ через эмбеддинги.

    Жадный алгоритм слева направо: для каждой границы генерирует пары кандидатов
    (левый чанк, правый чанк) в диапазоне [min_tokens, max_tokens], батчем эмбеддит
    все кандидаты и выбирает пару с максимальным cosine distance.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        counter: TokenCounter,
        min_tokens: int = 50,
        max_tokens: int = 250,
        accept_pair: bool = False,
    ) -> None:
        self._embed_fn = embed_fn
        self._counter = counter
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._accept_pair = accept_pair

    async def chunk(self, block: str) -> list[str]:
        sentences = split_into_semantic_units(
            block, self._counter, self._max_tokens,
        )
        if not sentences:
            return [block] if block.strip() else []

        sentence_tokens = [self._counter.count(s) for s in sentences]
        total_tokens = sum(sentence_tokens)
        oversized_units = sum(1 for t in sentence_tokens if t > self._max_tokens)
        logger.info(
            'SemanticChunker: %d unit(s), ~%d tokens total, %d oversized (>%d)',
            len(sentences), total_tokens, oversized_units, self._max_tokens,
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, (s, t) in enumerate(zip(sentences, sentence_tokens)):
                logger.debug('  unit[%d] %d tok: %s', i, t, repr(s[:80]))

        # Весь блок влезает в max_tokens — один чанк
        if total_tokens <= self._max_tokens:
            logger.info('SemanticChunker: entire block fits in max_tokens, 1 chunk')
            return [block]

        chunks: list[str] = []
        pos = 0

        while pos < len(sentences):
            remaining_tokens = sum(sentence_tokens[pos:])

            # Остаток влезает в max_tokens — последний чанк
            if remaining_tokens <= self._max_tokens:
                chunks.append(' '.join(sentences[pos:]))
                break

            # Индексы конца (exclusive) левых кандидатов: [min_tokens, max_tokens]
            left_ends = self._candidate_ends(sentence_tokens, pos, forward=True)

            if not left_ends:
                # Ни одно предложение не набирает min_tokens — берём до max_tokens
                end = self._greedy_end(sentence_tokens, pos)
                chunks.append(' '.join(sentences[pos:end]))
                pos = end
                continue

            if len(left_ends) == 1:
                # Единственный вариант — без сравнения
                end = left_ends[0]
                chunks.append(' '.join(sentences[pos:end]))
                pos = end
                continue

            # Для каждого левого кандидата — правые кандидаты
            # (left_end, right_end, left_text, right_text)
            pairs: list[tuple[int, int, str, str]] = []

            for left_end in left_ends:
                right_ends = self._candidate_ends(sentence_tokens, left_end, forward=True)
                if right_ends:
                    for right_end in right_ends:
                        left_text = ' '.join(sentences[pos:left_end])
                        right_text = ' '.join(sentences[left_end:right_end])
                        pairs.append((left_end, right_end, left_text, right_text))
                else:
                    # Остаток < min_tokens — берём что есть как правый кандидат
                    remaining = sentences[left_end:]
                    if remaining:
                        left_text = ' '.join(sentences[pos:left_end])
                        right_text = ' '.join(remaining)
                        pairs.append((left_end, len(sentences), left_text, right_text))

            if not pairs:
                end = left_ends[0]
                chunks.append(' '.join(sentences[pos:end]))
                pos = end
                continue

            # Батчевый embed всех уникальных текстов (в потоке, чтобы не блокировать event loop)
            unique_texts = list({t for _, _, lt, rt in pairs for t in (lt, rt)})
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._embed_fn, unique_texts)
            text_to_emb = dict(zip(unique_texts, embeddings))

            # Выбрать пару с максимальным cosine distance
            best_dist = -1.0
            best_left_end = left_ends[0]
            best_right_end = left_ends[0]

            for left_end, right_end, left_text, right_text in pairs:
                dist = self._cosine_distance(
                    text_to_emb[left_text], text_to_emb[right_text],
                )
                if dist > best_dist:
                    best_dist = dist
                    best_left_end = left_end
                    best_right_end = right_end

            chunks.append(' '.join(sentences[pos:best_left_end]))
            if self._accept_pair:
                right_tokens = sum(sentence_tokens[best_left_end:best_right_end])
                if right_tokens <= self._max_tokens:
                    chunks.append(' '.join(sentences[best_left_end:best_right_end]))
                    pos = best_right_end
                else:
                    pos = best_left_end
            else:
                pos = best_left_end

        result = chunks if chunks else [block]

        # Статистика по финальным чанкам
        chunk_token_counts = [self._counter.count(c) for c in result]
        avg_tokens = sum(chunk_token_counts) / len(chunk_token_counts) if chunk_token_counts else 0
        over_512 = sum(1 for t in chunk_token_counts if t > 512)
        over_max = sum(1 for t in chunk_token_counts if t > self._max_tokens)
        logger.info(
            'SemanticChunker: %d chunk(s), avg=%d tok, over_max(%d)=%d, over_512=%d',
            len(result), int(avg_tokens), self._max_tokens, over_max, over_512,
        )
        for i, (c, t) in enumerate(zip(result, chunk_token_counts)):
            if t > 512:
                logger.warning(
                    'SemanticChunker: chunk[%d] exceeds 512 tokens (%d tok): %s...',
                    i, t, repr(c[:100]),
                )
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug('  chunk[%d] %d tok: %s', i, t, repr(c[:80]))

        return result

    def _candidate_ends(
        self, sentence_tokens: list[int], start: int, *, forward: bool = True,
    ) -> list[int]:
        """Индексы конца (exclusive) кандидатов в диапазоне [min_tokens, max_tokens]."""
        ends: list[int] = []
        total = 0
        for i in range(start, len(sentence_tokens)):
            total += sentence_tokens[i]
            if total > self._max_tokens:
                break
            if total >= self._min_tokens:
                ends.append(i + 1)
        return ends

    def _greedy_end(self, sentence_tokens: list[int], start: int) -> int:
        """Набирает предложения до max_tokens, возвращает exclusive end."""
        total = 0
        end = start
        for i in range(start, len(sentence_tokens)):
            if total + sentence_tokens[i] > self._max_tokens and i > start:
                break
            total += sentence_tokens[i]
            end = i + 1
        return end

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Cosine distance между двумя векторами."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        similarity = float(np.dot(va, vb) / (norm + 1e-8))
        return 1.0 - similarity
