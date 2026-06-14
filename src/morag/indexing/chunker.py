from __future__ import annotations

import dataclasses
import logging
import re
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

import numpy as np

from morag.indexing.splitter import (
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    TableRowSplitter,
    _split_by_headers,
    _top_level_blocks,
    pack_blocks,
    split_into_semantic_units,
    split_sentences,
)
from morag.indexing.token_counter import TokenCounter
from morag.llm.client import GenerationParams

logger = logging.getLogger(__name__)

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
        fallback_token_limit: int = _FALLBACK_TOKEN_LIMIT,
        halving_retries: int = 0,
        fallback_enabled: bool = False,
    ) -> None:
        self._client = client
        self._max_retries = max_retries
        self._token_counter = token_counter
        self._fallback_token_limit = fallback_token_limit
        self._halving_retries = halving_retries
        self._fallback_enabled = fallback_enabled

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
        splitters = [
            MarkdownHeaderSplitter(),
            FixedSizeSplitter(self._token_counter, limit),
        ]

        recursive = RecursiveSplitter(self._token_counter, limit, splitters)
        chunks = recursive.split(block)

        logger.warning(
            'LLMChunker: all attempts failed for block (%d chars), '
            'fixed-size fallback split into %d chunk(s) of ≤%d tokens',
            len(block), len(chunks), limit,
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
                schema_name='chunks',
                params=GenerationParams(enable_thinking=False),
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


_BOUNDARY_SCHEMA = {
    'type': 'object',
    'properties': {
        'segments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'start': {'type': 'integer'},
                    'end': {'type': 'integer'},
                    'topic': {'type': 'string'},
                },
                'required': ['start', 'end', 'topic'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['segments'],
    'additionalProperties': False,
}

_BOUNDARY_PROMPT = """\
Ты сегментируешь транскрипт/документ на тематические чанки для RAG-поиска.
Вход — пронумерованные юниты (абзацы / реплики), по одному на строку, в формате `[N] текст`.

Твоя задача — сгруппировать ПОДРЯД идущие юниты в тематические сегменты. Сегмент — это
законченная мысль / одна тема / один аспект, понятный сам по себе.

ВАЖНО — верни ТОЛЬКО границы по НОМЕРАМ юнитов, НЕ переписывай текст:
- Каждый сегмент = {start, end, topic}, где start/end — номера юнитов (включительно).
- Сегменты должны покрывать ВСЕ юниты без пропусков и нахлёстов:
  первый start = 1, последний end = <число юнитов>, и каждый следующий start = предыдущий end + 1.
- topic — короткая формулировка темы сегмента (3-7 слов).
- Дели по смене темы, а не по размеру. Не дроби одну мысль; не склеивай разные темы.

Верни строго JSON: {"segments": [{"start": 1, "end": 4, "topic": "..."}, ...]}.
"""


class BoundaryChunker(Chunker):
    """LLM-чанкер по ГРАНИЦАМ: LLM возвращает номера юнитов где чанк начинается/кончается,
    а не переписывает текст. Обходит проблему «LLM копирует текст в JSON → таймаут».

    Юниты = абзацы (split по пустой строке) — для аудио-транскриптов это реплики
    `[Имя] <!-- t:СЕК --> текст`, для обычного markdown — абзацы. Материализация чанка —
    склейка юнитов по диапазону [start, end]. Таймкоды/маркеры остаются в тексте — их
    извлекает TimestampProcessor на стадии chunk-processors.

    Валидатор границ: монотонность, покрытие [1..N], непрерывность, диапазон. При нарушении —
    1 retry с указанием ошибки; иначе fallback на склейку по token-бюджету (max_tokens).
    """

    _boundary_prompt = _BOUNDARY_PROMPT  # переопределяется в наследниках (TranscriptChunker)

    def __init__(
        self,
        client,
        token_counter: TokenCounter,
        max_tokens: int = 800,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._counter = token_counter
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def chunk(self, text: str) -> list[str]:
        results = await self.chunk_with_metadata(text)
        return [r.text for r in results]

    async def chunk_with_metadata(
        self, text: str, *, paged: bool = False,
    ) -> list[ChunkResult]:
        units = self._split_units(text)
        if not units:
            return []
        if len(units) == 1:
            return [ChunkResult(text=units[0])]

        segments = await self._get_boundaries(units)
        if segments is None:
            segments = self._fallback_segments(units)

        results: list[ChunkResult] = []
        for start, end in segments:
            chunk_text = '\n\n'.join(units[start - 1:end])
            if chunk_text.strip():
                results.append(ChunkResult(text=chunk_text))
        return results

    @staticmethod
    def _split_units(text: str) -> list[str]:
        """Юниты = непустые абзацы (split по пустой строке)."""
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    async def _get_boundaries(self, units: list[str]) -> list[tuple[int, int]] | None:
        numbered = '\n'.join(f'[{i + 1}] {u}' for i, u in enumerate(units))
        messages = [
            {'role': 'system', 'content': self._boundary_prompt},
            {'role': 'user', 'content': numbered},
        ]
        n = len(units)
        last_err = ''
        for attempt in range(1, self._max_retries + 2):
            msgs = messages
            if last_err:
                msgs = messages + [{
                    'role': 'user',
                    'content': f'Предыдущий ответ невалиден: {last_err}. '
                               f'Исправь: покрой юниты 1..{n} подряд без пропусков и нахлёстов.',
                }]
            try:
                data = await self._client.complete_json(
                    msgs, schema=_BOUNDARY_SCHEMA, schema_name='segments',
                    params=GenerationParams(enable_thinking=False),
                )
            except Exception:
                logger.warning('BoundaryChunker: attempt %d — LLM call failed', attempt, exc_info=True)
                continue
            segs = self._validate(data.get('segments'), n)
            if segs is not None:
                return segs
            last_err = self._last_validation_error
            logger.warning('BoundaryChunker: attempt %d — invalid boundaries: %s', attempt, last_err)
        return None

    _last_validation_error = ''

    def _validate(self, segments, n: int) -> list[tuple[int, int]] | None:
        """Проверить границы. Возвращает список (start, end) или None при нарушении."""
        if not segments or not isinstance(segments, list):
            self._last_validation_error = 'пустой/некорректный список segments'
            return None
        out: list[tuple[int, int]] = []
        expected = 1
        for s in segments:
            try:
                start, end = int(s['start']), int(s['end'])
            except (KeyError, TypeError, ValueError):
                self._last_validation_error = 'нет полей start/end'
                return None
            if start != expected:
                self._last_validation_error = f'разрыв/нахлёст: ожидался start={expected}, получен {start}'
                return None
            if end < start or end > n:
                self._last_validation_error = f'диапазон вне [start..{n}]: {start}-{end}'
                return None
            out.append((start, end))
            expected = end + 1
        if expected != n + 1:
            self._last_validation_error = f'покрытие неполное: последний end={expected - 1}, нужно {n}'
            return None
        return out

    def _fallback_segments(self, units: list[str]) -> list[tuple[int, int]]:
        """Fallback: жадно набираем юниты в чанки до max_tokens."""
        logger.warning('BoundaryChunker: fallback — greedy pack по %d токенов', self._max_tokens)
        segs: list[tuple[int, int]] = []
        start = 1
        acc = 0
        for i, u in enumerate(units, 1):
            t = self._counter.count(u)
            if acc and acc + t > self._max_tokens:
                segs.append((start, i - 1))
                start, acc = i, 0
            acc += t
        segs.append((start, len(units)))
        return segs


class SemanticChunker(Chunker):
    """Чанкер на основе семантических границ через эмбеддинги.

    Жадный алгоритм слева направо: для каждой границы генерирует пары кандидатов
    (левый чанк, правый чанк) в диапазоне [min_tokens, max_tokens], батчем эмбеддит
    все кандидаты и выбирает пару с максимальным cosine distance.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
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

            # Батчевый embed всех уникальных текстов
            unique_texts = list({t for _, _, lt, rt in pairs for t in (lt, rt)})
            embeddings = await self._embed_fn(unique_texts)
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


# ---------------------------------------------------------------------------
# HybridChunker
# ---------------------------------------------------------------------------

_DIAGRAM_LANGS = frozenset({'plantuml', 'mermaid', 'ditaa'})
_FENCE_OPEN_RE = re.compile(r'^(`{3,})\s*(.*)')
_PAGE_MARKER_RE = re.compile(r'<!-- page:(\d+) -->\n?')


def _merge_pages(a: list[int], b: list[int]) -> list[int]:
    """Объединить два списка страниц, сохранив уникальность и порядок."""
    return sorted(set(a) | set(b))


@dataclasses.dataclass
class _Block:
    """Атомарный блок документа после структурного разбора."""
    text: str
    block_type: str   # heading | paragraph | table | fence | diagram | list
    tokens: int
    pages: list[int] = dataclasses.field(default_factory=list)
    char_offset: int = 0  # позиция начала блока в оригинальном тексте документа


@dataclasses.dataclass
class ChunkResult:
    """Результат чанкинга: текст + метаданные позиционирования."""
    text: str
    pages: list[int] = dataclasses.field(default_factory=list)
    char_offset: int = 0  # позиция начала чанка в оригинальном тексте документа
    # транскрипт-метаданные (TranscriptChunker): аудио-оффсеты + спикеры чанка
    start_sec: float | None = None
    end_sec: float | None = None
    speakers: list[str] = dataclasses.field(default_factory=list)


# Промпт зафиксирован экспериментом (adventures/experiments/transcript_chunking_20260609.md): по-новостной
# («лента новостей»), сегмент = одна новость/тема. С max_tokens≈900 даёт retrieval recall@1 90-100%.
_TRANSCRIPT_PROMPT = """\
Ты сегментируешь ТРАНСКРИПТ подкаста/беседы про технологии на тематические чанки для RAG-поиска.
Вход — пронумерованные юниты (реплики спикеров или их части): `[N] [Спикер] текст`.

Разговор идёт КАК ЛЕНТА НОВОСТЕЙ/ТЕЗИСОВ: ведущие переходят «следующая новость», «перейдём к», «а ещё».
ПРАВИЛО: каждая ОТДЕЛЬНАЯ новость / тема / законченный тезис = ОТДЕЛЬНЫЙ сегмент.
- НЕ склеивай разные новости/темы в один сегмент, даже если каждая короткая.
- Реплики РАЗНЫХ спикеров про ОДНУ тему — ОДИН сегмент (дели по смене ТЕМЫ, не по смене спикера).
- Длинный монолог, где спикер перешёл к другой подтеме, — РАЗНЫЕ сегменты.
- Сегмент понятен сам по себе. Сомневаешься «одна тема или две» — ДЕЛИ (дробнее лучше, чем смешать).

Верни ТОЛЬКО границы по НОМЕРАМ юнитов (НЕ переписывай текст): каждый сегмент {start, end, topic},
покрой ВСЕ юниты подряд без пропусков/нахлёстов (первый start=1, последний end=<число юнитов>,
каждый следующий start = предыдущий end + 1; topic — 3-7 слов).
Верни строго JSON: {"segments": [{"start": 1, "end": 4, "topic": "..."}, ...]}.
"""

_TURN_RE = re.compile(
    r'^\[(?P<speaker>[^\]]+)\]\s*(?:<!--\s*t:(?P<t>[\d.]+)\s*-->)?\s*(?P<text>.*)$', re.S)


@dataclasses.dataclass
class _Unit:
    """Юнит транскрипт-чанкинга: реплика целиком или её часть (предложения длинного монолога)."""
    text: str
    speaker: str
    start: float | None
    end: float | None
    char_offset: int


class TranscriptChunker(BoundaryChunker):
    """Транскрипт-специализация BoundaryChunker (`mode: transcript`).

    Юниты = реплики `[Имя] <!-- t:сек --> текст`; длинная реплика режется на суб-юниты по предложениям
    (≤`unit_max_tokens`) с ИНТЕРПОЛЯЦИЕЙ таймкода → LLM может ставить границу ВНУТРИ монолога на смене
    темы (а не только на стыке реплик). Материализация переприписывает `[Имя]` чанку, начавшемуся в
    середине реплики (спикер/тайминг не теряются), и отдаёт `start_sec`/`end_sec`/`speakers`/`char_offset`.
    Размерный пост-пасс (`_cap_by_size`) добивает затянувшийся монолог одной темы по `max_tokens`.
    Таймкод внутри реплики интерполируется (TODO про тонкий тайминг — adventures/podlodka-asr/CLAUDE.md).
    """

    _boundary_prompt = _TRANSCRIPT_PROMPT

    def __init__(self, client, token_counter: TokenCounter, max_tokens: int = 900,
                 max_retries: int = 1, unit_max_tokens: int = 400) -> None:
        super().__init__(client, token_counter, max_tokens, max_retries)
        self._unit_max = unit_max_tokens

    async def chunk_with_metadata(self, text: str, *, paged: bool = False) -> list[ChunkResult]:
        turns = self._parse_turns(text)
        if not turns:
            return []
        units = self._build_units(turns)
        if not units:
            return []
        if len(units) == 1:
            return [self._materialize(units)]
        segments = await self._get_boundaries([f'[{u.speaker}] {u.text}' for u in units])
        if segments is None:
            segments = self._fallback_segments([u.text for u in units])
        segments = self._cap_by_size(units, segments)
        return [self._materialize(units[s - 1:e]) for s, e in segments]

    def _parse_turns(self, text: str) -> list[dict]:
        turns, pos = [], 0
        for para in re.split(r'\n\s*\n', text):
            stripped = para.strip()
            if not stripped:
                continue
            off = text.find(stripped, pos)
            pos = off + len(stripped) if off >= 0 else pos
            m = _TURN_RE.match(stripped)
            if m and m['text'].strip():
                turns.append({'speaker': m['speaker'].strip(),
                              'start': float(m['t']) if m['t'] else None,
                              'text': m['text'].strip(), 'char_offset': max(off, 0)})
        # end реплики = start следующей (с таймкодом); у последней — оценка по словам (~2.5 сл/с)
        for i, t in enumerate(turns):
            nxt = next((turns[j]['start'] for j in range(i + 1, len(turns))
                        if turns[j]['start'] is not None), None)
            t['end'] = nxt if nxt is not None else (
                t['start'] + len(t['text'].split()) / 2.5 if t['start'] is not None else None)
        return turns

    def _build_units(self, turns: list[dict]) -> list[_Unit]:
        units: list[_Unit] = []
        for t in turns:
            if self._counter.count(t['text']) <= self._unit_max:
                units.append(_Unit(t['text'], t['speaker'], t['start'], t['end'], t['char_offset']))
            else:
                units.extend(self._split_turn(t))
        return units

    def _split_turn(self, t: dict) -> list[_Unit]:
        """Длинную реплику → суб-юниты по предложениям ≤unit_max, таймкод интерполируется по позиции."""
        groups, cur, cur_tok = [], [], 0
        for s in split_sentences(t['text']):
            st = self._counter.count(s)
            if cur and cur_tok + st > self._unit_max:
                groups.append(' '.join(cur)); cur, cur_tok = [], 0
            cur.append(s); cur_tok += st
        if cur:
            groups.append(' '.join(cur))
        length = max(1, len(t['text']))
        t0, t1 = t['start'], t['end']
        out, cpos = [], 0
        for g in groups:
            start = (t0 + (cpos / length) * (t1 - t0)) if (t0 is not None and t1 is not None) else t0
            out.append(_Unit(g, t['speaker'], round(start, 1) if start is not None else None,
                             None, t['char_offset'] + cpos))
            cpos += len(g) + 1
        for i, u in enumerate(out):
            u.end = out[i + 1].start if i + 1 < len(out) else t1
        return out

    def _cap_by_size(self, units: list[_Unit], segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Затянувшийся сегмент одной темы (> max_tokens) добиваем по юнитам, не дробя юнит."""
        out: list[tuple[int, int]] = []
        for s, e in segments:
            start, tok = s, 0
            for idx in range(s, e + 1):
                ut = self._counter.count(units[idx - 1].text)
                if idx > start and tok + ut > self._max_tokens:
                    out.append((start, idx - 1)); start, tok = idx, 0
                tok += ut
            out.append((start, e))
        return out

    @staticmethod
    def _materialize(seg_units: list[_Unit]) -> ChunkResult:
        lines, prev = [], None
        for u in seg_units:                      # склейка: новая `[Имя]` при смене спикера
            if u.speaker != prev:
                lines.append(f'[{u.speaker}] {u.text}'); prev = u.speaker
            else:
                lines[-1] += ' ' + u.text
        starts = [u.start for u in seg_units if u.start is not None]
        ends = [u.end for u in seg_units if u.end is not None]
        return ChunkResult(
            text='\n\n'.join(lines), char_offset=seg_units[0].char_offset,
            start_sec=starts[0] if starts else None, end_sec=ends[-1] if ends else None,
            speakers=list(dict.fromkeys(u.speaker for u in seg_units)))


class HybridChunker(Chunker):
    """Структурный чанкер: CommonMark AST → greedy packing → oversized handling → post-merge.

    Три стадии:
    1. _parse_blocks  — разбор документа на типизированные блоки
    2. _greedy_fill   — жадное наполнение чанков (магнитные заголовки, oversized handling)
    3. _post_merge    — склейка мелких чанков (< min_tokens) с соседями
    """

    # Стратегии по умолчанию для каждого типа блока
    _DEFAULT_STRATEGIES = {
        'table': 'transform',
        'list': 'split',
        'paragraph': 'split',
        'fence': 'asis',
        'diagram': 'asis',
        'heading': 'asis',
    }

    def __init__(
        self,
        counter: TokenCounter,
        min_tokens: int = 50,
        max_tokens: int = 250,
        oversized_strategies: dict[str, str] | None = None,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        llm_chunker: LLMChunker | None = None,
    ) -> None:
        self._counter = counter
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._strategies = {**self._DEFAULT_STRATEGIES, **(oversized_strategies or {})}
        self._embed_fn = embed_fn
        self._llm_chunker = llm_chunker
        self._table_splitter = TableRowSplitter(counter, max_tokens)

    async def chunk(self, text: str) -> list[str]:
        result = await self._run_pipeline(text, paged=False)
        return [r.text for r in result]

    async def chunk_with_metadata(
        self, text: str, *, paged: bool = False,
    ) -> list[ChunkResult]:
        """Разбить текст на чанки с метаданными (pages, char_offset).

        Args:
            text: полный текст документа (может содержать маркеры <!-- page:N -->).
            paged: документ страничный (PDF, DOCX). Если True — извлекаем маркеры
                   и валидируем что каждый чанк получил pages. Если False — маркеры
                   не ищем, pages будут пустыми.

        Каждый чанк получает char_offset — позицию в оригинальном тексте документа.
        """
        return await self._run_pipeline(text, paged=paged)

    async def _run_pipeline(
        self, text: str, *, paged: bool = False,
    ) -> list[ChunkResult]:
        """Общий пайплайн: parse → greedy fill → post-merge."""
        blocks = self._parse_blocks(text, paged=paged)
        raw_chunks = await self._greedy_fill(blocks)
        chunks = self._post_merge(raw_chunks)

        if paged:
            missing = sum(1 for c in chunks if not c.pages)
            if missing:
                logger.warning(
                    'HybridChunker: paged document has %d/%d chunk(s) without pages',
                    missing, len(chunks),
                )

        logger.info(
            'HybridChunker: %d block(s) → %d raw chunk(s) → %d final chunk(s)',
            len(blocks), len(raw_chunks), len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Stage 1: Parse blocks
    # ------------------------------------------------------------------

    def _parse_blocks(self, text: str, *, paged: bool = False) -> list[_Block]:
        """Разбор документа на типизированные блоки через CommonMark AST.

        Если paged=True: маркеры <!-- page:N --> извлекаются, каждый блок получает
        pages. Блоки без маркеров наследуют последнюю известную страницу.
        Если paged=False: маркеры не ищем, pages остаются пустыми.

        Каждый блок получает char_offset — позицию в оригинальном тексте.
        """
        sections = _split_by_headers(text)
        blocks: list[_Block] = []
        current_page: int | None = None
        search_start = 0  # позиция поиска в оригинальном тексте

        for section in sections:
            section_stripped = section.strip()
            if not section_stripped:
                continue

            # Находим позицию секции в оригинальном тексте
            section_offset = text.find(section_stripped, search_start)
            if section_offset < 0:
                section_offset = search_start
            search_start = section_offset + len(section_stripped)

            source_lines = section_stripped.split('\n')
            ast_blocks = _top_level_blocks(section_stripped)

            if not ast_blocks:
                if section_stripped:
                    if paged:
                        clean, pages, current_page = self._extract_pages(
                            section_stripped, current_page,
                        )
                    else:
                        clean, pages = section_stripped, []
                    if clean.strip():
                        blocks.append(_Block(
                            text=clean.strip(),
                            block_type='paragraph',
                            tokens=self._counter.count(clean.strip()),
                            pages=pages,
                            char_offset=section_offset,
                        ))
                continue

            for token_type, start, end in ast_blocks:
                block_text = '\n'.join(source_lines[start:end]).strip()
                if not block_text:
                    continue
                # Offset блока = offset секции + offset строки внутри секции
                line_offset = sum(
                    len(source_lines[ln]) + 1 for ln in range(start)
                )
                block_offset = section_offset + line_offset

                if paged:
                    clean, pages, current_page = self._extract_pages(
                        block_text, current_page,
                    )
                else:
                    clean, pages = block_text, []
                if not clean.strip():
                    continue
                block_type = self._classify_block(token_type, clean.strip())
                blocks.append(_Block(
                    text=clean.strip(),
                    block_type=block_type,
                    tokens=self._counter.count(clean.strip()),
                    pages=pages,
                    char_offset=block_offset,
                ))

        return blocks

    @staticmethod
    def _extract_pages(
        text: str, current_page: int | None,
    ) -> tuple[str, list[int], int | None]:
        """Извлечь маркеры страниц из текста блока.

        Возвращает (очищенный текст, список страниц, обновлённый current_page).
        """
        markers = _PAGE_MARKER_RE.findall(text)
        clean = _PAGE_MARKER_RE.sub('', text)

        if markers:
            page_nums = sorted({int(m) for m in markers})
            current_page = page_nums[-1]
            return clean, page_nums, current_page

        if current_page is not None:
            return clean, [current_page], current_page

        return clean, [], current_page

    @staticmethod
    def _classify_block(token_type: str, text: str) -> str:
        """Определить тип блока по типу CommonMark-токена."""
        if token_type == 'heading_open':
            return 'heading'
        if token_type == 'table_open':
            return 'table'
        if token_type == 'fence' or token_type == 'code_block':
            m = _FENCE_OPEN_RE.match(text)
            if m:
                lang = m.group(2).strip().lower()
                if lang in _DIAGRAM_LANGS:
                    return 'diagram'
            return 'fence'
        if token_type in ('bullet_list_open', 'ordered_list_open'):
            return 'list'
        return 'paragraph'

    # ------------------------------------------------------------------
    # Stage 2: Greedy fill
    # ------------------------------------------------------------------

    async def _greedy_fill(self, blocks: list[_Block]) -> list[ChunkResult]:
        """Жадное наполнение чанков из блоков."""
        if not blocks:
            return []

        chunks: list[ChunkResult] = []
        current_parts: list[str] = []
        current_pages: list[int] = []
        current_offset: int = 0
        # Offset последнего добавленного блока — нужен для магнитного заголовка
        last_block_offset: int = 0

        for block in blocks:
            combined_tokens = self._count_combined(current_parts, block.text)

            if combined_tokens <= self._max_tokens:
                if not current_parts:
                    current_offset = block.char_offset
                current_parts.append(block.text)
                current_pages = _merge_pages(current_pages, block.pages)
                last_block_offset = block.char_offset
            elif block.tokens <= self._max_tokens:
                heading_offset = self._flush_chunk(
                    chunks, current_parts, current_pages,
                    current_offset, last_block_offset,
                )
                if heading_offset is not None:
                    current_offset = heading_offset
                else:
                    current_offset = block.char_offset
                current_parts.append(block.text)
                current_pages = list(block.pages)
                last_block_offset = block.char_offset
            else:
                heading_offset = self._flush_chunk(
                    chunks, current_parts, current_pages,
                    current_offset, last_block_offset,
                )
                oversized_chunks = await self._split_oversized(block)
                # Magnetic heading из flush_chunk (current_parts = [trailing_heading])
                # может содержать висящий заголовок — префиксируем его к первому
                # oversized-чанку, чтобы не потерять.
                if current_parts and oversized_chunks:
                    head_prefix = '\n\n'.join(current_parts)
                    first = oversized_chunks[0]
                    oversized_chunks[0] = ChunkResult(
                        text=head_prefix + '\n\n' + first.text,
                        pages=first.pages,
                        char_offset=(
                            heading_offset if heading_offset is not None
                            else first.char_offset
                        ),
                    )
                elif current_parts:
                    # Нет oversized-чанков (странно, но защитимся) — флашим heading как чанк
                    chunks.append(ChunkResult(
                        text='\n\n'.join(current_parts),
                        pages=list(current_pages),
                        char_offset=(
                            heading_offset if heading_offset is not None
                            else current_offset
                        ),
                    ))
                chunks.extend(oversized_chunks)
                current_parts = []
                current_pages = []
                current_offset = 0
                last_block_offset = 0

        if current_parts:
            chunks.append(ChunkResult(
                text='\n\n'.join(current_parts),
                pages=current_pages,
                char_offset=current_offset,
            ))

        return chunks

    def _count_combined(self, current_parts: list[str], new_text: str) -> int:
        """Подсчитать токены объединённого текста (с разделителями)."""
        if not current_parts:
            return self._counter.count(new_text)
        combined = '\n\n'.join(current_parts) + '\n\n' + new_text
        return self._counter.count(combined)

    def _flush_chunk(
        self,
        chunks: list[ChunkResult],
        parts: list[str],
        pages: list[int],
        char_offset: int,
        last_block_offset: int = 0,
    ) -> int | None:
        """Закрыть текущий чанк с проверкой магнитных заголовков.

        Выталкивает все trailing headings — они пойдут в начало следующего чанка.
        Возвращает char_offset первого вытолкнутого heading или None.
        """
        if not parts:
            return None

        # Собираем все trailing headings
        trailing: list[str] = []
        while len(parts) > 1 and self._looks_like_heading(parts[-1]):
            trailing.append(parts.pop())
        trailing.reverse()  # восстановить порядок

        if trailing:
            chunks.append(ChunkResult(
                text='\n\n'.join(parts),
                pages=list(pages),
                char_offset=char_offset,
            ))
            parts.clear()
            parts.extend(trailing)
            return last_block_offset
        else:
            chunks.append(ChunkResult(
                text='\n\n'.join(parts),
                pages=list(pages),
                char_offset=char_offset,
            ))
            parts.clear()
            return None

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        """Проверить, является ли текст Markdown-заголовком."""
        stripped = text.strip()
        return bool(stripped) and stripped.splitlines()[0].lstrip().startswith('#')

    # ------------------------------------------------------------------
    # Stage 2b: Oversized handling
    # ------------------------------------------------------------------

    async def _split_oversized(self, block: _Block) -> list[ChunkResult]:
        """Разбить oversized блок по стратегии для его типа.

        Стратегии: asis, split, embed, transform, llm.
        Все куски наследуют pages и offset блока.
        """
        strategy = self._strategies.get(block.block_type, 'asis')
        logger.info(
            'HybridChunker: oversized %s block (%d tokens), strategy=%s',
            block.block_type, block.tokens, strategy,
        )
        texts = await self._apply_oversized_strategy(block, strategy)
        return [
            ChunkResult(text=t, pages=list(block.pages), char_offset=block.char_offset)
            for t in texts
        ]

    async def _apply_oversized_strategy(
        self, block: _Block, strategy: str,
    ) -> list[str]:
        """Применить стратегию к oversized блоку.

        asis      — вернуть как есть
        split     — структурное разбиение (предложения / элементы / строки)
        embed     — SemanticChunker (embedding-based границы)
        transform — преобразовать формат + рекурсия (depth=1)
        llm       — LLM преобразует + рекурсия
        """
        if strategy == 'asis':
            return [block.text]

        if strategy == 'split':
            return await self._split_structural(block)

        if strategy == 'embed':
            if self._embed_fn is not None:
                semantic = SemanticChunker(
                    embed_fn=self._embed_fn,
                    counter=self._counter,
                    min_tokens=self._min_tokens,
                    max_tokens=self._max_tokens,
                )
                return await semantic.chunk(block.text)
            logger.warning('HybridChunker: embed strategy but no embed_fn, falling back to split')
            return await self._split_structural(block)

        if strategy == 'transform':
            transformed = self._transform_block(block)
            if transformed != block.text:
                # Рекурсия depth=1: чанкируем трансформированный текст
                return await self.chunk(transformed)
            return [block.text]

        if strategy == 'llm':
            if self._llm_chunker is not None:
                return await self._llm_chunker.chunk(block.text)
            logger.warning('HybridChunker: llm strategy but no llm_chunker, falling back to split')
            return await self._split_structural(block)

        logger.warning('HybridChunker: unknown strategy %r, returning as-is', strategy)
        return [block.text]

    async def _split_structural(self, block: _Block) -> list[str]:
        """Структурное разбиение по типу блока."""
        if block.block_type == 'list':
            return await self._split_oversized_list(block.text)
        if block.block_type == 'paragraph':
            return self._split_by_sentences(block.text)
        if block.block_type == 'table':
            return self._split_table_by_rows(block.text)
        if block.block_type == 'fence':
            return self._split_oversized_code(block.text)
        return [block.text]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Текст → предложения → жадная упаковка."""
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sent_tokens = self._counter.count(sentence)
            if current_tokens + sent_tokens > self._max_tokens and current:
                chunks.append(' '.join(current))
                current = [sentence]
                current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _split_table_by_rows(self, text: str) -> list[str]:
        """Таблица → строки с шапкой."""
        if self._table_splitter.can_split(text):
            parts = self._table_splitter.split(text)
            if len(parts) > 1:
                return parts
        return [text]

    def _transform_block(self, block: _Block) -> str:
        """Преобразовать формат блока для последующего чанкинга.

        table → key-value текст (заголовки колонок: значения ячеек)
        Остальные типы — без изменений.
        """
        if block.block_type == 'table':
            return self._transform_table_to_text(block.text)
        return block.text

    @staticmethod
    def _transform_table_to_text(text: str) -> str:
        """Конвертировать таблицу в key-value markdown.

        | Name | Age | Role |        →  **Name:** Alice
        |------|-----|------|            **Age:** 30
        | Alice| 30  | Dev  |           **Role:** Dev
        """
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return text

        # Парсим шапку: ищем первую строку с непустыми ячейками (>1 непустой)
        header_line = lines[0]
        headers = [h.strip() for h in header_line.strip('|').split('|')]

        # Если шапка пустая (merged cell / заголовок группы) — ищем реальные заголовки дальше
        non_empty = [h for h in headers if h.strip('* ')]
        if len(non_empty) <= 1:
            for candidate_line in lines[1:]:
                if re.match(r'^\s*\|[\s\-:|]+\|\s*$', candidate_line):
                    continue  # separator
                candidate_headers = [h.strip() for h in candidate_line.strip('|').split('|')]
                candidate_non_empty = [h for h in candidate_headers if h.strip('* ')]
                if len(candidate_non_empty) >= 2:
                    headers = candidate_headers
                    break

        # Пропускаем separator
        data_start = 1
        for i, line in enumerate(lines[1:], 1):
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                data_start = i + 1
                break

        # Конвертируем строки: каждая ячейка → заголовок h4 + содержимое
        result_parts: list[str] = []
        for line in lines[data_start:]:
            if not line.strip().startswith('|'):
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            # Пропускаем строки совпадающие с заголовком или полностью пустые
            cells_stripped = [c.strip('* ') for c in cells]
            if cells_stripped == [h.strip('* ') for h in headers]:
                continue
            if not any(c.strip() for c in cells):
                continue
            row_parts: list[str] = []
            for j, cell in enumerate(cells):
                if j < len(headers) and cell.strip():
                    key = headers[j].strip('* ')
                    if key:
                        row_parts.append(f'#### {key}\n\n{cell}')
            if row_parts:
                result_parts.append('\n\n'.join(row_parts))

        if not result_parts:
            return text

        return '\n\n---\n\n'.join(result_parts)

    _LIST_ITEM_RE = re.compile(r'^(\s*[-*+]|\s*\d+[.)]\s)', re.MULTILINE)

    async def _split_oversized_list(self, text: str) -> list[str]:
        """Список → элементы → жадная упаковка. Oversized элемент → стратегия paragraph."""
        # Разбиваем по элементам списка: каждый начинается с `- `, `* `, `1. ` и т.д.
        items: list[str] = []
        matches = list(self._LIST_ITEM_RE.finditer(text))

        if len(matches) <= 1:
            return [text]

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            item = text[start:end].rstrip()
            if item:
                items.append(item)

        if not items:
            return [text]

        # Жадная упаковка элементов; oversized элемент разбивается по предложениям
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for item in items:
            item_tokens = self._counter.count(item)
            if item_tokens > self._max_tokens:
                # Oversized элемент — сбросить текущий и применить стратегию paragraph
                if current:
                    chunks.append('\n'.join(current))
                    current = []
                    current_tokens = 0
                para_strategy = self._strategies.get('paragraph', 'split')
                item_block = _Block(text=item, block_type='paragraph', tokens=item_tokens)
                item_chunks = await self._apply_oversized_strategy(item_block, para_strategy)
                chunks.extend(item_chunks)
            elif current_tokens + item_tokens > self._max_tokens and current:
                chunks.append('\n'.join(current))
                current = [item]
                current_tokens = item_tokens
            else:
                current.append(item)
                current_tokens += item_tokens

        if current:
            chunks.append('\n'.join(current))

        return chunks if chunks else [text]

    def _split_oversized_code(self, text: str) -> list[str]:
        """Code fence → нарезка по строкам с re-wrap fence markers."""
        lines = text.split('\n')
        # Извлекаем открывающий и закрывающий fence
        open_fence = ''
        close_fence = '```'

        if lines and lines[0].strip().startswith('```'):
            open_fence = lines[0]
            m = _FENCE_OPEN_RE.match(lines[0])
            if m:
                close_fence = m.group(1)
            # Убираем открывающий fence
            lines = lines[1:]

        if lines and lines[-1].strip().startswith('`'):
            lines = lines[:-1]

        # Жадная упаковка строк
        fence_overhead = self._counter.count(open_fence + '\n' + close_fence)
        effective_limit = max(1, self._max_tokens - fence_overhead)

        chunks: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._counter.count(line)
            if current_tokens + line_tokens > effective_limit and current_lines:
                body = '\n'.join(current_lines)
                chunks.append(f'{open_fence}\n{body}\n{close_fence}')
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            body = '\n'.join(current_lines)
            chunks.append(f'{open_fence}\n{body}\n{close_fence}')

        return chunks if chunks else [text]


    # ------------------------------------------------------------------
    # Stage 3: Post-merge
    # ------------------------------------------------------------------

    def _post_merge(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        """Склейка мелких чанков (< min_tokens) с соседями."""
        if len(chunks) <= 1:
            return chunks

        token_counts = [self._counter.count(c.text) for c in chunks]
        merged: list[ChunkResult] = []
        merged_tokens: list[int] = []
        skip: set[int] = set()

        for i in range(len(chunks)):
            if i in skip:
                continue

            ci = chunks[i]

            if token_counts[i] >= self._min_tokens:
                merged.append(ci)
                merged_tokens.append(token_counts[i])
                continue

            # Последний чанк документа — допустим < min_tokens
            if i == len(chunks) - 1:
                if merged:
                    prev = merged[-1]
                    combined = prev.text + '\n\n' + ci.text
                    combined_tok = self._counter.count(combined)
                    if combined_tok <= self._max_tokens:
                        merged[-1] = ChunkResult(
                            text=combined,
                            pages=_merge_pages(prev.pages, ci.pages),
                            char_offset=prev.char_offset,
                        )
                        merged_tokens[-1] = combined_tok
                        continue
                merged.append(ci)
                merged_tokens.append(token_counts[i])
                continue

            # Попробовать склеить с предыдущим (но не heading — он должен идти к следующему)
            if merged and not self._looks_like_heading(ci.text):
                prev = merged[-1]
                combined_prev = prev.text + '\n\n' + ci.text
                combined_prev_tok = self._counter.count(combined_prev)
                if combined_prev_tok <= self._max_tokens:
                    merged[-1] = ChunkResult(
                        text=combined_prev,
                        pages=_merge_pages(prev.pages, ci.pages),
                        char_offset=prev.char_offset,
                    )
                    merged_tokens[-1] = combined_prev_tok
                    continue

            # Склеить со следующим — принудительно даже если > max_tokens.
            # Мелкий чанк (< min_tokens) между двумя oversized лучше приклеить,
            # чем оставить одиноким заголовком или обрывком.
            if i + 1 < len(chunks) and i + 1 not in skip:
                nxt = chunks[i + 1]
                combined_next = ci.text + '\n\n' + nxt.text
                combined_next_tok = self._counter.count(combined_next)
                merged.append(ChunkResult(
                    text=combined_next,
                    pages=_merge_pages(ci.pages, nxt.pages),
                    char_offset=ci.char_offset,
                ))
                merged_tokens.append(combined_next_tok)
                skip.add(i + 1)
                continue

            # Последний чанк, ни с кем не склеить — оставляем as-is
            merged.append(ci)
            merged_tokens.append(token_counts[i])

        return merged


# ============================================================================
# SectionChunker — иерархическая упаковка по markdown-заголовкам
# ============================================================================


def _heading_level(text: str) -> int:
    """Определить уровень markdown-заголовка (1-6). 0 если не заголовок."""
    stripped = text.lstrip()
    if not stripped.startswith('#'):
        return 0
    level = 0
    for ch in stripped:
        if ch == '#':
            level += 1
        else:
            break
    return level if 1 <= level <= 6 else 0


@dataclasses.dataclass
class _Section:
    """Узел иерархического дерева markdown-разделов.

    Уровень 0 — корневой (весь документ). 1..6 — H1..H6.
    `prefix_blocks` — блоки, идущие ПЕРЕД heading этого раздела (обычно пусты).
    Используются при декомпозиции: heading+preamble родителя перекладываются
    в prefix_blocks первого дочернего раздела, чтобы при рендеринге попасть
    в начало его текста без необходимости оборачивать в новый _Section.
    `heading_block` — блок-заголовок этого раздела (None у корня).
    `preamble_blocks` — блоки контента между заголовком и первым дочерним
    подразделом (или до конца, если детей нет).
    `children` — дочерние разделы строго меньшего уровня.
    """

    level: int
    heading_block: _Block | None
    prefix_blocks: list[_Block] = dataclasses.field(default_factory=list)
    preamble_blocks: list[_Block] = dataclasses.field(default_factory=list)
    children: list['_Section'] = dataclasses.field(default_factory=list)


class SectionChunker(HybridChunker):
    """Иерархическая упаковка по заголовкам markdown.

    Принцип: раздел markdown (H1 → H2 → H3…) — атомарная логическая единица.
    Если раздел целиком вмещается в `max_tokens` — выдаётся одним чанком.
    Иначе — рекурсивный спуск на следующий уровень заголовков. Если дочерних
    заголовков нет (или дошли до H6) — fallback на `_greedy_fill` родительского
    HybridChunker с поддержкой oversized-обработки для отдельных блоков.

    Преамбула раздела (контент до первого подзаголовка) «прилипает» к первому
    дочернему подразделу при декомпозиции — сохраняем логическую связь.
    """

    async def _run_pipeline(
        self, text: str, *, paged: bool = False,
    ) -> list[ChunkResult]:
        blocks = self._parse_blocks(text, paged=paged)
        tree = self._build_tree(blocks)
        raw_chunks = await self._pack_section(tree)
        chunks = self._post_merge(raw_chunks)

        if paged:
            missing = sum(1 for c in chunks if not c.pages)
            if missing:
                logger.warning(
                    'SectionChunker: paged document has %d/%d chunk(s) without pages',
                    missing, len(chunks),
                )

        logger.info(
            'SectionChunker: %d block(s) → %d raw chunk(s) → %d final chunk(s)',
            len(blocks), len(raw_chunks), len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Build tree
    # ------------------------------------------------------------------

    def _build_tree(self, blocks: list[_Block]) -> _Section:
        """Собрать иерархическое дерево: heading-блок → новый раздел.

        Не-heading блоки попадают в preamble_blocks текущего внутреннего раздела
        (stack top). Heading уровня N закрывает все секции уровня ≥ N.
        """
        root = _Section(level=0, heading_block=None)
        stack: list[_Section] = [root]

        for block in blocks:
            if block.block_type == 'heading':
                level = _heading_level(block.text)
                if not level:
                    # Странный heading-блок без '#' — трактуем как обычный контент
                    stack[-1].preamble_blocks.append(block)
                    continue
                # Закрываем секции равного или большего уровня
                while stack and stack[-1].level >= level:
                    stack.pop()
                new_sect = _Section(level=level, heading_block=block)
                stack[-1].children.append(new_sect)
                stack.append(new_sect)
            else:
                stack[-1].preamble_blocks.append(block)

        return root

    # ------------------------------------------------------------------
    # Pack
    # ------------------------------------------------------------------

    async def _pack_section(self, sect: _Section) -> list[ChunkResult]:
        """Упаковать раздел в чанки.

        Если весь раздел влезает в max_tokens — один чанк. Иначе рекурсивно
        упаковываем детей, приклеивая преамбулу+heading к первому ребёнку
        через его `prefix_blocks` (без создания оборачивающих _Section —
        избегаем бесконечной рекурсии).
        Если детей нет — fallback на greedy_fill.
        """
        full_text = self._section_text(sect)
        if not full_text.strip():
            return []

        tokens = self._counter.count(full_text)
        if tokens <= self._max_tokens:
            return [self._section_to_chunk(sect)]

        # Не влезает — декомпозируем
        if not sect.children:
            # Нет дочерних разделов: плоский greedy по блокам раздела
            flat_blocks = self._section_blocks_flat(sect)
            return await self._greedy_fill(flat_blocks)

        # Есть дети: собираем «родительский вклад» (heading + преамбула + уже накопленный prefix)
        parent_contribution: list[_Block] = []
        parent_contribution.extend(sect.prefix_blocks)
        if sect.heading_block:
            parent_contribution.append(sect.heading_block)
        parent_contribution.extend(sect.preamble_blocks)

        children = list(sect.children)
        if parent_contribution:
            first = children[0]
            children[0] = dataclasses.replace(
                first,
                prefix_blocks=list(parent_contribution) + list(first.prefix_blocks),
            )

        chunks: list[ChunkResult] = []
        for child in children:
            chunks.extend(await self._pack_section(child))
        return chunks

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    def _section_text(self, sect: _Section) -> str:
        """Сериализовать раздел: prefix + heading + preamble + дети."""
        parts: list[str] = []
        for b in sect.prefix_blocks:
            parts.append(b.text)
        if sect.heading_block:
            parts.append(sect.heading_block.text)
        for b in sect.preamble_blocks:
            parts.append(b.text)
        for c in sect.children:
            child_text = self._section_text(c)
            if child_text:
                parts.append(child_text)
        return '\n\n'.join(p for p in parts if p.strip())

    def _section_pages(self, sect: _Section) -> list[int]:
        pages: list[int] = []
        for b in sect.prefix_blocks:
            pages = _merge_pages(pages, b.pages)
        if sect.heading_block:
            pages = _merge_pages(pages, sect.heading_block.pages)
        for b in sect.preamble_blocks:
            pages = _merge_pages(pages, b.pages)
        for c in sect.children:
            pages = _merge_pages(pages, self._section_pages(c))
        return pages

    def _section_char_offset(self, sect: _Section) -> int:
        if sect.prefix_blocks:
            return sect.prefix_blocks[0].char_offset
        if sect.heading_block:
            return sect.heading_block.char_offset
        if sect.preamble_blocks:
            return sect.preamble_blocks[0].char_offset
        if sect.children:
            return self._section_char_offset(sect.children[0])
        return 0

    def _section_to_chunk(self, sect: _Section) -> ChunkResult:
        return ChunkResult(
            text=self._section_text(sect),
            pages=self._section_pages(sect),
            char_offset=self._section_char_offset(sect),
        )

    def _section_blocks_flat(self, sect: _Section) -> list[_Block]:
        """Все блоки раздела в порядке появления — для fallback на greedy_fill."""
        out: list[_Block] = []
        out.extend(sect.prefix_blocks)
        if sect.heading_block:
            out.append(sect.heading_block)
        out.extend(sect.preamble_blocks)
        for c in sect.children:
            out.extend(self._section_blocks_flat(c))
        return out
