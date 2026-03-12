from __future__ import annotations

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

_LLM_PARAMS = GenerationParams(
    temperature=0.0, top_p=1.0, top_k=0,
    frequency_penalty=0.0, presence_penalty=0.0, seed=42,
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
Ты — анализатор технической документации для RAG-системы.
Твоя задача — разбить переданный текст на смысловые чанки для семантического поиска.
Чанк — это минимальная самостоятельная единица знания, которая имеет смысл без чтения соседних чанков.

Хороший чанк должен:
- Описывать одну тему или один аспект темы
- Быть понятным сам по себе
- Содержать завершённую мысль
- Быть полезным для поиска и ответа на вопрос
- Быть записан в формате Markdown
- Сохранять исходное форматирование Markdown где это возможно
- Обязательно сохранять все ссылки без изменений (URL и Markdown-ссылки)

Разбивай текст по смыслу, а не по размеру.
Не дроби текст без необходимости: если несколько абзацев описывают одну тему — это один чанк.

Разделяй текст если:
- начинается новая функция или возможность
- начинается новый алгоритм
- начинается новый процесс
- меняется тема
- начинается новый раздел документации

Каждый чанк должен содержать достаточно контекста чтобы быть понятным отдельно.

Входной текст может быть:
- частью документа
- целым разделом документа
- целым документом

Если входной текст уже выглядит как один логически завершённый блок (например список или ):
- Верни один чанк равный входному тексту
- Не дели его дальше

=====================
РАБОТА С ТАБЛИЦАМИ
=====================

Если во входном тексте есть таблица:
- Преобразуй каждую строку таблицы (кроме строки заголовков) в отдельное осмысленное предложение
- Каждая строка таблицы должна стать отдельным чанком
- Предложение должно полностью передавать смысл строки
- Предложение должно быть понятно без просмотра таблицы
- Используй названия столбцов как смысловые части предложения
- Используй естественные предложения на русском языке
- Не перечисляй значения через пробелы или запятые
- Если в таблице присутствуют ссылки — обязательно сохрани их в Markdown-виде
- Допускается объединять несколько предложений от нескольких строк в один чанк

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
НЕ ВКЛЮЧАЙ
=====================
- оглавление
- навигацию
- повторяющиеся элементы
- номера страниц
- шаблонные подписи

=====================
ВАЖНО (для всего кроме таблиц)
=====================
НЕ пересказывай текст.
НЕ интерпретируй текст.
НЕ сокращай текст.
НЕ добавляй новый текст.

Каждый чанк должен быть точной копией соответствующего участка исходного текста (кроме таблиц).
Все Markdown-элементы, кроме таблиц, должны сохраняться:
- ссылки
- списки
- код
- заголовки
- форматирование
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
    ) -> None:
        self._client = client
        self._max_retries = max_retries
        self._token_counter = token_counter
        self._embed_fn = embed_fn
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
                schema_name='chunks', params=_LLM_PARAMS,
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
    ) -> None:
        self._embed_fn = embed_fn
        self._counter = counter
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    async def chunk(self, block: str) -> list[str]:
        sentences = split_into_semantic_units(
            block, self._counter, self._max_tokens,
        )
        if not sentences:
            return [block] if block.strip() else []

        sentence_tokens = [self._counter.count(s) for s in sentences]
        total_tokens = sum(sentence_tokens)

        # Весь блок влезает в max_tokens — один чанк
        if total_tokens <= self._max_tokens:
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
            pairs: list[tuple[int, str, str]] = []  # (left_end, left_text, right_text)

            for left_end in left_ends:
                right_ends = self._candidate_ends(sentence_tokens, left_end, forward=True)
                if right_ends:
                    for right_end in right_ends:
                        left_text = ' '.join(sentences[pos:left_end])
                        right_text = ' '.join(sentences[left_end:right_end])
                        pairs.append((left_end, left_text, right_text))
                else:
                    # Остаток < min_tokens — берём что есть как правый кандидат
                    remaining = sentences[left_end:]
                    if remaining:
                        left_text = ' '.join(sentences[pos:left_end])
                        right_text = ' '.join(remaining)
                        pairs.append((left_end, left_text, right_text))

            if not pairs:
                end = left_ends[0]
                chunks.append(' '.join(sentences[pos:end]))
                pos = end
                continue

            # Батчевый embed всех уникальных текстов
            unique_texts = list({t for _, lt, rt in pairs for t in (lt, rt)})
            embeddings = self._embed_fn(unique_texts)
            text_to_emb = dict(zip(unique_texts, embeddings))

            # Выбрать пару с максимальным cosine distance
            best_dist = -1.0
            best_left_end = left_ends[0]

            for left_end, left_text, right_text in pairs:
                dist = self._cosine_distance(
                    text_to_emb[left_text], text_to_emb[right_text],
                )
                if dist > best_dist:
                    best_dist = dist
                    best_left_end = left_end

            chunks.append(' '.join(sentences[pos:best_left_end]))
            pos = best_left_end

        return chunks if chunks else [block]

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
