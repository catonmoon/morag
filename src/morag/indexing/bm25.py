"""BM25 sparse vector builder.

Вычисляет BM25Okapi веса по всему корпусу чанков и записывает
sparse vectors в Qdrant. Запускается как post-indexing шаг,
когда все чанки уже в коллекции.

Поддерживает несколько BM25 представлений:
- bm25: стемминг (морфология)
- bm25_phonetic: фонетическая нормализация (Russian Metaphone + триграммы)
- bm25_translit: транслитерация кириллица↔латиница
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointVectors

from morag.indexing.embedder import _MD5_MOD

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r'\w+')
_CYRILLIC_RE = re.compile(r'[а-яё]')

_STOP_WORDS: frozenset[str] = frozenset(
    stopwords.words('russian') + stopwords.words('english')
)

_stemmer_ru = SnowballStemmer('russian')
_stemmer_en = SnowballStemmer('english')

# ── Триграммы ─────────────────────────────────────────────────────────────

def _trigrams(word: str) -> list[str]:
    """Символьные триграммы слова (с padding)."""
    padded = f'__{word}__'
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


# ── Токенизаторы ──────────────────────────────────────────────────────────

def _stem(word: str) -> str:
    """Стемминг с автоопределением языка по кириллице."""
    if _CYRILLIC_RE.search(word):
        return _stemmer_ru.stem(word)
    return _stemmer_en.stem(word)


def _word_to_index(word: str) -> int:
    """Хэш токена → индекс sparse-вектора. Совместимо с GTE sparse."""
    return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD


def tokenize(text: str) -> list[str]:
    """Токенизация: lowercase + стоп-слова + Snowball stemming (ru/en)."""
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def tokenize_trigram(text: str) -> list[str]:
    """Токенизация: символьные триграммы оригинальных слов (без стемминга).

    Ловит опечатки через пересечение триграмм:
    адаптация/адоптация — 70-80% триграмм совпадают.
    """
    tokens = []
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP_WORDS:
            continue
        for tri in _trigrams(w):
            tokens.append(tri)
    return tokens


def build_bm25_vectors(
    texts: list[str],
    tokenizer=tokenize,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """Построить BM25 sparse vectors для корпуса текстов.

    Args:
        texts: корпус текстов
        tokenizer: функция text → list[str] (tokenize, tokenize_phonetic, tokenize_translit)
        k1, b: параметры BM25

    Возвращает список {'indices': [...], 'values': [...]}.
    """
    # Токенизация
    docs = [tokenizer(t) for t in texts]
    n = len(docs)
    if n == 0:
        return []

    # Средняя длина документа
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / n if n > 0 else 1

    # Document frequency: в скольких документах встречается терм
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    # IDF (BM25 variant)
    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    # BM25 score per document
    vectors = []
    for doc, dl in zip(docs, doc_lens):
        tf = Counter(doc)
        index_weight: dict[int, float] = {}
        for term, count in tf.items():
            score = idf[term] * (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avgdl))
            idx = _word_to_index(term)
            if idx in index_weight:
                index_weight[idx] = max(index_weight[idx], score)
            else:
                index_weight[idx] = score
        vectors.append({
            'indices': list(index_weight.keys()),
            'values': list(index_weight.values()),
        })

    return vectors


# Все BM25 представления: (имя вектора, токенизатор)
BM25_VARIANTS: list[tuple[str, callable]] = [
    ('bm25', tokenize),
    ('bm25_trigram', tokenize_trigram),
]


async def build_bm25_index(
    client: AsyncQdrantClient,
    collection: str = 'chunks',
    batch_size: int = 64,
) -> None:
    """Post-indexing: построить BM25 sparse vectors для всех чанков в коллекции.

    Строит только те варианты из BM25_VARIANTS, которые есть в схеме коллекции.
    """
    # Определить какие BM25 вектора есть в схеме
    info = await client.get_collection(collection)
    available_sparse = set()
    if info.config.params.sparse_vectors:
        available_sparse = set(info.config.params.sparse_vectors.keys())

    variants_to_build = [
        (name, tok) for name, tok in BM25_VARIANTS if name in available_sparse
    ]
    if not variants_to_build:
        logger.warning('BM25: no BM25 sparse vectors in collection schema, skipping')
        return

    logger.info(
        'BM25: will build %d variants: %s',
        len(variants_to_build), [v[0] for v in variants_to_build],
    )
    logger.info('BM25: loading all chunks from %s...', collection)

    # Scroll all chunks (только с vectors — пропускаем битые точки)
    all_points: list[tuple[str | int, str]] = []  # (point_id, text)
    skipped = 0
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=['text', 'doc_summary'],
            with_vectors=['full'],
        )
        if not points:
            break
        for p in points:
            if not p.vector or not p.vector.get('full'):
                skipped += 1
                continue
            text = p.payload.get('text', '')
            doc_summary = p.payload.get('doc_summary', '')
            combined = f'{text}\n{doc_summary}' if doc_summary else text
            all_points.append((p.id, combined))
        if offset is None:
            break
    if skipped:
        logger.warning('BM25: skipped %d chunks without vectors', skipped)

    if not all_points:
        logger.info('BM25: no chunks found, skipping')
        return

    ids = [pid for pid, _ in all_points]
    texts = [text for _, text in all_points]

    for vector_name, tokenizer in variants_to_build:
        logger.info('BM25 [%s]: building vectors for %d chunks...', vector_name, len(ids))
        vectors = build_bm25_vectors(texts, tokenizer=tokenizer)

        # Update in batches
        total_batches = (len(ids) + batch_size - 1) // batch_size
        skipped_empty = 0
        for i in range(total_batches):
            start = i * batch_size
            end = min(start + batch_size, len(ids))
            batch_points = []
            for j in range(start, end):
                if vectors[j]['indices']:
                    batch_points.append(
                        PointVectors(id=ids[j], vector={vector_name: vectors[j]})
                    )
                else:
                    skipped_empty += 1
            if batch_points:
                await client.update_vectors(
                    collection_name=collection,
                    points=batch_points,
                )
            if (i + 1) % 10 == 0 or i + 1 == total_batches:
                logger.info(
                    'BM25 [%s]: batch %d/%d', vector_name, i + 1, total_batches,
                )
        if skipped_empty:
            logger.warning(
                'BM25 [%s]: skipped %d chunks with empty vectors',
                vector_name, skipped_empty,
            )
        logger.info('BM25 [%s]: done.', vector_name)

    logger.info('BM25: all variants built for %d chunks.', len(ids))
