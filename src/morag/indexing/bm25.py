"""BM25 sparse vector builder.

Вычисляет BM25Okapi веса по всему корпусу чанков и записывает
sparse vector 'bm25' в Qdrant. Запускается как post-indexing шаг,
когда все чанки уже в коллекции.
"""
from __future__ import annotations

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


def _stem(word: str) -> str:
    """Стемминг с автоопределением языка по кириллице."""
    if _CYRILLIC_RE.search(word):
        return _stemmer_ru.stem(word)
    return _stemmer_en.stem(word)


def _word_to_index(word: str) -> int:
    """Хэш токена → индекс sparse-вектора. Совместимо с GTE sparse."""
    import hashlib
    return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD


def tokenize(text: str) -> list[str]:
    """Токенизация: lowercase + стоп-слова + Snowball stemming (ru/en)."""
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def build_bm25_vectors(
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """Построить BM25 sparse vectors для корпуса текстов.

    Возвращает список {'indices': [...], 'values': [...]}.
    """
    # Токенизация
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    if n == 0:
        return []

    # Средняя длина документа
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / n

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


async def build_bm25_index(
    client: AsyncQdrantClient,
    collection: str = 'chunks',
    batch_size: int = 64,
) -> None:
    """Post-indexing: построить BM25 sparse vectors для всех чанков в коллекции."""
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

    logger.info('BM25: building vectors for %d chunks...', len(all_points))
    ids = [pid for pid, _ in all_points]
    texts = [text for _, text in all_points]
    vectors = build_bm25_vectors(texts)

    # Update in batches, пропуская пустые vectors (стоп-слова, пустой текст)
    total_batches = (len(ids) + batch_size - 1) // batch_size
    skipped_empty = 0
    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, len(ids))
        batch_points = []
        for j in range(start, end):
            if vectors[j]['indices']:
                batch_points.append(PointVectors(id=ids[j], vector={'bm25': vectors[j]}))
            else:
                skipped_empty += 1
        if batch_points:
            await client.update_vectors(
                collection_name=collection,
                points=batch_points,
            )
        logger.info('BM25: batch %d/%d (%d chunks)', i + 1, total_batches, end - start)
    if skipped_empty:
        logger.warning('BM25: skipped %d chunks with empty vectors (stop-words only)', skipped_empty)

    logger.info('BM25: done. Updated %d vectors.', len(ids))
