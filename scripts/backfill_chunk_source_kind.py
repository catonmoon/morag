"""Backfill source_kind/source_name на чанках без переэмбеддинга.

После ADR-0012 в payload документов появились source_kind/source_name. Чанки
их НЕ унаследовали (исторический баг — см. memory/chunks-payload-source-kind.md),
что ломает retrieval-фильтры по kind на коллекции chunks.

Этот скрипт:
1. Сканирует docs collection, строит карту {doc_id: (source_kind, source_name)}.
2. Группирует doc_id по парам (kind, name).
3. Для каждой группы делает Qdrant set_payload с фильтром по doc_id —
   обновляет только payload, векторы не трогает.

Идемпотентный: повторный запуск — no-op (payload уже выставлен).
Безопасно во время индексации: set_payload не блокирует upsert'ы.

Запуск:
    python -m scripts.backfill_chunk_source_kind --config conf/config.yml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from morag.config import load_config

logger = logging.getLogger(__name__)

# Размер chunk'а doc_id'ов в одном set_payload вызове — Qdrant MatchAny не
# любит огромные списки. 500 — компромисс между числом запросов и payload size.
_DOC_ID_BATCH = 500


async def backfill(qdrant_url: str, docs_collection: str, chunks_collection: str) -> None:
    qd = AsyncQdrantClient(url=qdrant_url, timeout=120)

    # 1. Карта doc_id → (source_kind, source_name) из docs collection.
    logger.info('Scrolling docs collection %r...', docs_collection)
    doc_map: dict[str, tuple[str, str]] = {}
    skipped_no_kind = 0
    offset = None
    while True:
        points, offset = await qd.scroll(
            collection_name=docs_collection,
            with_payload=['id', 'source_kind', 'source_name'],
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            did = pl.get('id')
            kind = pl.get('source_kind')
            name = pl.get('source_name')
            if not did or not kind:
                skipped_no_kind += 1
                continue
            doc_map[did] = (kind, name or '')
        if offset is None:
            break
    logger.info(
        'Built doc map: %d entries (skipped %d docs without source_kind)',
        len(doc_map), skipped_no_kind,
    )

    # 2. Группируем doc_id по (kind, name).
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for did, (kind, name) in doc_map.items():
        groups[(kind, name)].append(did)
    logger.info('Groups: %s', {k: len(v) for k, v in groups.items()})

    # 3. Для каждой группы делаем set_payload батчами doc_id.
    for (kind, name), doc_ids in groups.items():
        payload = {'source_kind': kind}
        if name:
            payload['source_name'] = name
        total = len(doc_ids)
        updated_pts_estimate = 0
        for i in range(0, total, _DOC_ID_BATCH):
            batch = doc_ids[i:i + _DOC_ID_BATCH]
            await qd.set_payload(
                collection_name=chunks_collection,
                payload=payload,
                points=Filter(must=[
                    FieldCondition(key='doc_id', match=MatchAny(any=batch)),
                ]),
                wait=False,
            )
            updated_pts_estimate += len(batch)
            logger.info(
                'set_payload kind=%s name=%s doc-batch %d/%d',
                kind, name, min(i + _DOC_ID_BATCH, total), total,
            )
        logger.info(
            'Group (%s/%s): payload set on chunks of %d docs',
            kind, name, total,
        )

    await qd.close()
    logger.info('Backfill complete.')


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/config.yml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    qd_url = f'http://{cfg.qdrant.host}:{cfg.qdrant.port}'
    asyncio.run(backfill(
        qd_url,
        docs_collection=cfg.qdrant.collection_docs,
        chunks_collection=cfg.qdrant.collection_chunks,
    ))


if __name__ == '__main__':
    main()
