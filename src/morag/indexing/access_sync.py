"""Сверка меток доступа в индексе с источником — без переэмбеддинга (ADR-0034).

Метка (поле payload `access`) живёт и в точке документа, и в каждом его чанке: поиск идёт по
чанкам, точечные выдачи — по документу. Правка прав не меняет текста, поэтому индексатор её не
видит (решает по `updated_at`), и метку переносит эта сверка: только `set_payload` /
`delete_payload`, ни одного вектора.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from morag.access import ACCESS_FIELD

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    changed: list[str] = field(default_factory=list)
    # Был открыт всем → получил метку. Такой документ мог остаться в Карте знаний (она общая на
    # все запросы) — вызывающий пересобирает Карту.
    closed: list[str] = field(default_factory=list)


async def sync_access(
    client: AsyncQdrantClient,
    docs_collection: str,
    chunks_collection: str,
    desired: dict[str, list[str]],
) -> SyncResult:
    """`desired` — doc_id → метка (пустой список = открыт). Документы вне `desired` не трогаются;
    документы из `desired`, которых нет в индексе, — тоже (их метку поставит индексация)."""
    result = SyncResult()
    if not desired:
        return result
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=docs_collection, limit=256, offset=offset,
            with_payload=['id', ACCESS_FIELD], with_vectors=False,
        )
        for pt in points:
            pl = pt.payload or {}
            doc_id = pl.get('id')
            if doc_id not in desired:
                continue
            current = sorted({str(x) for x in (pl.get(ACCESS_FIELD) or [])})
            want = sorted(set(desired[doc_id]))
            if current == want:
                continue
            for coll, key in ((chunks_collection, 'doc_id'), (docs_collection, 'id')):
                flt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=doc_id))])
                if want:
                    await client.set_payload(collection_name=coll, payload={ACCESS_FIELD: want},
                                             points=flt)
                else:
                    await client.delete_payload(collection_name=coll, keys=[ACCESS_FIELD],
                                                points=flt)
            result.changed.append(doc_id)
            if want and not current:
                result.closed.append(doc_id)
        if offset is None:
            break
    if result.changed:
        logger.info('[access] sync: %d label(s) changed, %d closed', len(result.changed),
                    len(result.closed))
    return result
