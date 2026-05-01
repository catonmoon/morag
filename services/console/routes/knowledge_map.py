"""GET /api/knowledge-map — карта в виде markdown.
POST /api/knowledge-map/rebuild — перестроить KM без полной индексации.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from morag.config import load_config
from services.console.indexer_client import AlreadyRunning, IndexerError

logger = logging.getLogger(__name__)
router = APIRouter()


class KMNode(BaseModel):
    doc_id: str
    map_text: str


class KMResponse(BaseModel):
    nodes: list[KMNode]
    full_text: str  # все nodes склеены через --- разделитель — то что агент видит в system prompt
    qdrant_reachable: bool
    error: str | None = None


@router.get('', response_model=KMResponse)
async def get_knowledge_map(request: Request) -> KMResponse:
    """Прочитать knowledge_map collection из Qdrant.

    Возвращает все map-точки (отфильтровывает служебные `_cluster_membership`).
    Если коллекции нет / Qdrant недоступен — пустой ответ с error-полем,
    UI показывает соответствующий placeholder.
    """
    cfg_path = request.app.state.config_path
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Cannot load config: {e}') from e

    km_collection = (
        cfg.indexing.knowledge_map.collection
        if cfg.indexing and cfg.indexing.knowledge_map.enabled
        else 'knowledge_map'
    )

    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=10)
    try:
        try:
            cols = {c.name for c in (await client.get_collections()).collections}
        except Exception as e:
            return KMResponse(
                nodes=[], full_text='', qdrant_reachable=False,
                error=f'Qdrant unreachable: {e}',
            )

        if km_collection not in cols:
            return KMResponse(
                nodes=[], full_text='', qdrant_reachable=True,
                error=f'Collection «{km_collection}» does not exist yet — run indexing first.',
            )

        # Сгребаем все точки коллекции — их обычно мало (по числу root-секций).
        nodes: list[KMNode] = []
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=km_collection,
                limit=200,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                # Служебные точки flat_topics (без map_text) пропускаем.
                if 'map_text' not in payload:
                    continue
                nodes.append(KMNode(
                    doc_id=str(payload.get('doc_id', '')),
                    map_text=payload['map_text'],
                ))
            if offset is None:
                break

        full_text = '\n\n---\n\n'.join(n.map_text for n in nodes) if nodes else ''
        return KMResponse(
            nodes=nodes,
            full_text=full_text,
            qdrant_reachable=True,
            error=None if nodes else f'Collection «{km_collection}» is empty.',
        )
    finally:
        await client.close()


@router.post('/rebuild')
async def rebuild_km(request: Request) -> dict[str, Any]:
    try:
        return await request.app.state.indexer.start_rebuild_km()
    except AlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except IndexerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
