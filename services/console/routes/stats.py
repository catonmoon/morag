"""GET /api/stats и /api/links — счётчики из Qdrant + ссылки на смежные UI."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from morag.config import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


class StatsResponse(BaseModel):
    docs: int
    chunks: int
    knowledge_map_nodes: int
    collections: list[str]
    qdrant_reachable: bool
    qdrant_error: str | None = None
    # ISO timestamp самого свежего документа в коллекции docs (max по updated_at).
    # null если коллекции нет / docs пустые / Qdrant не дал отсортировать.
    last_indexed_at: str | None = None


class LinksResponse(BaseModel):
    qdrant: str
    open_webui: str
    # Параметры для подключения «внешнего» OWUI (не из нашего docker-compose)
    # к нашему pipelines-сервису. Внутренний OWUI уже подключён через env
    # (см. docker-compose.yml: OPENAI_API_BASE_URL=http://pipelines:9099).
    external_owui: 'ExternalOwuiConnection'


class ExternalOwuiConnection(BaseModel):
    base_url: str
    model: str
    api_key: str


@router.get('/stats', response_model=StatsResponse)
async def get_stats(request: Request) -> StatsResponse:
    """Счётчики из Qdrant. Если Qdrant недоступен — все нули + qdrant_reachable=False.

    Не бросаем 500 — на freshly-deployed системе (config указывает на докер-имя
    qdrant, но консоль запущена локально с другим хостом, или Qdrant ещё не поднят)
    UI должен оставаться рабочим, чтобы юзер мог поправить host в настройках.
    """
    cfg_path = request.app.state.config_path
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Cannot load config: {e}') from e

    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=5)
    try:
        try:
            cols = await client.get_collections()
        except Exception as e:
            logger.warning('Qdrant unreachable at %s:%d — %s', cfg.qdrant.host, cfg.qdrant.port, e)
            return StatsResponse(
                docs=0, chunks=0, knowledge_map_nodes=0, collections=[],
                qdrant_reachable=False,
                qdrant_error=f'{cfg.qdrant.host}:{cfg.qdrant.port} — {type(e).__name__}',
            )

        col_names = [c.name for c in cols.collections]
        docs = await _safe_count(client, cfg.qdrant.collection_docs, col_names)
        chunks = await _safe_count(client, cfg.qdrant.collection_chunks, col_names)
        km_collection = (
            cfg.indexing.knowledge_map.collection
            if cfg.indexing and cfg.indexing.knowledge_map.enabled
            else 'knowledge_map'
        )
        km_nodes = await _safe_count(client, km_collection, col_names)
        last_indexed_at = await _last_indexed_at(client, cfg.qdrant.collection_docs, col_names)

        return StatsResponse(
            docs=docs,
            chunks=chunks,
            knowledge_map_nodes=km_nodes,
            collections=col_names,
            qdrant_reachable=True,
            last_indexed_at=last_indexed_at,
        )
    finally:
        await client.close()


async def _last_indexed_at(client, collection: str, available: list[str]) -> str | None:
    """Найти max(updated_at) в коллекции docs.

    Использует scroll(order_by=...) — требует datetime payload-индекс на updated_at.
    Если индекса нет (старые установки) — создаём на лету (one-shot миграция,
    идемпотентно). При следующих ошибках возвращаем None — пусть UI просто
    скроет поле, не падаем.
    """
    if collection not in available:
        return None
    try:
        from qdrant_client.models import Direction, OrderBy, PayloadSchemaType

        for attempt in range(2):
            try:
                points, _ = await client.scroll(
                    collection_name=collection,
                    limit=1,
                    order_by=OrderBy(key='updated_at', direction=Direction.DESC),
                    with_payload=['updated_at'],
                )
                if not points:
                    return None
                return points[0].payload.get('updated_at')
            except Exception as e:
                # Первый промах — пробуем создать индекс. Второй — сдаёмся.
                if attempt == 0 and 'index' in str(e).lower():
                    logger.info('Creating datetime index on %s.updated_at', collection)
                    await client.create_payload_index(
                        collection_name=collection,
                        field_name='updated_at',
                        field_schema=PayloadSchemaType.DATETIME,
                    )
                    continue
                logger.warning('Cannot order docs by updated_at: %s', e)
                return None
    except Exception as e:
        logger.warning('last_indexed_at failed: %s', e)
        return None
    return None


async def _safe_count(client, collection: str, available: list[str]) -> int:
    """Если коллекции нет — возвращаем 0 а не 404."""
    if collection not in available:
        return 0
    try:
        result = await client.count(collection_name=collection, exact=False)
        return result.count
    except Exception:
        logger.warning('Failed to count %s', collection, exc_info=True)
        return 0


@router.get('/links', response_model=LinksResponse)
async def get_links(request: Request) -> LinksResponse:
    """Внешние ссылки в смежные сервисы.

    Дефолты — для нашего docker-compose (qdrant на 6333, OWUI на 3000,
    pipelines на 9099). Env-переменные могут переопределить.
    """
    import os
    qdrant_url = os.environ.get('QDRANT_DASHBOARD_URL', 'http://localhost:6333/dashboard')
    owui_url = os.environ.get('OPENWEBUI_URL', 'http://localhost:3000')
    pipelines_url = os.environ.get('PIPELINES_PUBLIC_URL', 'http://localhost:9099')
    api_key = os.environ.get('PIPELINES_API_KEY', '0p3n-w3bu!')
    return LinksResponse(
        qdrant=qdrant_url,
        open_webui=owui_url,
        external_owui=ExternalOwuiConnection(
            base_url=pipelines_url,
            model='morag_pipeline',
            api_key=api_key,
        ),
    )
