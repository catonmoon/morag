from __future__ import annotations

import uuid
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from morag.sources.base import Chunk, Document

# Namespace для детерминированной генерации UUID из строкового идентификатора документа
_DOC_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')


def _doc_id_to_point_id(doc_id: str) -> str:
    """Преобразовать строковый doc_id в детерминированный UUID для Qdrant."""
    return str(uuid.uuid5(_DOC_NAMESPACE, doc_id))


def _payload_to_document(payload: dict) -> Document:
    """Восстановить Document из Qdrant payload."""
    core_keys = {'id', 'path', 'text', 'updated_at', 'source_type', 'size', 'indexed_at'}
    indexed_at_raw = payload.get('indexed_at')
    return Document(
        id=payload['id'],
        path=payload['path'],
        text=payload['text'],
        updated_at=datetime.fromisoformat(payload['updated_at']),
        source_type=payload['source_type'],
        size=payload.get('size', 0),
        indexed_at=datetime.fromisoformat(indexed_at_raw) if indexed_at_raw else None,
        payload={k: v for k, v in payload.items() if k not in core_keys},
    )


class DocRepository:
    """CRUD для коллекции документов."""

    def __init__(self, client: AsyncQdrantClient, collection: str = 'docs') -> None:
        self._client = client
        self._collection = collection

    async def get_by_id(self, doc_id: str) -> Document | None:
        """Найти документ по id. Возвращает None если не найден."""
        point_id = _doc_id_to_point_id(doc_id)
        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return _payload_to_document(results[0].payload)

    async def upsert(self, document: Document) -> None:
        """Сохранить или обновить документ. Автоматически выставляет indexed_at."""
        point_id = _doc_id_to_point_id(document.id)
        payload = {
            'id': document.id,
            'path': document.path,
            'text': document.text,
            'updated_at': document.updated_at.isoformat(),
            'source_type': document.source_type,
            'size': document.size,
            'indexed_at': datetime.now(timezone.utc).isoformat(),
            **document.payload,
        }
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector={}, payload=payload)],
        )

    async def delete(self, doc_id: str) -> None:
        """Удалить документ по id."""
        point_id = _doc_id_to_point_id(doc_id)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[point_id]),
        )


class ChunkRepository:
    """CRUD для коллекции чанков."""

    def __init__(self, client: AsyncQdrantClient, collection: str = 'chunks') -> None:
        self._client = client
        self._collection = collection

    async def get_index_status(self, doc_id: str) -> tuple[int, int] | None:
        """Вернуть (count, total) чанков для документа или None если чанков нет.

        Используется для idempotency-проверки: если count == total,
        документ проиндексирован полностью.
        """
        doc_filter = Filter(
            must=[FieldCondition(key='doc_id', match=MatchValue(value=doc_id))]
        )

        count_result = await self._client.count(
            collection_name=self._collection,
            count_filter=doc_filter,
            exact=True,
        )
        count = count_result.count

        if count == 0:
            return None

        points, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=doc_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None

        total = points[0].payload.get('total')
        if total is None:
            return None

        return count, int(total)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Удалить все чанки документа."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key='doc_id', match=MatchValue(value=doc_id))]
                )
            ),
        )

    async def upsert_batch(self, chunks: list[Chunk]) -> None:
        """Сохранить пачку чанков."""
        if not chunks:
            return

        points = []
        for chunk in chunks:
            payload = {
                'doc_id': chunk.doc_id,
                'path': chunk.path,
                'order': chunk.order,
                'total': chunk.total,
                'text': chunk.text,
                'context': chunk.context,
                'updated_at': chunk.updated_at.isoformat(),
                **chunk.payload,
            }
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.vectors if chunk.vectors else {},
                    payload=payload,
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
