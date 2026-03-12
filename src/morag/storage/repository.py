from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
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
    core_keys = {'id', 'path', 'text', 'updated_at', 'source_type', 'size', 'url', 'indexed_at', 'creator', 'created_at', 'parent_doc_ids', 'structural'}
    indexed_at_raw = payload.get('indexed_at')
    created_at_raw = payload.get('created_at')
    # backwards compat: path мог быть строкой в старых данных
    path_raw = payload['path']
    path = path_raw if isinstance(path_raw, list) else [path_raw]
    return Document(
        id=payload['id'],
        path=path,
        text=payload['text'],
        updated_at=datetime.fromisoformat(payload['updated_at']),
        source_type=payload['source_type'],
        size=payload.get('size', 0),
        url=payload.get('url'),
        indexed_at=datetime.fromisoformat(indexed_at_raw) if indexed_at_raw else None,
        creator=payload.get('creator'),
        created_at=datetime.fromisoformat(created_at_raw) if created_at_raw else None,
        parent_doc_ids=payload.get('parent_doc_ids', []),
        structural=payload.get('structural', False),
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
            'parent_doc_ids': document.parent_doc_ids,
            'structural': document.structural,
            **document.payload,
        }
        if document.url is not None:
            payload['url'] = document.url
        if document.creator is not None:
            payload['creator'] = document.creator
        if document.created_at is not None:
            payload['created_at'] = document.created_at.isoformat()
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

    async def find_children(self, parent_doc_id: str) -> list[Document]:
        """Найти все документы, у которых parent_doc_id есть в parent_doc_ids."""
        scroll_filter = Filter(
            must=[FieldCondition(key='parent_doc_ids', match=MatchValue(value=parent_doc_id))]
        )
        docs: list[Document] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    docs.append(_payload_to_document(point.payload))
            if offset is None:
                break
        return docs

    async def update_parent_doc_ids(self, doc_id: str, parent_doc_ids: list[str]) -> None:
        """Обновить список родительских документов."""
        point_id = _doc_id_to_point_id(doc_id)
        await self._client.set_payload(
            collection_name=self._collection,
            payload={'parent_doc_ids': parent_doc_ids},
            points=[point_id],
        )

    async def delete_attached(self, doc_id: str, chunk_repo: 'ChunkRepository') -> None:
        """Удалить attached-детей документа (jira, pdf и пр.), не трогая дочерние страницы.

        Удаляет только документы с source_type, начинающимся на 'attached_'.
        Дочерние Confluence-страницы (source_type='confluence') не затрагиваются.
        """
        children = await self.find_children(doc_id)
        for child in children:
            if not child.source_type.startswith('attached_'):
                continue
            await chunk_repo.delete_by_doc_id(child.id)
            await self.delete(child.id)
            logger.info('Deleted attached child: %s (source_type=%s)', child.id, child.source_type)

    async def cascade_delete(self, doc_id: str, chunk_repo: 'ChunkRepository') -> None:
        """Каскадное удаление документа.

        Сначала рекурсивно обрабатывает дочерние документы (по parent_doc_ids):
        - если у ребёнка остаются другие родители — убирает только ссылку на удаляемый doc_id
        - если parent_doc_ids ребёнка становится пустым — каскадно удаляет и ребёнка
        Затем удаляет чанки и сам документ.
        """
        children = await self.find_children(doc_id)
        for child in children:
            new_parent_ids = [p for p in child.parent_doc_ids if p != doc_id]
            if new_parent_ids:
                await self.update_parent_doc_ids(child.id, new_parent_ids)
                logger.info('Removed parent reference %s from child %s', doc_id, child.id)
            else:
                await self.cascade_delete(child.id, chunk_repo)

        await chunk_repo.delete_by_doc_id(doc_id)
        await self.delete(doc_id)
        logger.info('Cascade deleted: %s', doc_id)

    async def get_ids_by_source_type(self, source_type: str) -> set[str]:
        """Вернуть множество doc_id всех документов с указанным source_type."""
        scroll_filter = Filter(
            must=[FieldCondition(key='source_type', match=MatchValue(value=source_type))]
        )
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=['id'],
                with_vectors=False,
            )
            for point in points:
                if point.payload and 'id' in point.payload:
                    ids.add(point.payload['id'])
            if offset is None:
                break
        return ids

    async def scroll_all(self, exclude_source_types: list[str] | None = None) -> list[Document]:
        """Вернуть все документы. Опционально исключить по source_type."""
        scroll_filter = None
        if exclude_source_types:
            scroll_filter = Filter(
                must_not=[FieldCondition(key='source_type', match=MatchAny(any=exclude_source_types))]
            )

        docs: list[Document] = []
        offset = None

        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    docs.append(_payload_to_document(point.payload))
            if offset is None:
                break

        return docs


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
                'source_type': chunk.source_type,
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
