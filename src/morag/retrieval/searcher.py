"""HybridSearcher — фасад вокруг Qdrant для retrieval.

Объединяет RRF-поиск (dense + sparse + BM25 + BM25-trigram) и набор fetch-хелперов
(чанки по order, summary документов, дерево parent→children, knowledge map,
cluster membership, title). Кеширует метаданные.

OWUI/pipelines-независимо — чистый async API, возвращает dict-ы универсального
shape'а. Потребители: retrieval-pipeline (`services/pipeline/morag_pipeline.py`),
CLI-скрипты, тесты.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

from morag.indexing.bm25 import tokenize, tokenize_trigram, to_sparse_vector
from morag.indexing.embedder import Embedder, SparseEmbedder

logger = logging.getLogger(__name__)


def _point_to_chunk(p: Any) -> dict[str, Any]:
    """Конвертер Qdrant-point (ScoredPoint/Record/dict) → наш chunk-dict."""
    if isinstance(p, dict):
        point_id = p.get('id')
        payload = p.get('payload') or {}
        score = p.get('score', 0.0)
    else:
        point_id = p.id
        payload = p.payload or {}
        score = float(p.score) if getattr(p, 'score', None) is not None else 0.0
    path_raw = payload.get('path', '')
    paths: list[str] = path_raw if isinstance(path_raw, list) else ([path_raw] if path_raw else [])
    return {
        'chunk_id': str(point_id),
        'doc_id': payload.get('doc_id', ''),
        'path': paths,
        'order': payload.get('order', 0),
        'total': payload.get('total', 0),
        'text': payload.get('text', ''),
        'context': payload.get('context', ''),
        'updated_at': payload.get('updated_at', ''),
        'creator': payload.get('creator', ''),
        'url': payload.get('url'),
        'source_type': payload.get('source_type', ''),
        'score': score,
    }


def _bm25_query_vector(text: str) -> tuple[list[int], list[float]]:
    return to_sparse_vector(tokenize(text))


def _bm25_trigram_query_vector(text: str) -> tuple[list[int], list[float]]:
    return to_sparse_vector(tokenize_trigram(text))


class HybridSearcher:
    """Qdrant-фасад: RRF-поиск по chunks/docs + fetch хелперы.

    Все методы async. Кеширует метаданные (sparse vector names per-collection,
    doc tree, indexed doc ids, doc titles, knowledge map, cluster membership).

    Instance безопасен для параллельных вызовов (внутренние кеши — idempotent
    populate, race-condition безвредна — оба потока получают одинаковый результат).
    """

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        dense_embedder: Embedder,
        sparse_embedder: SparseEmbedder,
        chunks_collection: str,
        docs_collection: str,
        knowledge_map_collection: str = 'knowledge_map',
    ) -> None:
        self._qdrant = qdrant
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._chunks_collection = chunks_collection
        self._docs_collection = docs_collection
        self._km_collection = knowledge_map_collection
        self._sparse_vector_names_cache: dict[str, set[str]] = {}
        self._doc_tree: dict[str, list[str]] | None = None
        self._indexed_doc_ids: set[str] | None = None
        self._doc_titles: dict[str, str] = {}
        self._cluster_membership: dict[str, list[str]] | None = None
        self._knowledge_map: str | None = None

    # ── Schema helpers ────────────────────────────────────────────────────────

    async def get_sparse_vector_names(self, collection: str) -> set[str]:
        """Имена sparse-векторов коллекции (кеш per-collection)."""
        if collection in self._sparse_vector_names_cache:
            return self._sparse_vector_names_cache[collection]
        names: set[str] = set()
        try:
            info = await self._qdrant.get_collection(collection)
            sparse_params = info.config.params.sparse_vectors or {}
            names = set(sparse_params.keys())
        except Exception as exc:
            logger.warning('HybridSearcher: sparse vector names for %s failed: %s', collection, exc)
        self._sparse_vector_names_cache[collection] = names
        return names

    # ── Search (RRF) ──────────────────────────────────────────────────────────

    async def _build_rrf_prefetch(self, collection: str, text: str, limit: int) -> list[Prefetch]:
        """Двухуровневый RRF: dense `full` (1 голос) vs nested-RRF по sparse (1 голос).

        Sparse-каналы: `keywords` (GTE), `bm25` (Snowball stem), `bm25_trigram`
        (символьные триграммы). Используются только существующие в схеме коллекции.
        """
        dense = await self._dense.embed_query(text)
        indices, values = await self._sparse.embed_query(text)
        available_sparse = await self.get_sparse_vector_names(collection)

        lexical: list[Prefetch] = []
        if 'keywords' in available_sparse:
            lexical.append(Prefetch(
                query=SparseVector(indices=indices, values=values),
                using='keywords',
                limit=limit * 2,
            ))
        for vec_fn, vec_name in [
            (_bm25_query_vector, 'bm25'),
            (_bm25_trigram_query_vector, 'bm25_trigram'),
        ]:
            if vec_name not in available_sparse:
                continue
            idx, val = vec_fn(text)
            if idx:
                lexical.append(Prefetch(
                    query=SparseVector(indices=idx, values=val),
                    using=vec_name,
                    limit=limit * 2,
                ))

        prefetch: list[Prefetch] = [Prefetch(query=dense, using='full', limit=limit * 2)]
        if lexical:
            prefetch.append(Prefetch(
                prefetch=lexical,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit * 2,
            ))
        return prefetch

    async def search_chunks(self, text: str, limit: int) -> list[dict]:
        """RRF-поиск по chunks collection. Возвращает до `limit` chunk-dict'ов."""
        prefetch = await self._build_rrf_prefetch(self._chunks_collection, text, limit)
        result = await self._qdrant.query_points(
            collection_name=self._chunks_collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_point_to_chunk(p) for p in result.points]

    async def search_docs(self, text: str, limit: int) -> list[dict]:
        """RRF-поиск по docs collection (doc-level эмбеддинги полного текста).

        Возвращает dict'ы с полями: doc_id, title, path, parent_doc_ids,
        doc_summary, score. Используется в section-level retrieval.
        """
        prefetch = await self._build_rrf_prefetch(self._docs_collection, text, limit)
        result = await self._qdrant.query_points(
            collection_name=self._docs_collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        docs: list[dict] = []
        for p in result.points:
            pl = p.payload or {}
            path_raw = pl.get('path', '')
            paths: list[str] = path_raw if isinstance(path_raw, list) else ([path_raw] if path_raw else [])
            docs.append({
                'doc_id': pl.get('id', ''),
                'title': pl.get('title') or pl.get('id', ''),
                'path': paths,
                'parent_doc_ids': pl.get('parent_doc_ids', []) or [],
                'doc_summary': pl.get('doc_summary', ''),
                'score': float(p.score) if p.score is not None else 0.0,
            })
        return docs

    # ── Fetch helpers ─────────────────────────────────────────────────────────

    async def fetch_chunk_by_order(self, doc_id: str, order: int) -> dict | None:
        """Конкретный чанк по doc_id + order. Для get_neighbors tool."""
        records, _ = await self._qdrant.scroll(
            collection_name=self._chunks_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key='doc_id', match=MatchValue(value=doc_id)),
                FieldCondition(key='order', match=MatchValue(value=order)),
            ]),
            limit=1,
            with_payload=True,
        )
        if not records:
            return None
        chunk = _point_to_chunk(records[0])
        chunk['score'] = 0.0
        return chunk

    async def fetch_doc_summaries(self, doc_ids: list[str]) -> dict[str, str]:
        """Batch-fetch doc_summary по списку doc_id."""
        if not doc_ids:
            return {}
        try:
            records, _ = await self._qdrant.scroll(
                collection_name=self._docs_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key='id', match=MatchAny(any=doc_ids)),
                ]),
                with_payload=['id', 'doc_summary'],
                with_vectors=False,
                limit=len(doc_ids),
            )
        except Exception as exc:
            logger.warning('fetch_doc_summaries failed: %s', exc)
            return {}
        summaries: dict[str, str] = {}
        for rec in records:
            pl = rec.payload or {}
            did = pl.get('id')
            summary = pl.get('doc_summary')
            if did and summary:
                summaries[did] = summary
        return summaries

    async def build_doc_tree(self) -> tuple[dict[str, list[str]], set[str]]:
        """Parent→children дерево + set всех indexed doc_id. Кешируется при первом вызове."""
        if self._doc_tree is not None and self._indexed_doc_ids is not None:
            return self._doc_tree, self._indexed_doc_ids
        tree: dict[str, list[str]] = {}
        indexed: set[str] = set()
        offset = None
        while True:
            try:
                records, next_offset = await self._qdrant.scroll(
                    collection_name=self._docs_collection,
                    with_payload=['id', 'parent_doc_ids'],
                    with_vectors=False,
                    limit=100,
                    offset=offset,
                )
            except Exception as exc:
                logger.warning('build_doc_tree failed: %s', exc)
                break
            if not records:
                break
            for rec in records:
                pl = rec.payload or {}
                did = pl.get('id', '')
                if did:
                    indexed.add(did)
                for parent_id in pl.get('parent_doc_ids', []):
                    tree.setdefault(parent_id, []).append(did)
            offset = next_offset
            if offset is None:
                break
        self._doc_tree = tree
        self._indexed_doc_ids = indexed
        return self._doc_tree, self._indexed_doc_ids

    async def get_indexed_doc_ids(self) -> set[str]:
        """Set всех doc_id проиндексированных в docs collection."""
        if self._indexed_doc_ids is None:
            await self.build_doc_tree()
        return self._indexed_doc_ids or set()

    async def get_descendant_doc_ids(self, section_ids: list[str]) -> set[str]:
        """Развернуть section_ids в конкретные doc_id через BFS по parent-tree.

        Для flat_topics (cluster_membership) — подставляем список. Для иерархических
        — BFS от указанных section_ids вниз по tree.
        """
        membership = await self.fetch_cluster_membership()
        result: set[str] = set()
        tree_ids: list[str] = []
        for sid in section_ids:
            if sid in membership:
                result.update(membership[sid])
            else:
                tree_ids.append(sid)
        if tree_ids:
            tree, _ = await self.build_doc_tree()
            result.update(tree_ids)
            queue = list(tree_ids)
            while queue:
                parent = queue.pop(0)
                for child in tree.get(parent, []):
                    if child not in result:
                        result.add(child)
                        queue.append(child)
        return result

    async def fetch_knowledge_map(self) -> str:
        """Текст Knowledge Map (system prompt) из knowledge_map collection. Кеш."""
        if self._knowledge_map is not None:
            return self._knowledge_map
        try:
            records, _ = await self._qdrant.scroll(
                collection_name=self._km_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key='doc_id', match=MatchValue(value='_system_prompt')),
                ]),
                with_payload=['map_text'],
                with_vectors=False,
                limit=1,
            )
            self._knowledge_map = (records[0].payload or {}).get('map_text', '') if records else ''
        except Exception as exc:
            logger.warning('fetch_knowledge_map failed: %s', exc)
            self._knowledge_map = ''
        return self._knowledge_map

    async def fetch_cluster_membership(self) -> dict[str, list[str]]:
        """cluster_membership из knowledge_map collection (для flat_topics). Кеш."""
        if self._cluster_membership is not None:
            return self._cluster_membership
        try:
            records, _ = await self._qdrant.scroll(
                collection_name=self._km_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key='doc_id', match=MatchValue(value='_cluster_membership')),
                ]),
                with_payload=['cluster_membership'],
                with_vectors=False,
                limit=1,
            )
            if records:
                raw = (records[0].payload or {}).get('cluster_membership') or {}
                self._cluster_membership = {
                    k: list(v) for k, v in raw.items()
                    if isinstance(k, str) and isinstance(v, list)
                }
            else:
                self._cluster_membership = {}
        except Exception as exc:
            logger.warning('fetch_cluster_membership failed: %s', exc)
            self._cluster_membership = {}
        return self._cluster_membership

    async def get_doc_title(self, doc_id: str) -> str:
        """Title документа с кешом. Fallback = doc_id если не найдено."""
        if doc_id in self._doc_titles:
            return self._doc_titles[doc_id]
        try:
            records, _ = await self._qdrant.scroll(
                collection_name=self._docs_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key='id', match=MatchValue(value=doc_id)),
                ]),
                with_payload=['title'],
                with_vectors=False,
                limit=1,
            )
            if records:
                title = (records[0].payload or {}).get('title', doc_id)
                self._doc_titles[doc_id] = title
                return title
        except Exception:
            pass
        self._doc_titles[doc_id] = doc_id
        return doc_id
