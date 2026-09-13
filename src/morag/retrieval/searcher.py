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
import time
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
    SearchParams,
    SparseVector,
)

from morag import shortid
from morag.indexing.bm25 import tokenize, tokenize_trigram, to_sparse_vector
from morag.indexing.embedder import Embedder, SparseEmbedder

logger = logging.getLogger(__name__)


def build_payload_filter(
    exclude_source_names: list[str] | None,
    filters: dict[str, list[str]] | None,
) -> Filter | None:
    """Один Qdrant Filter на все Prefetch'и: исключённые источники (must_not) и сужение
    агента (must, по полю — MatchAny по списку значений). Пусто — None, поведение прежнее."""
    must = [
        FieldCondition(key=field, match=MatchAny(any=[str(v) for v in values]))
        for field, values in (filters or {}).items() if values
    ]
    must_not = [
        FieldCondition(key='source_name', match=MatchAny(any=exclude_source_names)),
    ] if exclude_source_names else []
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _point_to_chunk(p: Any, extra_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    """Конвертер Qdrant-point (ScoredPoint/Record/dict) → наш chunk-dict.

    `extra_fields` — поля payload, которые обязаны доехать до форматтера сверх белого списка
    (ADR-0027: поле чанка из аннотаций)."""
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
    extra = {f: payload.get(f) for f in extra_fields if payload.get(f) is not None}
    return {
        **extra,
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
        # Секундные таймкоды аудио-чанков → deep-link `#t=СЕК` на оригинал.
        'start_sec': payload.get('start_sec'),
        'end_sec': payload.get('end_sec'),
        # Спикеры реплики (для метки момент-цитаты «Выпуск · MM:SS · Спикер»).
        'speakers': payload.get('speakers'),
        'score': score,
        # Метки для table-row-narrative swap (ADR-0013). Если chunk_type
        # отсутствует в payload (старые чанки) — None, swap-логика их пропускает.
        'chunk_type': payload.get('chunk_type'),
        'parent_chunk_id': payload.get('parent_chunk_id'),
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
        hnsw_ef: int = 0,
        source_roles: dict[str, str] | None = None,
        source_kinds: dict[str, str] | None = None,
        cache_ttl_seconds: float = 300.0,
        annotation_field: str | None = None,
        annotation_leg: bool = False,
    ) -> None:
        self._qdrant = qdrant
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        # ADR-0027: поле чанка из аннотаций — доезжает до форматтера; при `annotation_leg` и наличии
        # одноимённого dense-вектора в схеме коллекции даёт ногу RRF верхнего уровня тем же вектором
        # запроса (по умолчанию выключено — замерено, что нога шумит титульными слайдами).
        self._annotation_field = annotation_field
        self._annotation_leg = annotation_leg
        self._extra_fields: tuple[str, ...] = (annotation_field,) if annotation_field else ()
        self._dense_vector_names_cache: dict[str, set[str]] = {}
        self._chunks_collection = chunks_collection
        self._docs_collection = docs_collection
        self._km_collection = knowledge_map_collection
        # search-time HNSW ef для dense Prefetch. 0 = Qdrant default.
        self._hnsw_ef = hnsw_ef
        # Snapshot ролей источников из config: source_name → 'primary'|'supplementary'|'hidden'.
        # Используется для построения фильтра по source_name на каждый search:
        # - hidden: всегда исключаем
        # - supplementary: исключаем если scope пустой И kinds не запросил этот kind
        # - primary: всегда включаем
        # Если snapshot пуст ({}) — фильтрация отключена (legacy/empty-config fallback).
        self._source_roles: dict[str, str] = dict(source_roles or {})
        self._source_kinds: dict[str, str] = dict(source_kinds or {})
        self._sparse_vector_names_cache: dict[str, set[str]] = {}
        self._doc_tree: dict[str, list[str]] | None = None
        self._indexed_doc_ids: set[str] | None = None
        self._doc_titles: dict[str, str] = {}
        self._cluster_membership: dict[str, list[str]] | None = None
        self._knowledge_map: str | None = None
        # Короткие коды документов (ADR-0025), двусторонне. Живут в том же корпус-снапшот кэше,
        # что дерево и заголовки: у них один жизненный цикл — «состав корпуса на сейчас».
        self._short_to_doc: dict[str, str] | None = None
        self._doc_to_short: dict[str, str] | None = None
        # TTL корпус-снапшот кэшей (KM/doc_tree/membership/titles). Без него
        # долгоживущий pipelines-процесс месяцами отвечает по KM и дереву
        # документов с момента своего старта, игнорируя cron-переиндексации
        # (реальный кейс: prod держал KM двухнедельной давности). 0 = кэш вечный.
        self._cache_ttl = cache_ttl_seconds
        self._cache_expires_at = 0.0
        # Значения полей payload для схемы `search(filters=…)`: собираются из коллекции
        # документов, живут по TTL — после переиндексации новая категория появится сама.
        self._filter_values: dict[str, list[str]] = {}

    def _maybe_expire_caches(self) -> None:
        """Сбросить корпус-снапшот кэши по TTL — следующий доступ перечитает из Qdrant.

        Зовётся из ленивых fetch'ей; fetch_knowledge_map выполняется на каждый
        pipe() → протухание проверяется на каждом запросе агента."""
        if self._cache_ttl <= 0:
            return
        now = time.monotonic()
        if now < self._cache_expires_at:
            return
        self._cache_expires_at = now + self._cache_ttl
        self._knowledge_map = None
        self._cluster_membership = None
        self._doc_tree = None
        self._indexed_doc_ids = None
        self._short_to_doc = None
        self._doc_to_short = None
        self._doc_titles.clear()
        self._sparse_vector_names_cache.clear()
        self._filter_values.clear()

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

    async def get_dense_vector_names(self, collection: str) -> set[str]:
        """Имена dense-векторов коллекции (кеш per-collection) — по ним решается, есть ли нога
        для поля аннотаций: старая коллекция без вектора → ноги нет, поиск прежний."""
        if collection in self._dense_vector_names_cache:
            return self._dense_vector_names_cache[collection]
        names: set[str] = set()
        try:
            info = await self._qdrant.get_collection(collection)
            vectors = info.config.params.vectors
            names = set(vectors.keys()) if isinstance(vectors, dict) else set()
        except Exception as exc:
            logger.warning('HybridSearcher: dense vector names for %s failed: %s', collection, exc)
        self._dense_vector_names_cache[collection] = names
        return names

    # ── Search (RRF) ──────────────────────────────────────────────────────────

    def _excluded_source_names(
        self,
        kinds: list[str] | None,
        scope_active: bool,
    ) -> list[str]:
        """Какие source_name исключить из выдачи в данном вызове.

        - hidden: всегда (admin kill-switch).
        - При scope_active=True (search со section_ids/doc_ids): больше ничего —
          descendants раздела сами решают что включать (тикеты Jira привязанные
          к Confluence-разделу естественно попадают через parent_doc_ids).
        - При scope_active=False:
            - kinds=None → исключаем все supplementary
            - kinds=['jira'] → исключаем только supplementary НЕ запрошенного kind
        """
        if not self._source_roles:
            return []
        excluded = {n for n, r in self._source_roles.items() if r == 'hidden'}
        if scope_active:
            result = sorted(excluded)
        else:
            requested_kinds = set(kinds or [])
            for name, role in self._source_roles.items():
                if role != 'supplementary':
                    continue
                if self._source_kinds.get(name) not in requested_kinds:
                    excluded.add(name)
            result = sorted(excluded)
        logger.debug(
            '[searcher] exclude: kinds=%s scope_active=%s → excluded=%s',
            kinds, scope_active, result,
        )
        return result

    async def _build_rrf_prefetch(
        self,
        collection: str,
        text: str,
        limit: int,
        exclude_source_names: list[str] | None = None,
        filters: dict[str, list[str]] | None = None,
    ) -> list[Prefetch]:
        """Двухуровневый RRF: dense `full` (1 голос) vs nested-RRF по sparse (1 голос).

        Sparse-каналы: `keywords` (GTE), `bm25` (Snowball stem), `bm25_trigram`
        (символьные триграммы). Используются только существующие в схеме коллекции.

        `exclude_source_names` фильтрует payload.source_name на КАЖДОМ sub-Prefetch
        (до RRF-мерджа): исключённые источники не отбирают слотов у остальных
        в top-N каналов. Применяется к dense Prefetch и к каждому lexical
        sub-Prefetch внутри nested-RRF.

        `filters` — сужение агента `{поле: [значения]}` по payload чанка: тем же
        Filter'ом, на тех же Prefetch'ах, до слияния. Поле-список (например, темы)
        совпадает, если совпал хотя бы один элемент — так работает MatchAny в Qdrant.
        """
        dense = await self._dense.embed_query(text)
        indices, values = await self._sparse.embed_query(text)
        available_sparse = await self.get_sparse_vector_names(collection)

        kind_filter = build_payload_filter(exclude_source_names, filters)

        lexical: list[Prefetch] = []
        if 'keywords' in available_sparse:
            lexical.append(Prefetch(
                query=SparseVector(indices=indices, values=values),
                using='keywords',
                limit=limit * 2,
                filter=kind_filter,
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
                    filter=kind_filter,
                ))

        dense_params = SearchParams(hnsw_ef=self._hnsw_ef) if self._hnsw_ef else None
        prefetch: list[Prefetch] = [
            Prefetch(
                query=dense, using='full', limit=limit * 2,
                params=dense_params, filter=kind_filter,
            ),
        ]
        # ADR-0027: нога по вектору поля аннотаций — ВЕРХНЕГО уровня (свой голос RRF, как у dense
        # речи), тем же вектором запроса; только если вектор есть в схеме коллекции.
        ann_field = getattr(self, '_annotation_field', None) if getattr(self, '_annotation_leg', False) else None
        if ann_field and ann_field in await self.get_dense_vector_names(collection):
            prefetch.append(Prefetch(
                query=dense, using=ann_field, limit=limit * 2,
                params=dense_params, filter=kind_filter,
            ))
        if lexical:
            prefetch.append(Prefetch(
                prefetch=lexical,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit * 2,
            ))
        return prefetch

    async def search_chunks(
        self,
        text: str,
        limit: int,
        kinds: list[str] | None = None,
        scope_active: bool = False,
        filters: dict[str, list[str]] | None = None,
    ) -> list[dict]:
        """RRF-поиск по chunks collection. Возвращает до `limit` chunk-dict'ов.

        Постобработка: narrative-чанки (chunk_type='table_row_narrative')
        свапаются на родительские table-чанки (см. ADR-0013).

        Фильтрация по ролям источников (см. `_excluded_source_names`):
          - `kinds`: дополнительные kind'ы supplementary-источников которые
            агент явно хочет видеть (например ['jira']).
          - `scope_active`: True если есть section_ids/doc_ids — тогда
            supplementary не фильтруется (descendants естественно тянут).
        """
        prefetch = await self._build_rrf_prefetch(
            self._chunks_collection, text, limit,
            exclude_source_names=self._excluded_source_names(kinds, scope_active),
            filters=filters,
        )
        result = await self._qdrant.query_points(
            collection_name=self._chunks_collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        chunks = [_point_to_chunk(p, getattr(self, '_extra_fields', ())) for p in result.points]
        return await self._swap_narratives_to_parents(chunks)

    async def filter_values(self, fields: list[str]) -> dict[str, list[str]]:
        """Различные значения полей payload по коллекции ДОКУМЕНТОВ (структурные узлы не в счёт).

        Идут в схему инструмента `search` как `enum`: агент выбирает из того, что есть, а не
        угадывает написание. Кэш по TTL — после переиндексации значения подъедут сами.
        """
        self._maybe_expire_caches()
        missing = [f for f in fields if f not in self._filter_values]
        if missing:
            found: dict[str, set[str]] = {f: set() for f in missing}
            offset = None
            while True:
                points, offset = await self._qdrant.scroll(
                    collection_name=self._docs_collection,
                    limit=256, offset=offset,
                    with_payload=[*missing, 'structural'], with_vectors=False,
                )
                for pt in points:
                    payload = pt.payload or {}
                    if payload.get('structural'):
                        continue
                    for f in missing:
                        value = payload.get(f)
                        for v in (value if isinstance(value, list) else [value]):
                            if v not in (None, ''):
                                found[f].add(str(v))
                if offset is None:
                    break
            for f, vals in found.items():
                self._filter_values[f] = sorted(vals)
        return {f: list(self._filter_values.get(f, [])) for f in fields}

    async def search_docs(
        self,
        text: str,
        limit: int,
        kinds: list[str] | None = None,
        scope_active: bool = False,
    ) -> list[dict]:
        """RRF-поиск по docs collection (doc-level эмбеддинги полного текста).

        Возвращает dict'ы с полями: doc_id, title, path, parent_doc_ids,
        doc_summary, score. Используется в section-level retrieval.

        Фильтрация — та же что и в search_chunks (см. `_excluded_source_names`).
        В find_section вызывается с kinds=None и scope_active=False → supplementary
        источники не голосуют за секции.
        """
        prefetch = await self._build_rrf_prefetch(
            self._docs_collection, text, limit,
            exclude_source_names=self._excluded_source_names(kinds, scope_active),
        )
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

    async def fetch_doc_chunks_lite(self, doc_id: str) -> list[dict]:
        """Все чанки документа в LITE-формате `[{order, text}]`, отсортированы по order.

        Для DocReranker (get_doc tool): он рассуждает только по тексту и order,
        полная payload-обвязка избыточна. Narrative-чанки (order=-1) исключаются.
        """
        chunks: list[dict] = []
        offset = None
        while True:
            points, offset = await self._qdrant.scroll(
                collection_name=self._chunks_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key='doc_id', match=MatchValue(value=doc_id))],
                    must_not=[FieldCondition(
                        key='chunk_type',
                        match=MatchValue(value='table_row_narrative'),
                    )],
                ),
                limit=200,
                offset=offset,
                with_payload=['order', 'text'],
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                order = pl.get('order')
                text = pl.get('text', '')
                if order is None or order < 0:
                    continue
                chunks.append({'order': order, 'text': text})
            if offset is None:
                break
        chunks.sort(key=lambda c: c['order'])
        return chunks

    async def fetch_chunks_by_orders(
        self, doc_id: str, orders: list[int],
    ) -> list[dict]:
        """Полные чанки одного документа по списку order'ов (после DocReranker).

        Возвращает chunk-dict'ы со всеми полями (text, context, path, ...) — те же
        что отдаёт search_chunks. Сортируются по order asc.
        """
        if not orders:
            return []
        records, _ = await self._qdrant.scroll(
            collection_name=self._chunks_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key='doc_id', match=MatchValue(value=doc_id)),
                    FieldCondition(key='order', match=MatchAny(any=list(orders))),
                ],
                must_not=[FieldCondition(
                    key='chunk_type',
                    match=MatchValue(value='table_row_narrative'),
                )],
            ),
            limit=len(orders) + 10,
            with_payload=True,
        )
        chunks = [_point_to_chunk(r, getattr(self, '_extra_fields', ())) for r in records]
        chunks.sort(key=lambda c: c.get('order', 0))
        return chunks

    async def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch чанков по их UUID point-id'ам.

        Используется в _swap_narratives_to_parents для подгрузки parent-чанков.
        Возвращает dict {chunk_id: chunk_dict}.
        """
        if not chunk_ids:
            return {}
        records = await self._qdrant.retrieve(
            collection_name=self._chunks_collection,
            ids=list(chunk_ids),
            with_payload=True,
        )
        return {str(r.id): _point_to_chunk(r, getattr(self, '_extra_fields', ())) for r in records}

    async def _swap_narratives_to_parents(self, chunks: list[dict]) -> list[dict]:
        """Заменить narrative-чанки на их parent table-чанки в выдаче поиска.

        Логика (ADR-0013):
          - chunks приходят отсортированные по score (RRF output).
          - Идём по списку в порядке скорa.
          - Для narrative (chunk_type='table_row_narrative'): если parent ещё не
            был добавлен в result — fetch'им parent, добавляем со score=narrative.score
            (строгое наследование). Если parent уже добавлен — drop narrative.
          - Для обычного chunk: если он уже добавлен через swap другого narrative
            — drop. Иначе — добавляем со своим score.

        Дедупликация: каждый chunk_id появляется в result ровно один раз.
        """
        # Pre-scan: какие parent_id нужно подтянуть (не входящие в текущий result)
        regular_ids = {
            c['chunk_id'] for c in chunks
            if c.get('chunk_type') != 'table_row_narrative'
        }
        parent_ids_to_fetch = {
            c['parent_chunk_id'] for c in chunks
            if c.get('chunk_type') == 'table_row_narrative' and c.get('parent_chunk_id')
        }
        # Parent, который уже среди обычных результатов — не fetch'им.
        parent_ids_to_fetch -= regular_ids
        parents = await self.fetch_chunks_by_ids(list(parent_ids_to_fetch))

        seen_ids: set[str] = set()
        result: list[dict] = []
        for c in chunks:
            if c.get('chunk_type') == 'table_row_narrative':
                pid = c.get('parent_chunk_id')
                if not pid or pid in seen_ids:
                    continue  # malformed или parent уже в result
                parent = parents.get(pid)
                if parent is None:
                    # parent был среди regular_ids — он появится сам ниже по итерации.
                    # Drop narrative, не дублируем (parent отыграет со своим score).
                    continue
                parent_copy = dict(parent)
                parent_copy['score'] = c['score']  # строго наследуем narrative.score
                result.append(parent_copy)
                seen_ids.add(pid)
            else:
                cid = c['chunk_id']
                if cid in seen_ids:
                    continue  # уже свапнут narrative'ом с более высокого ранга
                result.append(c)
                seen_ids.add(cid)
        return result

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
        """Parent→children дерево + set всех indexed doc_id. Кеш с TTL."""
        self._maybe_expire_caches()
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
        """Текст Knowledge Map (system prompt) из knowledge_map collection. Кеш с TTL."""
        self._maybe_expire_caches()
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

    async def fetch_short_id_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """`(код → doc_id, doc_id → код)` по всему корпусу. Кеш с TTL, один скролл.

        Источник истины — поле `short_id` в payload. У документов без него код ВЫВОДИТСЯ из
        `doc_id` тем же алгоритмом, что у индексатора: поэтому инстанс, который ещё не прошёл
        миграцию, работает как обычно, а смешанное состояние (часть записей с полем, часть без)
        не даёт ни дублей, ни пропаж.

        ⚠️ Порядок вывода — по возрастанию `doc_id`, и он значим: при коллизии код удлиняется у
        того, кто пришёл позже, а «позже» здесь задаётся именно сортировкой. Иначе два процесса
        морага на одном корпусе разошлись бы в кодах для той доли документов, что удлинилась.
        """
        self._maybe_expire_caches()
        if self._short_to_doc is not None and self._doc_to_short is not None:
            return self._short_to_doc, self._doc_to_short
        stored: dict[str, str] = {}       # код → doc_id, как записано индексатором
        structural: dict[str, bool] = {}  # doc_id → структурный ли (для буквы типа)
        derive: list[str] = []            # doc_id без кода в payload
        offset = None
        try:
            while True:
                points, offset = await self._qdrant.scroll(
                    collection_name=self._docs_collection,
                    limit=256, offset=offset,
                    with_payload=['id', 'short_id', 'structural'], with_vectors=False,
                )
                for pt in points:
                    pl = pt.payload or {}
                    doc_id = pl.get('id')
                    if not doc_id:
                        continue
                    structural[str(doc_id)] = bool(pl.get('structural'))
                    code = pl.get('short_id')
                    if code:
                        stored[str(code)] = str(doc_id)
                    else:
                        derive.append(str(doc_id))
                if offset is None:
                    break
        except Exception as exc:
            logger.warning('fetch_short_id_maps failed: %s', exc)
            self._short_to_doc, self._doc_to_short = {}, {}
            return self._short_to_doc, self._doc_to_short
        for doc_id in sorted(derive):
            stored[shortid.assign(
                doc_id, structural=structural.get(doc_id, False), taken=stored,
            )] = doc_id
        self._short_to_doc = stored
        self._doc_to_short = {doc: code for code, doc in stored.items()}
        if derive:
            logger.info(
                'short ids: %d from payload, %d derived on the fly (run backfill-short-ids)',
                len(stored) - len(derive), len(derive),
            )
        return self._short_to_doc, self._doc_to_short

    async def fetch_cluster_membership(self) -> dict[str, list[str]]:
        """cluster_membership из knowledge_map collection (для flat_topics). Кеш с TTL."""
        self._maybe_expire_caches()
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
