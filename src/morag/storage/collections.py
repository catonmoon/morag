from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, SparseVectorParams, VectorParams

logger = logging.getLogger(__name__)


async def ensure_docs_collection(
    client: AsyncQdrantClient,
    name: str = 'docs',
    vectors_config: dict[str, VectorParams] | None = None,
    sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
) -> None:
    """Создать коллекцию документов если не существует.

    По умолчанию — payload-only (без векторов). Если переданы vectors_config /
    sparse_vectors_config — создаётся с именованными векторами для section-level
    retrieval (doc-level embeddings: полный текст документа эмбеддится как один чанк).

    Payload-индексы: 'id' для idempotency + 'parent_doc_ids' для section aggregation.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        return

    await client.create_collection(
        collection_name=name,
        vectors_config=vectors_config or {},
        sparse_vectors_config=sparse_vectors_config,
    )
    await client.create_payload_index(
        collection_name=name,
        field_name='id',
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=name,
        field_name='parent_doc_ids',
        field_schema=PayloadSchemaType.KEYWORD,
    )


async def ensure_chunks_collection(
    client: AsyncQdrantClient,
    name: str = 'chunks',
    vectors_config: dict[str, VectorParams] | None = None,
    sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
) -> None:
    """Создать коллекцию чанков если не существует.

    Именованные векторы определяются конфигурацией embedding-процессоров.
    Payload-индекс на поле 'doc_id' для каскадного удаления при переиндексации.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        return

    await client.create_collection(
        collection_name=name,
        vectors_config=vectors_config or {},
        sparse_vectors_config=sparse_vectors_config,
    )
    await client.create_payload_index(
        collection_name=name,
        field_name='doc_id',
        field_schema=PayloadSchemaType.KEYWORD,
    )


async def ensure_payload_indexes(
    client: AsyncQdrantClient,
    collection: str,
    fields: list[str],
) -> None:
    """Keyword-индексы на поля payload, по которым фильтрует `search`.

    ⚠️ Зовётся на КАЖДОМ запуске индексатора, а не при создании коллекции: `ensure_*_collection`
    делает early-return у существующей коллекции, и индекс на новое поле сам не появится
    (грабли `backfill-short-ids`). «Уже существует» Qdrant отдаёт исключением — это норма.
    """
    for field in fields:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001
            # «Уже есть» — норма; всё остальное (нет коллекции, нет связи) — в лог, но не
            # ронять индексацию: без индекса фильтр медленнее, а не сломан.
            if 'already' not in str(exc).lower():
                logger.warning('payload index %s.%s: %s', collection, field, exc)


def make_dense_vector_config(size: int, distance: Distance = Distance.COSINE) -> VectorParams:
    """Вспомогательная функция для создания конфига dense-вектора."""
    return VectorParams(size=size, distance=distance)


def frida_vectors_config(dim: int) -> dict[str, VectorParams]:
    """Конфиг именованных векторов для коллекции чанков с FRIDA-эмбеддингами."""
    return {'full': VectorParams(size=dim, distance=Distance.COSINE)}


def gte_sparse_vectors_config() -> dict[str, SparseVectorParams]:
    """Конфиг sparse-векторов для коллекции чанков с GTE-эмбеддингами и BM25.

    bm25 — стемминг (морфология)
    bm25_phonetic — фонетическая нормализация + триграммы
    bm25_translit — транслитерация кириллица↔латиница
    """
    return {
        'keywords': SparseVectorParams(),
        'bm25': SparseVectorParams(),
        'bm25_trigram': SparseVectorParams(),
    }


async def upgrade_sparse_vectors(
    client: AsyncQdrantClient,
    name: str = 'chunks',
) -> bool:
    """Проверить что коллекция имеет все нужные sparse vectors.

    Qdrant 1.x не поддерживает добавление sparse vectors к существующей коллекции.
    Если не хватает — логирует предупреждение, возвращает False.
    Для добавления нужна полная переиндексация (--reset).

    Returns: True если все вектора на месте, False если нужен --reset.
    """
    import logging
    logger = logging.getLogger(__name__)

    existing = {c.name for c in (await client.get_collections()).collections}
    if name not in existing:
        return True  # коллекция будет создана с полной схемой

    info = await client.get_collection(name)
    current_sparse = set()
    if info.config.params.sparse_vectors:
        current_sparse = set(info.config.params.sparse_vectors.keys())

    target = gte_sparse_vectors_config()
    missing = {k for k in target if k not in current_sparse}

    if not missing:
        logger.info('All sparse vectors present: %s', sorted(current_sparse))
        return True

    logger.warning(
        'Missing sparse vectors: %s. Run with --reset to recreate collection.',
        sorted(missing),
    )
    return False
