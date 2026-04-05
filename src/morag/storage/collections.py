from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, SparseVectorParams, VectorParams


async def ensure_docs_collection(client: AsyncQdrantClient, name: str = 'docs') -> None:
    """Создать коллекцию документов если не существует.

    Хранит полный текст документов и метаданные без векторов.
    Payload-индекс на поле 'id' для быстрых idempotency-проверок.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        return

    await client.create_collection(
        collection_name=name,
        vectors_config={},  # документы хранятся без векторов
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
