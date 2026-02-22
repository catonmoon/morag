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


FRIDA_DIM = 1024


def make_dense_vector_config(size: int, distance: Distance = Distance.COSINE) -> VectorParams:
    """Вспомогательная функция для создания конфига dense-вектора."""
    return VectorParams(size=size, distance=distance)


def frida_vectors_config() -> dict[str, VectorParams]:
    """Конфиг именованных векторов для коллекции чанков с FRIDA-эмбеддингами."""
    return {'full': VectorParams(size=FRIDA_DIM, distance=Distance.COSINE)}


def gte_sparse_vectors_config() -> dict[str, SparseVectorParams]:
    """Конфиг sparse-векторов для коллекции чанков с GTE-эмбеддингами."""
    return {'keywords': SparseVectorParams()}
