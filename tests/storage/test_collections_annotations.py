"""Именованный dense-вектор поля аннотаций в коллекции чанков (ADR-0027): создание и детектор."""
from qdrant_client import AsyncQdrantClient

from morag.storage.collections import ensure_chunks_collection, frida_vectors_config, missing_dense_vectors


def test_extra_vector_in_config():
    cfg = frida_vectors_config(4, extra=['screen'])
    assert set(cfg) == {'full', 'screen'} and cfg['screen'].size == 4
    assert set(frida_vectors_config(4)) == {'full'}


async def test_new_collection_gets_extra_vector_and_old_one_is_reported():
    client = AsyncQdrantClient(location=':memory:')
    try:
        await ensure_chunks_collection(client, 'old', vectors_config=frida_vectors_config(4))
        await ensure_chunks_collection(client, 'new', vectors_config=frida_vectors_config(4, extra=['screen']))
        assert await missing_dense_vectors(client, 'new', {'screen'}) == set()
        assert await missing_dense_vectors(client, 'old', {'screen'}) == {'screen'}
        # ensure у существующей коллекции схему не сверяет — вектор сам не появится
        await ensure_chunks_collection(client, 'old', vectors_config=frida_vectors_config(4, extra=['screen']))
        assert await missing_dense_vectors(client, 'old', {'screen'}) == {'screen'}
        assert await missing_dense_vectors(client, 'absent', {'screen'}) == set()
    finally:
        await client.close()
