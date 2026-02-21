import pytest
from qdrant_client import AsyncQdrantClient

from morag.storage.collections import ensure_chunks_collection, ensure_docs_collection
from morag.storage.repository import ChunkRepository, DocRepository


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(location=':memory:')
    yield client
    await client.close()


@pytest.fixture
async def doc_repo(qdrant):
    await ensure_docs_collection(qdrant, name='docs_test')
    return DocRepository(qdrant, collection='docs_test')


@pytest.fixture
async def chunk_repo(qdrant):
    await ensure_chunks_collection(qdrant, name='chunks_test')
    return ChunkRepository(qdrant, collection='chunks_test')
