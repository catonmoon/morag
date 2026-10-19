"""ADR-0034: Карта знаний — общий текст на все запросы, документы с меткой доступа в неё не идут."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from morag.access import ACCESS_FIELD
from morag.indexing.knowledge_map import KnowledgeMapGenerator
from morag.sources.base import Document


def _doc(doc_id: str, label=None) -> Document:
    return Document(id=doc_id, path=[doc_id], text='', source_type='local',
                    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    payload={ACCESS_FIELD: label} if label is not None else {})


@pytest.mark.asyncio
async def test_labeled_documents_never_reach_the_map():
    gen = KnowledgeMapGenerator.__new__(KnowledgeMapGenerator)
    gen._exclude_source_types = []
    gen._prompt_strategy = 'flat_topics'
    gen._counter = MagicMock(count=lambda t: 1)
    gen.ensure_collection = AsyncMock()
    gen._load_all_docs = AsyncMock(return_value=[
        _doc('open'), _doc('empty-label', []), _doc('closed', ['rec:42'])])
    seen: list[str] = []

    async def _flat(docs):
        seen.extend(d.id for d in docs)
        return 'map', {}
    gen._build_flat_topics_prompt = _flat
    gen._save_maps = AsyncMock()
    await gen.generate()
    assert seen == ['open', 'empty-label']


@pytest.mark.asyncio
async def test_access_label_survives_the_payload_whitelist():
    client = AsyncQdrantClient(location=':memory:')
    await client.create_collection('docs', vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    await client.upsert('docs', [PointStruct(id=1, vector=[1.0, 0.0], payload={
        'id': 'closed', 'path': ['closed'], 'updated_at': '2026-01-01T00:00:00+00:00',
        'source_type': 'local', ACCESS_FIELD: ['rec:42'], 'text': 'секрет'})])
    gen = KnowledgeMapGenerator.__new__(KnowledgeMapGenerator)
    gen._client = client
    gen._doc_repo = MagicMock(collection='docs')
    docs = await gen._load_all_docs()
    assert docs[0].payload[ACCESS_FIELD] == ['rec:42']
