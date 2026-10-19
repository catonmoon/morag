"""ADR-0034: метки доступа — из сайдкара при загрузке и сверкой без переэмбеддинга."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from morag.access import ACCESS_FIELD, read_sidecar
from morag.indexing.access_sync import sync_access
from morag.sources.local import LocalDocumentSource
from morag.sources.markdown import MarkdownSource

SUFFIX = '.access.json'


def _write(root: Path, rel: str, text: str = '# x\n') -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def test_sidecar_read_rules(tmp_path):
    md = _write(tmp_path, 'a/record.md')
    assert read_sidecar(md, SUFFIX) is None                     # сайдкара нет — метку не трогаем
    assert read_sidecar(md, None) is None                       # чтение выключено
    md.with_name('record' + SUFFIX).write_text(json.dumps({ACCESS_FIELD: ['b', 'a', 'a']}))
    assert read_sidecar(md, SUFFIX) == ['a', 'b']
    md.with_name('record' + SUFFIX).write_text('{битый')
    assert read_sidecar(md, SUFFIX) == ['!unreadable']           # битый — закрыт, а не открыт


@pytest.mark.asyncio
async def test_document_is_born_with_its_label_and_sidecar_beats_front_matter(tmp_path):
    md = _write(tmp_path, 'r/record.md', '---\ntitle: T\naccess: [everyone]\n---\nтекст\n')
    md.with_name('record' + SUFFIX).write_text(json.dumps({ACCESS_FIELD: ['rec:42']}))
    src = MarkdownSource(tmp_path, name='c', access_suffix=SUFFIX)
    doc = await src.load_one('local:c:r/record.md')
    assert doc.payload[ACCESS_FIELD] == ['rec:42']
    # mtime сайдкара не двигает updated_at: правка прав не стоит переэмбеддинга
    stub = (await src.get_metadata())[0]
    assert stub.updated_at == doc.updated_at


@pytest.mark.asyncio
async def test_without_access_the_source_is_unchanged(tmp_path):
    md = _write(tmp_path, 'r/record.md')
    md.with_name('record' + SUFFIX).write_text(json.dumps({ACCESS_FIELD: ['rec:42']}))
    doc = await MarkdownSource(tmp_path, name='c').load_one('local:c:r/record.md')
    assert ACCESS_FIELD not in doc.payload


def test_access_map_covers_every_file_open_without_sidecar(tmp_path):
    a = _write(tmp_path, 'x/a.md')
    _write(tmp_path, 'y/b.md')
    a.with_name('a' + SUFFIX).write_text(json.dumps({ACCESS_FIELD: ['team:x']}))
    m = LocalDocumentSource(tmp_path, name='c', access_suffix=SUFFIX).access_map()
    assert m == {'local:c:x/a.md': ['team:x'], 'local:c:y/b.md': []}
    assert LocalDocumentSource(tmp_path, name='c').access_map() == {}


async def _index(client):
    for coll in ('docs', 'chunks'):
        await client.create_collection(coll, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    docs = {'d-open': None, 'd-closed': ['rec:1'], 'd-same': ['rec:2'], 'd-other': None}
    await client.upsert('docs', [
        PointStruct(id=i, vector=[1.0, 0.0],
                    payload={'id': d, **({ACCESS_FIELD: lab} if lab else {})})
        for i, (d, lab) in enumerate(docs.items(), 1)])
    await client.upsert('chunks', [
        PointStruct(id=10 * i + k, vector=[1.0, 0.0],
                    payload={'doc_id': d, 'text': f'{d} {k}', **({ACCESS_FIELD: lab} if lab else {})})
        for i, (d, lab) in enumerate(docs.items(), 1) for k in range(2)])


async def _labels(client, coll, key):
    pts, _ = await client.scroll(coll, limit=100, with_payload=True, with_vectors=True)
    return {(p.payload[key], p.id): (p.payload.get(ACCESS_FIELD), p.vector) for p in pts}


@pytest.mark.asyncio
async def test_sync_writes_only_payload_and_reports_closed():
    client = AsyncQdrantClient(location=':memory:')
    await _index(client)
    before = await _labels(client, 'chunks', 'doc_id')
    res = await sync_access(client, 'docs', 'chunks', {
        'd-open': ['rec:9'],       # открытый → закрыт
        'd-closed': [],            # закрытый → открыт
        'd-same': ['rec:2'],       # без изменений
        'd-missing': ['rec:7'],    # в индексе нет — не трогаем
    })
    assert sorted(res.changed) == ['d-closed', 'd-open'] and res.closed == ['d-open']
    after = await _labels(client, 'chunks', 'doc_id')
    for (doc_id, pid), (label, vec) in after.items():
        assert vec == before[(doc_id, pid)][1]                   # векторы не тронуты
        expected = {'d-open': ['rec:9'], 'd-closed': None, 'd-same': ['rec:2'], 'd-other': None}
        assert label == expected[doc_id]
    docs = {k[0]: v[0] for k, v in (await _labels(client, 'docs', 'id')).items()}
    assert docs == {'d-open': ['rec:9'], 'd-closed': None, 'd-same': ['rec:2'], 'd-other': None}
