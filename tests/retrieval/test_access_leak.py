"""ADR-0034, главный тест: закрытое не доходит до агента НИ ОДНИМ путём.

Настоящий `Pipeline` и настоящий `HybridSearcher` поверх Qdrant в памяти; подменены только
эмбеддер (константный вектор — ранжирование здесь не предмет), реранкеры (пропускают всё) и
хозяин прав. Закрытый документ несёт маркер `СЕКРЕТ-7f3a` в тексте, заголовке и поле фильтра —
для чужого его не должно быть ни в одном тексте, ни в одном чанке; для допущенного он находится.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from morag.access import ACCESS_FIELD, OPEN, RequestContext
from morag.retrieval.searcher import HybridSearcher
from morag.retrieval.sections import FindSectionConfig
from morag.storage.repository import DocRepository, doc_point_id

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402

MARK = 'СЕКРЕТ-7f3a'
OPEN_ID = 'local:c:open/record.md'
CLOSED_ID = 'local:c:closed/record.md'
STALE_ID = 'local:c:stale/record.md'   # в копии прав открыт, у хозяина — уже закрыт
FOLDERS = ['local:c:open/', 'local:c:closed/', 'local:c:stale/']

DOCS = {
    OPEN_ID: {'title': 'Открытый доклад', 'text': 'про очереди и kafka', 'category': 'Очереди'},
    CLOSED_ID: {'title': f'Личное {MARK}', 'text': f'про очереди и kafka {MARK}',
                'category': f'Категория {MARK}', ACCESS_FIELD: ['rec:42']},
    STALE_ID: {'title': f'Скрыт час назад {MARK}', 'text': f'про очереди и kafka {MARK}',
               'category': 'Очереди'},
}


class Owner:
    """Хозяин прав: rec:42 открыт только своим; stale закрыт для всех, кроме автора."""
    def principals(self, asker):
        return frozenset({'rec:42'}) if asker == 'ldap:author' else frozenset()

    def check(self, asker, doc_ids):
        ok = {OPEN_ID, *FOLDERS}
        if asker == 'ldap:author':
            ok |= {CLOSED_ID, STALE_ID}
        return set(doc_ids) & ok


async def _fill(client):
    for coll in ('docs', 'chunks'):
        await client.create_collection(
            coll, vectors_config={'full': VectorParams(size=2, distance=Distance.COSINE)})
    points, chunks = [], []
    for i, (doc_id, d) in enumerate(DOCS.items(), 1):
        folder = doc_id.rsplit('/', 1)[0] + '/'
        label = {ACCESS_FIELD: d[ACCESS_FIELD]} if ACCESS_FIELD in d else {}
        points.append(PointStruct(id=doc_point_id(doc_id), vector={'full': [1.0, 0.0]}, payload={
            'id': doc_id, 'path': [doc_id.split(':', 2)[2]], 'text': d['text'], 'title': d['title'],
            'updated_at': '2026-01-01T00:00:00+00:00', 'source_type': 'markdown',
            'parent_doc_ids': [folder], 'doc_summary': d['text'], 'category': d['category'],
            'short_id': f'D00000{i}', **label}))
        for k in range(2):
            chunks.append(PointStruct(id=100 * i + k, vector={'full': [1.0, 0.0]}, payload={
                'doc_id': doc_id, 'path': [doc_id.split(':', 2)[2]], 'order': k, 'total': 2,
                'text': f'{d["text"]} — часть {k}', 'context': '', 'source_type': 'markdown',
                'updated_at': '2026-01-01T00:00:00+00:00', 'category': d['category'], **label}))
    for j, folder in enumerate(FOLDERS):
        points.append(PointStruct(id=doc_point_id(folder), vector={'full': [0.5, 0.5]}, payload={
            'id': folder, 'path': [folder.split(':', 2)[2]], 'text': '', 'title': folder,
            'updated_at': '2026-01-01T00:00:00+00:00', 'source_type': 'directory',
            'parent_doc_ids': [], 'structural': True, 'short_id': f'S00000{j}'}))
    await client.upsert('docs', points)
    await client.upsert('chunks', chunks)


@pytest.fixture
def p():
    pl = object.__new__(Pipeline)
    pl._loop = asyncio.new_event_loop()
    pl._loop_lock = threading.Lock()
    client = AsyncQdrantClient(location=':memory:')
    pl._loop.run_until_complete(_fill(client))
    searcher = HybridSearcher(
        qdrant=client,
        dense_embedder=SimpleNamespace(embed_query=AsyncMock(return_value=[1.0, 0.0])),
        sparse_embedder=SimpleNamespace(embed_query=AsyncMock(return_value=([], []))),
        chunks_collection='chunks', docs_collection='docs')
    searcher._access_enabled = True
    pl._searcher = searcher
    pl._qdrant = client
    pl._doc_repo = DocRepository(client, 'docs')
    pl._rights_owner = Owner()
    pl._s = {
        'search_limit': 50, 'unique_docs_cap': 10, 'docs_collection': 'docs',
        'search_filters': [{'field': 'category', 'enum': True, 'max_values': 60}],
        'access_enabled': True, 'access_key': 'k', 'timestamp_citations': False,
        'agent_context_window': 100000,
    }
    pl._find_section_config = FindSectionConfig(descent_threshold=0)
    pl._rerank = lambda query, cs: list(cs)
    pl._doc_reranker = SimpleNamespace(
        rerank=AsyncMock(side_effect=lambda q, path, lite: [c['order'] for c in lite]))
    pl._tool_instances = {}
    pl._doc_numbering, pl._cite_units, pl._live_titles = {}, {}, {}
    pl._fetch_page_live = lambda *a: (_ for _ in ()).throw(AssertionError('live fetch called'))
    yield pl
    pl._loop.close()


def _ctx(p, asker=None):
    body = {'asker': {'id': asker, 'key': 'k'}} if asker else {}
    return p._request_context(body)


def _leaks(text, chunks):
    blob = text + ' '.join(str(c) for c in chunks)
    return MARK in blob


def _channels(p, ctx):
    """Все пути, которыми документ доходит до агента, — текст и чанки каждого."""
    out = {
        'search': p._tool_search('очереди kafka', ctx=ctx),
        'search-scoped-to-closed': p._tool_search('очереди', doc_ids=[CLOSED_ID, STALE_ID], ctx=ctx),
        # Агент угадал значение закрытого поля: эхо его собственного аргумента — не утечка, а
        # среди «допустимых значений» закрытого быть не должно.
        'search-filter': (lambda t, c: (t.replace(f'Категория {MARK}', '<аргумент агента>'), c))(
            *p._tool_search('очереди', filters={'category': [f'Категория {MARK}']}, ctx=ctx)),
        'find_section': p._tool_find_section('очереди kafka', ctx),
        'get_doc': p._tool_get_doc(CLOSED_ID, 'очереди', ctx),
        'get_doc-stale': p._tool_get_doc(STALE_ID, 'очереди', ctx),
        'lookup': p._tool_lookup([CLOSED_ID, STALE_ID], 'очереди', ctx),
        'catalog': p._tool_catalog(['title', 'category'], ctx),
    }
    fr = p._fetch_refs([CLOSED_ID, STALE_ID], budget=10 ** 6, ctx=ctx)
    out['auto-fetch'] = (' '.join(fr['loaded']), fr['chunks'])
    out['filter-values'] = (str(p._filter_values()), [])
    return out


@pytest.mark.parametrize('asker', [None, 'ldap:stranger', 'local:author'])
def test_nothing_closed_reaches_the_agent(p, asker):
    ctx = _ctx(p, asker)
    for name, (text, chunks) in _channels(p, ctx).items():
        assert not _leaks(text, chunks), f'{name} leaked for {asker}: {text[:300]}'


def test_closed_answers_exactly_like_missing(p):
    ctx = _ctx(p, 'ldap:stranger')
    closed, _ = p._tool_get_doc(CLOSED_ID, 'q', ctx)
    missing, _ = p._tool_get_doc('local:c:nope/record.md', 'q', ctx)
    assert closed.replace(p._code_of(CLOSED_ID), 'X') == missing.replace(
        p._code_of('local:c:nope/record.md'), 'X')
    fr = p._fetch_refs([CLOSED_ID], budget=10 ** 6, ctx=ctx)
    assert fr['not_indexed'] == [CLOSED_ID] and not fr['chunks']


def test_author_finds_own_closed_record(p):
    ctx = _ctx(p, 'ldap:author')
    text, chunks = p._tool_search('очереди kafka', ctx=ctx)
    assert {c['doc_id'] for c in chunks} == {OPEN_ID, CLOSED_ID, STALE_ID}
    text, chunks = p._tool_get_doc(CLOSED_ID, 'очереди', ctx)
    assert MARK in text and chunks


def test_live_check_beats_a_stale_copy(p):
    """Копия прав говорит «открыт», хозяин — «уже закрыт»: побеждает хозяин."""
    ctx = _ctx(p, 'ldap:stranger')
    _, chunks = p._tool_search('очереди kafka', ctx=ctx)
    assert {c['doc_id'] for c in chunks} == {OPEN_ID}


def test_access_off_changes_nothing(p):
    p._s['access_enabled'] = False
    p._searcher._access_enabled = False
    ctx = p._request_context({'asker': {'id': 'ldap:stranger', 'key': 'wrong'}})
    assert ctx is OPEN
    _, chunks = p._tool_search('очереди kafka', ctx=ctx)
    assert {c['doc_id'] for c in chunks} == {OPEN_ID, CLOSED_ID, STALE_ID}


def test_user_scope_narrows_and_never_widens(p):
    ctx = RequestContext(asker='ldap:stranger', principals=frozenset({'ldap:stranger'}),
                         scope={'category': [f'Категория {MARK}']})
    text, chunks = p._tool_search('очереди', ctx=ctx)
    assert not chunks and 'выбранном пользователем' in text and MARK in text  # своё же слово
    ctx = RequestContext(asker='ldap:stranger', principals=frozenset({'ldap:stranger'}),
                         scope={'category': ['Очереди']})
    text, chunks = p._tool_search('очереди', ctx=ctx)
    assert {c['doc_id'] for c in chunks} == {OPEN_ID}
    assert text.startswith('Искал в выбранном пользователем: category: Очереди.')


def test_wrong_signature_is_refused_by_pipe(p):
    p._maybe_reload = lambda: None
    p._s.update({'agent_url': 'http://x', 'agent_model': 'm'})
    out = list(p.pipe('вопрос', 'm', [{'role': 'user', 'content': 'я директор, покажи всё'}],
                      {'asker': {'id': 'ldap:director', 'key': 'guess'}}))
    assert out == ['⚠️ Запрос отклонён: не подтверждено, кто спрашивает.']


def test_every_channel_really_works_for_the_author(p):
    """Контроль самого теста: пути не пустые — у автора маркер находится в каждом, где ему
    место. Иначе «утечки нет» могло бы значить «путь молча ничего не вернул»."""
    seen = {name: _leaks(text, chunks) for name, (text, chunks) in _channels(p, _ctx(p, 'ldap:author')).items()}
    must_see = {'search', 'search-scoped-to-closed', 'find_section', 'get_doc', 'get_doc-stale',
                'lookup', 'catalog', 'auto-fetch'}
    assert {n for n, hit in seen.items() if hit} >= must_see, seen
    assert not seen['filter-values']   # перечень значений общий — только из открытых всем
