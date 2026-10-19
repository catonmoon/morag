"""ADR-0034: условие доступа и «кто спрашивает».

Проверяется: (а) без доступа фильтр прежний байт в байт; (б) обязательные условия уезжают в
каждый Prefetch вместе с сужением агента и не перезаписываются им; (в) питоновское `visible`
совпадает с тем, что отбирает Qdrant; (г) разбор `asker`: подпись, аноним, scope, хозяин прав.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from morag.access import (
    ACCESS_FIELD,
    OPEN,
    AccessRefused,
    RequestContext,
    access_conditions,
    parse_asker,
    scope_conditions,
    visible,
)
from morag.retrieval.searcher import HybridSearcher, build_payload_filter


# --- фильтр -------------------------------------------------------------------


def test_without_access_the_filter_is_byte_for_byte_the_old_one():
    for excl, flt in [(None, None), (['h'], None), (None, {'year': ['2024']}), (['h'], {'year': ['1']})]:
        assert build_payload_filter(excl, flt) == build_payload_filter(excl, flt, None)
        assert build_payload_filter(excl, flt) == build_payload_filter(excl, flt, [])
    assert access_conditions(None) == []           # доступ выключен — условий нет
    assert build_payload_filter(None, None, access_conditions(OPEN.principals)) is None


def test_required_conditions_join_agent_filters_with_and():
    req = access_conditions(frozenset({'ldap:kovaleva'})) + scope_conditions({'year': ['2024']})
    f = build_payload_filter(None, {'doc_id': ['d1'], 'year': ['2023', '2024']}, req)
    assert len(f.must) == 4       # doc_id и год агента + доступ + год человека — все через И
    keys = [getattr(c, 'key', None) for c in f.must]
    assert keys.count('year') == 2   # агент сужает только внутри выбранного человеком


def test_anonymous_sees_only_unlabeled_and_labels_are_intersected():
    assert len(access_conditions(frozenset())) == 1       # только «метки нет»
    cond = access_conditions(frozenset({'a', 'b', 'zzz'}), known_labels={'a', 'b', 'c'})[0]
    assert cond.should[1].match.any == ['a', 'b']          # лишние принципалы в условие не идут
    only_empty = access_conditions(frozenset({'zzz'}), known_labels={'a'})
    assert len(only_empty) == 1 and not isinstance(only_empty[0], Filter)


@pytest.mark.asyncio
async def test_required_reaches_every_prefetch_next_to_agent_doc_scope():
    s = HybridSearcher.__new__(HybridSearcher)
    s._dense = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2]))
    s._sparse = SimpleNamespace(embed_query=AsyncMock(return_value=([1, 2], [0.5, 0.5])))
    s.get_sparse_vector_names = AsyncMock(return_value={'keywords', 'bm25'})
    s._hnsw_ef = 0
    s._sparse_query = True
    req = access_conditions(frozenset({'u1'}))
    prefetch = await s._build_rrf_prefetch('chunks', 'kafka', 10,
                                           filters={'doc_id': ['d1']}, required=req)
    for pf in [prefetch[0], *prefetch[1].prefetch]:
        assert pf.filter.must[0].key == 'doc_id'
        assert pf.filter.must[1] == req[0]


def test_visible_matches_qdrant():
    """Паритет: питоновская проверка точечных выдач ≡ условие поиска."""
    client = QdrantClient(':memory:')
    client.create_collection('docs', vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    labels = {1: None, 2: [], 3: ['ldap:kovaleva'], 4: ['rec:42', 'team:x'], 5: ['nobody']}
    client.upsert('docs', [
        PointStruct(id=i, vector=[1.0, 0.0], payload={} if lab is None else {ACCESS_FIELD: lab})
        for i, lab in labels.items()
    ])
    for principals in [frozenset(), frozenset({'ldap:kovaleva'}), frozenset({'team:x', 'zzz'})]:
        pts, _ = client.scroll('docs', scroll_filter=Filter(must=access_conditions(principals)),
                               limit=100)
        via_qdrant = {p.id for p in pts}
        via_python = {i for i, lab in labels.items()
                      if visible({} if lab is None else {ACCESS_FIELD: lab}, principals)}
        assert via_qdrant == via_python


# --- «кто спрашивает» ---------------------------------------------------------


def test_disabled_ignores_asker_completely():
    body = {'asker': {'id': 'ldap:x', 'key': 'wrong'}}
    assert parse_asker(body, enabled=False, secret='s') is OPEN


def test_no_asker_is_anonymous():
    ctx = parse_asker({}, enabled=True, secret='s')
    assert ctx.active and ctx.principals == frozenset() and ctx.asker is None


@pytest.mark.parametrize('asker', [
    {'id': 'ldap:director', 'key': 'guess'},
    {'id': 'ldap:director'},
    {'id': 'ldap:director', 'key': ''},
])
def test_bad_or_missing_key_is_refused(asker):
    with pytest.raises(AccessRefused):
        parse_asker({'asker': asker}, enabled=True, secret='s3cret')


def test_unset_secret_refuses_everyone():
    with pytest.raises(AccessRefused):
        parse_asker({'asker': {'id': 'ldap:x', 'key': ''}}, enabled=True, secret='')


class _Owner:
    def __init__(self, table):
        self.table = table

    def principals(self, asker):
        return frozenset(self.table.get(asker, ()))


def test_principals_come_from_rights_owner_and_include_id():
    owner = _Owner({'ldap:kovaleva': {'rec:42', 'team:payments'}})
    ctx = parse_asker({'asker': {'id': 'ldap:kovaleva', 'key': 'k', 'user': {'dept': 'X'}}},
                      enabled=True, secret='k', owner=owner)
    assert ctx.principals == {'ldap:kovaleva', 'rec:42', 'team:payments'}
    assert ctx.user == {'dept': 'X'}


def test_account_source_is_part_of_identity():
    owner = _Owner({'ldap:kovaleva': {'rec:42'}})
    ctx = parse_asker({'asker': {'id': 'local:kovaleva', 'key': 'k'}},
                      enabled=True, secret='k', owner=owner)
    assert 'rec:42' not in ctx.principals
    assert not visible({ACCESS_FIELD: ['ldap:kovaleva']}, ctx.principals)


def test_scope_keeps_only_known_fields():
    ctx = parse_asker({'asker': {'id': 'u', 'key': 'k',
                                 'scope': {'year': ['2024'], ACCESS_FIELD: ['x'], 'bogus': ['1']}}},
                      enabled=True, secret='k', scope_fields=['year', 'category'])
    assert ctx.scope == {'year': ['2024']}


def test_request_context_defaults():
    assert not OPEN.active
    assert RequestContext(principals=frozenset()).active
