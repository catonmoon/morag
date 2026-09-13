"""Фильтр `search(filters=…)` по полям payload — Qdrant-фильтр внутри гибридного поиска.

До него агент умел сужать поиск только `section_ids`/`doc_ids`, и это был пост-фильтр по
безфильтровому top-N: раздел помогал, только если его чанки и так попали в общую выборку.
Здесь проверяется, что фильтр (а) уезжает в каждый Prefetch, (б) значения для схемы
инструмента собираются из коллекции документов, (в) неверный аргумент агента снимается с
объяснением, а не роняет поиск, (г) схема инструмента показывает допустимые значения.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny

from morag.config import RetrievalSearchConfig
from morag.retrieval.searcher import HybridSearcher, build_payload_filter
from morag.retrieval.tools.core import _search_status

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402


# --- searcher ---------------------------------------------------------------


def test_payload_filter_merges_exclusions_and_agent_filters():
    f = build_payload_filter(['hidden-src'], {'category': ['Базы данных'], 'year': ['2024', '2025']})
    assert isinstance(f, Filter)
    must = {c.key: c.match.any for c in f.must}
    assert must == {'category': ['Базы данных'], 'year': ['2024', '2025']}
    assert f.must_not[0].key == 'source_name' and f.must_not[0].match.any == ['hidden-src']


def test_payload_filter_is_none_when_nothing_to_filter():
    assert build_payload_filter(None, None) is None
    assert build_payload_filter([], {'category': []}) is None


@pytest.mark.asyncio
async def test_filter_reaches_every_prefetch():
    s = HybridSearcher.__new__(HybridSearcher)
    s._dense = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2]))
    s._sparse = SimpleNamespace(embed_query=AsyncMock(return_value=([1, 2], [0.5, 0.5])))
    s.get_sparse_vector_names = AsyncMock(return_value={'keywords'})
    s._hnsw_ef = 0
    prefetch = await s._build_rrf_prefetch('chunks', 'kafka', 10, filters={'category': ['Очереди']})
    dense = prefetch[0]
    lexical = prefetch[1].prefetch
    for pf in [dense, *lexical]:
        assert isinstance(pf.filter, Filter)
        assert pf.filter.must[0].key == 'category' and pf.filter.must[0].match.any == ['Очереди']


@pytest.mark.asyncio
async def test_filter_values_come_from_documents_and_skip_structural():
    s = HybridSearcher.__new__(HybridSearcher)
    s._docs_collection = 'docs'
    s._cache_ttl = 0
    s._cache_expires_at = 0.0
    s._filter_values = {}
    points = [
        SimpleNamespace(payload={'category': 'Очереди', 'topics': ['Kafka', 'Redis']}),
        SimpleNamespace(payload={'category': 'Базы данных', 'topics': ['Redis']}),
        SimpleNamespace(payload={'structural': True, 'category': 'Папка'}),
        SimpleNamespace(payload={'category': ''}),
    ]
    s._qdrant = SimpleNamespace(scroll=AsyncMock(return_value=(points, None)))
    values = await s.filter_values(['category', 'topics'])
    assert values == {'category': ['Базы данных', 'Очереди'], 'topics': ['Kafka', 'Redis']}
    # повторный вызов — из кэша, без похода в Qdrant
    await s.filter_values(['category'])
    assert s._qdrant.scroll.await_count == 1


# --- pipeline -----------------------------------------------------------------


def _pipeline(values: dict[str, list[str]] | None = None, fields: list[dict] | None = None) -> Pipeline:
    p = object.__new__(Pipeline)
    p._s = {'search_filters': fields if fields is not None else [
        {'field': 'category', 'description': 'Категория записи', 'enum': True, 'max_values': 60},
        {'field': 'topics', 'description': 'Темы', 'enum': True, 'max_values': 60},
        {'field': 'year', 'description': 'Год', 'enum': False, 'max_values': 60},
    ]}
    p._searcher = SimpleNamespace(filter_values=AsyncMock(return_value=values or {}))
    p._run = lambda coro: __import__('asyncio').run(coro)
    p._tools = [{'type': 'function', 'function': {'name': 'search', 'description': '', 'parameters': {
        'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}]
    return p


def test_check_filters_drops_unknown_field_and_value_with_explanation():
    p = _pipeline({'category': ['Базы данных', 'Очереди'], 'topics': ['Kafka']})
    clean, notes = p._check_filters({'category': ['очереди', 'Сети'], 'colour': ['red'], 'year': '2024'})
    assert clean == {'category': ['Очереди'], 'year': ['2024']}, 'регистр прощаем, неизвестное снимаем'
    assert any('colour' in n and 'category' in n for n in notes), 'незнакомое поле — с перечнем доступных'
    assert any('Сети' in n and 'Базы данных' in n for n in notes), 'незнакомое значение — с перечнем допустимых'


def test_check_filters_is_noop_without_config():
    p = _pipeline(fields=[])
    assert p._check_filters({'category': ['x']}) == ({}, [])


def test_tool_schema_carries_enum_values_for_small_fields_only():
    p = _pipeline({'category': ['Базы данных', 'Очереди'], 'topics': [f't{i}' for i in range(80)]})
    tools = p._tools_for_request()
    props = tools[0]['function']['parameters']['properties']['filters']['properties']
    assert props['category']['items']['enum'] == ['Базы данных', 'Очереди']
    assert 'enum' not in props['topics']['items'] and 'Примеры' in props['topics']['description']
    assert 'enum' not in props['year']['items']
    # шаблон реестра не тронут
    assert 'filters' not in p._tools[0]['function']['parameters']['properties']


def test_tools_unchanged_without_filters():
    p = _pipeline(fields=[])
    assert p._tools_for_request() is p._tools


def test_config_parses_filters():
    cfg = RetrievalSearchConfig(filters=[{'field': 'category'}, {'field': 'year', 'enum': False}])
    assert [f.field for f in cfg.filters] == ['category', 'year'] and cfg.filters[1].enum is False
    assert RetrievalSearchConfig().filters == []


def test_status_line_shows_the_filter():
    """Посетитель видит в ленте ходов, что поиск сужен, — иначе ответ «за один год» выглядит
    как неполный поиск."""
    line = _search_status({}, {'query': 'kafka', 'filters': {'year': ['2024'], 'category': ['Очереди']}},
                          lambda x: x)
    assert line == '[kafka] фильтр year: 2024; category: Очереди'
    assert _search_status({}, {'query': 'kafka', 'filters': {}}, lambda x: x) == '[kafka] по всей базе'
