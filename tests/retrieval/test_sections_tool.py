"""Тул `sections`: описания разделов по запросу из узлов Карты, сохранённых по одному."""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from morag.retrieval.searcher import HybridSearcher
from morag.retrieval.tools import REGISTRY, build_tool_schema
from morag.retrieval.tools.core import CORE_EXECUTION_METHODOLOGY

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402

# Коды — настоящие (`shortid.assign`, с контрольным знаком): испорченный код резолвер отвергает.
PEOPLE = 'confluence:c:100'
TEAMS = 'confluence:c:200'
HIRING = 'confluence:c:300'
PROMPT = (
    '# Карта документации\n\n## Люди (id: S3P0NFD)\n\nПро людей.\n\n'
    '### Команды (id: S8CWHFQ)\n\nПро команды.\n\n- Наём (id: DNPGGAF) — про наём\n'
)
NODES = [
    (PEOPLE, {'short_id': 'S3P0NFD', 'title': 'Люди', 'depth': 1,
              'map_text': PROMPT[len('# Карта документации\n\n'):],
              'brief': '## Люди (id: S3P0NFD)\n\nПро людей.'}),
    (TEAMS, {'short_id': 'S8CWHFQ', 'title': 'Команды', 'depth': 2,
             'map_text': '### Команды (id: S8CWHFQ)\n\nПро команды.\n\n- Наём (id: DNPGGAF) — про наём\n',
             'brief': '### Команды (id: S8CWHFQ)\n\nПро команды.'}),
    (HIRING, {'short_id': 'DNPGGAF', 'title': 'Наём', 'depth': 3,
              'map_text': '- Наём (id: DNPGGAF) — про наём',
              'brief': '- Наём (id: DNPGGAF) — про наём'}),
]


async def _fill(client: AsyncQdrantClient, with_nodes: bool) -> None:
    await client.create_collection('km', vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    points = [PointStruct(id=0, vector=[0.0], payload={
        'doc_id': '_system_prompt', 'map_text': PROMPT, 'kind': 'system_prompt'})]
    if with_nodes:
        points += [PointStruct(id=i + 1, vector=[0.0], payload={'doc_id': d, 'kind': 'node', **pl})
                   for i, (d, pl) in enumerate(NODES)]
    await client.upsert('km', points)


def _pipeline(with_nodes: bool = True) -> Pipeline:
    pl = object.__new__(Pipeline)
    pl._loop = asyncio.new_event_loop()
    pl._loop_lock = threading.Lock()
    client = AsyncQdrantClient(location=':memory:')
    pl._loop.run_until_complete(_fill(client, with_nodes))
    pl._searcher = HybridSearcher(
        qdrant=client, dense_embedder=None, sparse_embedder=None,
        chunks_collection='chunks', docs_collection='docs', knowledge_map_collection='km')
    short = {'S3P0NFD': PEOPLE, 'S8CWHFQ': TEAMS, 'DNPGGAF': HIRING}
    pl._searcher.fetch_short_id_maps = AsyncMock(
        return_value=(short, {v: k for k, v in short.items()}))
    spec = REGISTRY['sections']
    pl._tool_instances = {'sections': (spec, {'type': 'sections'})}
    return pl


def test_listing_gives_roots_only_in_map_order():
    text, chunks = _pipeline()._execute_tool('sections', {})
    assert chunks == []
    assert 'Верхнеуровневые разделы базы знаний — 1' in text
    assert '## Люди (id: S3P0NFD)\n\nПро людей.' in text
    assert 'Про команды' not in text and 'DNPGGAF' not in text


def test_section_by_code_returns_the_node_text_with_children():
    text, _ = _pipeline()._execute_tool('sections', {'section_id': 'S8CWHFQ'})
    assert text.startswith('### Команды (id: S8CWHFQ)')
    assert '- Наём (id: DNPGGAF) — про наём' in text


def test_long_doc_id_is_accepted_too():
    text, _ = _pipeline()._execute_tool('sections', {'section_id': TEAMS})
    assert text.startswith('### Команды (id: S8CWHFQ)')


def test_brief_child_gets_exactly_its_line():
    text, _ = _pipeline()._execute_tool('sections', {'section_id': 'DNPGGAF'})
    assert text == '- Наём (id: DNPGGAF) — про наём'


def test_unknown_but_valid_code_is_refused_without_an_exception():
    pl = _pipeline()
    pl._searcher.fetch_short_id_maps = AsyncMock(return_value=({'S3P0NFD': PEOPLE}, {PEOPLE: 'S3P0NFD'}))
    text, _ = pl._execute_tool('sections', {'section_id': 'S8CWHFQ'})
    assert 'такого документа в корпусе нет' in text and 'S8CWHFQ' in text


def test_section_not_in_map_points_back_to_search():
    pl = _pipeline()
    pl._searcher.fetch_short_id_maps = AsyncMock(return_value=(
        {'S3P0NFD': PEOPLE, 'SWMKDMY': 'confluence:c:999'}, {PEOPLE: 'S3P0NFD', 'confluence:c:999': 'SWMKDMY'}))
    text, _ = pl._execute_tool('sections', {'section_id': 'SWMKDMY'})
    assert 'не описан' in text and "section_ids=['SWMKDMY']" in text


def test_empty_collection_sends_agent_to_find_section():
    text, _ = _pipeline(with_nodes=False)._execute_tool('sections', {})
    assert 'Карта знаний не построена' in text


def test_schema_has_optional_section_id_and_a_switchable_prompt_hint():
    spec = REGISTRY['sections']
    schema = build_tool_schema(spec, {'type': 'sections'})
    assert schema['function']['name'] == 'sections'
    assert schema['function']['parameters']['required'] == []
    assert 'section_id' in schema['function']['parameters']['properties']
    assert '`sections()`' in spec.prompt_section({'type': 'sections'})
    assert '`map()`' in spec.prompt_section({'type': 'sections', 'name': 'map'})
    assert spec.prompt_section({'type': 'sections', 'hint': False}) == ''
    assert build_tool_schema(spec, {'type': 'sections', 'description': 'Своё.'})['function']['description'] == 'Своё.'


def test_methodology_override_keeps_tool_prompt_sections():
    inst = {'sections': (REGISTRY['sections'], {'type': 'sections'})}
    text = Pipeline._tool_methodology('МОЯ МЕТОДИКА.', inst)
    assert text.startswith('МОЯ МЕТОДИКА.') and '`sections()`' in text
    assert Pipeline._tool_methodology('', {}) == CORE_EXECUTION_METHODOLOGY
