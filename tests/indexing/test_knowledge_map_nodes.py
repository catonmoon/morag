"""Узлы Карты знаний сохраняются по одному — паритет с текстом системного промпта.

Для тула `sections`: каждый id, напечатанный в Карте, получает точку `kind='node'`, чей текст —
подстрока промпта. Отброшенный expandable-проход (H2-fallback) потомков не оставляет.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from morag.indexing.knowledge_map import KnowledgeMapGenerator
from morag.sources.base import Document


def _doc(doc_id: str, title: str, summary: str, parents: list[str] | None = None) -> Document:
    return Document(id=doc_id, path=[doc_id], text='', source_type='confluence', title=title,
                    parent_doc_ids=parents or [],
                    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    payload={'doc_summary': summary, 'short_id': f'S{doc_id.upper()}'})


def _gen(prompt_budget: int, node_min_tokens: int = 4) -> KnowledgeMapGenerator:
    gen = KnowledgeMapGenerator.__new__(KnowledgeMapGenerator)
    gen._exclude_source_types = []
    gen._prompt_strategy = 'weighted'
    gen._prompt_budget = prompt_budget
    gen._node_min_tokens = node_min_tokens
    gen._per_element_min_tokens = 1
    gen._counter = MagicMock(count=lambda t: len(t.split()))
    gen._node_maps = {}
    gen.ensure_collection = AsyncMock()
    gen._save_maps = AsyncMock()
    # LLM-сжатие не нужно: тексты и так короткие — возвращаем как есть.
    gen._compact_until_fits = AsyncMock(side_effect=lambda title, text, budget: text)
    gen._subtree_summary = AsyncMock(side_effect=lambda doc, cm: f'subtree of {doc.title}')
    return gen


ROOT = _doc('root', 'Люди', 'about people')
BIG = _doc('big', 'Команды', 'about teams', ['root'])
LEAF = _doc('leaf', 'Моти', 'about moti', ['big'])
SMALL = _doc('small', 'Наём', 'about hiring', ['root'])


@pytest.mark.asyncio
async def test_every_printed_id_has_a_node_and_text_is_a_slice_of_the_prompt():
    gen = _gen(prompt_budget=10_000)
    gen._load_all_docs = AsyncMock(return_value=[ROOT, BIG, LEAF, SMALL])
    maps = await gen.generate()
    prompt = maps['_system_prompt']

    kwargs = gen._save_maps.call_args.kwargs
    nodes = {doc_id: pl for doc_id, pl in kwargs['extra_points']}
    # Глубина Карты = 2: `big` на второй глубине свёрнут, `leaf` в Карте не напечатан —
    # значит и узла у него нет (паритет: тул не знает того, чего нет в промпте).
    assert set(nodes) == {'root', 'big', 'small'}
    assert 'SLEAF' not in prompt
    for pl in nodes.values():
        assert pl['kind'] == 'node'
        assert pl['map_text'] in prompt
        assert pl['brief'] in pl['map_text']
    assert nodes['root']['depth'] == 1 and nodes['root']['short_id'] == 'SROOT'
    assert nodes['root']['brief'] == '## Люди (id: SROOT)\n\nabout people'
    assert '### Команды' not in nodes['root']['brief']
    assert nodes['big']['map_text'].startswith('### Команды (id: SBIG)')
    assert 'subtree of Команды' in nodes['big']['map_text']


@pytest.mark.asyncio
async def test_brief_listed_child_gets_exactly_its_line():
    # Крошечный бюджет: детям не хватает на раздел → они уходят в перечень строками.
    gen = _gen(prompt_budget=24, node_min_tokens=100)
    gen._load_all_docs = AsyncMock(return_value=[ROOT, BIG, SMALL])
    maps = await gen.generate()
    nodes = {doc_id: pl for doc_id, pl in gen._save_maps.call_args.kwargs['extra_points']}
    assert nodes['small']['map_text'] == '- Наём (id: SSMALL) — about hiring'
    assert nodes['small']['map_text'] in maps['_system_prompt']


@pytest.mark.asyncio
async def test_h2_fallback_drops_descendants_captured_by_the_discarded_pass():
    gen = _gen(prompt_budget=10_000)
    gen._load_all_docs = AsyncMock(return_value=[ROOT, BIG, LEAF, SMALL])
    # Expandable всегда «переполняет» бюджет → collapse; потомков в узлах быть не должно.
    gen._counter = MagicMock(count=lambda t: 10_000 if t.startswith('## Люди') else 1)
    maps = await gen.generate()
    nodes = {doc_id: pl for doc_id, pl in gen._save_maps.call_args.kwargs['extra_points']}
    assert set(nodes) == {'root'}
    assert nodes['root']['map_text'] in maps['_system_prompt']
    assert 'subtree of Люди' in nodes['root']['map_text']
