"""Сужение `search(section_ids=…, doc_ids=…)` — Qdrant-фильтр, а не пост-фильтр (ADR-0028).

До этого сужение накладывалось В ПИТОНЕ на безфильтровый top-N: чанки нужного документа обязаны
были сами попасть в общую выборку, иначе сужение молча снималось, и агент получал выдачу по всему
корпусу с пометкой. Здесь проверяется, что (а) doc_ids уезжают в `_search` фильтром по `doc_id`,
(б) section_ids раскрываются в потомков и едут тем же фильтром, (в) фильтр агента и сужение
складываются, (г) пустая выдача с сужением — отказ, а не второй поиск по корпусу, и (д) подсказка
автозагрузки перекрывается конфигом, а без него остаётся прежней.
"""
from __future__ import annotations

import sys
from pathlib import Path

from morag.config import RetrievalPromptsConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import _AUTO_FETCH_NOTE, Pipeline  # noqa: E402


def _chunk(doc_id: str, n: int) -> dict:
    return {'chunk_id': f'{doc_id}#{n}', 'doc_id': doc_id, 'text': f'текст {n}',
            'score': 1.0 / (n + 1), 'source_type': 'local'}


def _pipeline(chunks: list[dict], descendants: set[str] | None = None,
              fields: list[str] | None = None, values: dict | None = None):
    p = object.__new__(Pipeline)
    p._s = {'search_limit': 50, 'unique_docs_cap': 10, 'search_filters': []}
    calls: list[dict] = []

    def _search(query, limit, scope_active=False, filters=None):
        calls.append({'query': query, 'limit': limit, 'scope_active': scope_active,
                      'filters': dict(filters or {})})
        return list(chunks)

    p._search = _search
    p._rerank = lambda query, cs: list(cs)
    p._get_descendant_doc_ids = lambda ids: set(descendants or [])
    p._render_chunks_block = lambda cs, header=None: f'{len(cs)} фрагментов'
    p._filter_fields = lambda: list(fields or [])
    p._filter_values = lambda: dict(values or {})
    return p, calls


def test_doc_ids_travel_as_qdrant_filter():
    p, calls = _pipeline([_chunk('local:x:a.md', 1), _chunk('local:x:a.md', 2)])
    text, out = p._tool_search('kafka', doc_ids=['local:x:a.md'])
    assert calls[0]['filters'] == {'doc_id': ['local:x:a.md']}
    assert calls[0]['scope_active'] is True
    assert len(out) == 2 and '2 фрагментов' in text


def test_section_ids_expand_to_descendants_in_the_same_filter():
    p, calls = _pipeline([_chunk('d2', 1)], descendants={'d2', 'd1'})
    p._tool_search('kafka', section_ids=['s1'])
    assert calls[0]['filters'] == {'doc_id': ['d1', 'd2']}, 'потомки раздела — в фильтр, отсортированы'


def test_agent_filters_and_scope_merge_into_one_filter():
    p, calls = _pipeline([_chunk('a', 1)], fields=['year'], values={'year': ['2024']})
    p._tool_search('kafka', doc_ids=['a'], filters={'year': ['2024']})
    assert calls[0]['filters'] == {'year': ['2024'], 'doc_id': ['a']}


def test_empty_scoped_result_is_a_refusal_not_a_corpus_search():
    p, calls = _pipeline([])
    text, out = p._tool_search('kafka', doc_ids=['local:x:nope.md'])
    assert out == []
    assert 'Сужение' in text and 'без section_ids/doc_ids' in text
    assert len(calls) == 1, 'молчаливого второго поиска по всему корпусу быть не должно'


def test_unscoped_search_sends_no_doc_filter():
    p, calls = _pipeline([_chunk('a', 1)])
    p._tool_search('kafka')
    assert calls[0]['filters'] == {} and calls[0]['scope_active'] is False


def test_auto_fetch_note_default_and_override():
    p = object.__new__(Pipeline)
    p._s = {}
    assert p._auto_fetch_note() == _AUTO_FETCH_NOTE
    p._s = {'auto_fetch_note': '  '}
    assert p._auto_fetch_note() == _AUTO_FETCH_NOTE, 'пробелы — это «не задано»'
    p._s = {'auto_fetch_note': 'Запись загружена целиком — отвечай по ней.'}
    assert p._auto_fetch_note() == 'Запись загружена целиком — отвечай по ней.'


def test_prompts_config_carries_auto_fetch_note():
    assert RetrievalPromptsConfig().auto_fetch_note == ''
    assert RetrievalPromptsConfig(auto_fetch_note='своё').auto_fetch_note == 'своё'
