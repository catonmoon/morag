"""Поле чанка из аннотаций на стороне поиска (ADR-0027, этап D): нога RRF по схеме коллекции,
поле доезжает до форматтера, реранкер и блок момента его печатают, документная ветка не меняется."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from morag import shortid
from morag.retrieval.reranker import _format_chunk_item
from morag.retrieval.searcher import HybridSearcher, _point_to_chunk

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline, _annotation_line  # noqa: E402

DOC = 'local:demo/talks/alpha.md'
CODE = shortid.make(DOC)
SCREEN = [{'kind': 'screen', 't0': 498.0, 't1': 517.0, 'sub': 'slide', 'label': 'Слайд 4',
           'title': 'Пирамида', 'text': 'Юнит\nИнтеграционные\nСистемные'},
          {'kind': 'ref', 'at': 558.8, 'to': 517.0, 'quote': 'вот здесь нарисую', 'text': 'стрелка и кролик'}]


# --- searcher -----------------------------------------------------------------------------------

def _searcher(dense_names: set[str], field: str | None = 'screen') -> HybridSearcher:
    s = HybridSearcher.__new__(HybridSearcher)
    s._dense = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2]))
    s._sparse = SimpleNamespace(embed_query=AsyncMock(return_value=([1, 2], [0.5, 0.5])))
    s.get_sparse_vector_names = AsyncMock(return_value={'keywords'})
    s.get_dense_vector_names = AsyncMock(return_value=dense_names)
    s._hnsw_ef = 0
    s._annotation_field = field
    s._extra_fields = (field,) if field else ()
    return s


async def test_leg_only_when_vector_exists_in_schema():
    with_vec = await _searcher({'full', 'screen'})._build_rrf_prefetch('chunks', 'пирамида', 10)
    assert [p.using for p in with_vec if p.using] == ['full', 'screen']   # верхний уровень, свой голос
    assert with_vec[1].query == with_vec[0].query                          # тот же вектор запроса
    old_schema = await _searcher({'full'})._build_rrf_prefetch('chunks', 'пирамида', 10)
    assert [p.using for p in old_schema if p.using] == ['full']
    off = await _searcher({'full', 'screen'}, field=None)._build_rrf_prefetch('chunks', 'пирамида', 10)
    assert [p.using for p in off if p.using] == ['full']


def test_point_to_chunk_carries_the_field_only_when_asked():
    point = {'id': 'p1', 'payload': {'doc_id': DOC, 'text': 'речь', 'screen': SCREEN, 'start_sec': 500.0}, 'score': 0.5}
    assert _point_to_chunk(point, ('screen',))['screen'] == SCREEN
    assert 'screen' not in _point_to_chunk(point)


# --- reranker -----------------------------------------------------------------------------------

def test_reranker_item_shows_annotations_before_text():
    c = {'path': ['alpha.md'], 'doc_id': DOC, 'text': 'речь', 'context': 'справка', 'screen': SCREEN}
    plain = _format_chunk_item(0, c)
    with_ann = _format_chunk_item(0, c, ('screen', 'На экране'))
    assert 'На экране' not in plain
    assert with_ann.splitlines()[2].startswith('На экране: Пирамида')
    assert with_ann.splitlines()[-1] == 'речь' and '«вот здесь нарисую» — стрелка и кролик' in with_ann


# --- блок search ---------------------------------------------------------------------------------

def _block(chunks, field='screen', timestamps=True):
    pipe = object.__new__(Pipeline)
    pipe._s = {'timestamp_citations': timestamps, 'doc_ids_in_results': 'none',
               'annotation_field': field, 'annotation_label': 'На экране'}
    pipe._doc_numbering = {}
    pipe._cite_units = {}
    pipe._live_titles = {}
    pipe._short_maps = lambda: ({CODE: DOC}, {DOC: CODE})
    return pipe._render_chunks_block(chunks)


def _moment(**kw):
    return {'doc_id': DOC, 'start_sec': 500, 'text': 'речь про пирамиду', 'title': 'Доклад',
            'speakers': ['Кузнецова'], 'path': ['alpha.md'], **kw}


def test_annotation_line_shapes():
    assert _annotation_line('На экране', SCREEN[0]) == 'На экране (8:18, Слайд 4 «Пирамида»): Юнит Интеграционные Системные'
    assert _annotation_line('На экране', SCREEN[1]) == 'На экране · указание (9:18, «вот здесь нарисую»): стрелка и кролик'
    assert _annotation_line('На экране', {'kind': 'screen', 'sub': 'app', 'text': 'psql'}) == 'На экране (app): psql'
    assert _annotation_line('На экране', {'kind': 'screen', 'text': 'x'}) == 'На экране: x'


def test_moment_prints_screen_lines_before_speech():
    out = _block([_moment(context='справка', screen=SCREEN)])
    lines = out.splitlines()
    i = lines.index('Контекст: справка')
    assert lines[i + 1].startswith('На экране (8:18, Слайд 4 «Пирамида»)')
    assert lines[i + 2].startswith('На экране · указание (9:18')
    assert lines[i + 3] == 'речь про пирамиду'


def test_moment_without_field_and_field_off_are_unchanged():
    plain = _block([_moment()])
    assert 'На экране' not in plain
    assert _block([_moment(screen=SCREEN)], field=None) == plain


@pytest.mark.parametrize('field', [None, 'screen'])
def test_document_branch_never_prints_annotations(field):
    doc_chunk = {'doc_id': DOC, 'text': 'страница', 'title': 'Страница', 'order': 0, 'path': ['alpha.md'], 'screen': SCREEN}
    out = _block([doc_chunk], field=field, timestamps=False)
    assert 'На экране' not in out and 'страница' in out
