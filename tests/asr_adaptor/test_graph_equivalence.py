"""Золотой тест графа: на одних заглушках `graph.run.run_graph` даёт то же, что `pipeline.run_pipeline`.

Равенство трёх вещей: результат (без таймингов и отпечатка), лента событий и строки прогресса
(вид и поля, без `at`/`sec`), вызовы аудио-бэкенда (какие куски и с какой подсказкой слушали).
Наборы флагов покрывают все условные узлы: переслушивание, арбитраж с воротами читателя и третьим
голосом по требованию, защита известных слов с арбитражем звуком, времена слов декодера.

⚠️ Заглушки — ТЕ ЖЕ, что у конвейера (`pipeline.<имя>`, см. fakes.py): граф берёт помощники
поздним связыванием, и это единственная причина, по которой один тест гоняет обе реализации.
"""
from __future__ import annotations

import json

import pytest

import pipeline
from app import _enriched
from fakes import HINTS
from graph.run import run_graph

CASES = {
    'default': {},
    'word_times': {'word_times': True},
    'relisten': {'relisten': True},
    'second_always': {'second_model': 'other', 'clean_ear': 'always'},
    'reader_demand': {'second_model': 'other', 'arbitrate_gate': 'reader',
                      'clean_ear': 'demand', 'clean_ear_window': 'window30'},
    'protect_final_ear': {'protect_known': True, 'final_ear': True},
}
def _norm_result(r: dict) -> dict:
    r = json.loads(json.dumps(r, ensure_ascii=False, default=str))
    r.pop('env', None)
    r.pop('graph', None)
    r['timing'] = {k: v for k, v in r['timing'].items() if not k.endswith('_s')}
    return r


def _norm_events(items: list) -> tuple[list[str], list[str]]:
    """(последовательность стадий и строк прогресса, мультимножество всех событий без времён)."""
    order, everything = [], []
    for e in items:
        if isinstance(e, str):
            order.append(e)
            continue
        body = {k: v for k, v in e.items() if k not in ('at', 'sec')}
        everything.append(json.dumps(body, ensure_ascii=False, sort_keys=True))
        if e['t'] in ('stage.start', 'stage.end', 'job.meta', 'spk.map'):
            order.append(json.dumps(body, ensure_ascii=False, sort_keys=True))
    return order, sorted(everything)


async def _run(fn, rich, wav) -> tuple[dict, list, list[str]]:
    rich.calls.clear()
    seen: list = []
    r = await fn(str(wav), llm=object(), episode='ep1', title='Демо', hints=HINTS, progress=seen.append)
    calls = sorted(json.dumps(c, sort_keys=True) for c in rich.calls)
    return r, seen, calls


@pytest.mark.parametrize('case', list(CASES))
async def test_graph_equals_pipeline(case, rich, silence, monkeypatch):
    for k, v in CASES[case].items():
        monkeypatch.setattr(pipeline.CFG, k, v)

    legacy, ev_l, calls_l = await _run(pipeline.run_pipeline, rich, silence)
    graph, ev_g, calls_g = await _run(run_graph, rich, silence)

    assert _norm_result(graph) == _norm_result(legacy)
    assert _norm_events(ev_g) == _norm_events(ev_l)
    assert calls_g == calls_l, 'граф слушал не те куски или не с той подсказкой'
    assert _enriched(graph)['segments'] == _enriched(legacy)['segments']
    # набор флагов действительно прошёл своим путём — иначе тест зелёный впустую
    if case == 'relisten':
        assert legacy.get('relisten'), 'переслушивание не сработало'
    if case.startswith(('second', 'reader')):
        assert legacy.get('arbitration'), 'арбитраж не дал решений'
        assert any(d['by'] == 'голосование' for d in legacy['arbitration'])
    if case == 'reader_demand':
        assert legacy['timing']['arbitrate_gate'].get('третий голос'), 'третий голос не звали'
    if case == 'protect_final_ear':
        assert any(f['why'].startswith('known_term') for f in legacy.get('fixes', ())), 'ухо финал-раунда не звали'
    if case == 'word_times':
        assert all(s.get('words') for t in legacy['turns'] for s in t['segments'] if s['text'])


async def test_graph_reports_its_nodes_and_keeps_the_artifact_shape(rich, silence):
    r, _, _ = await _run(run_graph, rich, silence)
    assert r['graph']['nodes'][:3] == ['prepare', 'diarize', 'pass1'] and r['graph']['nodes'][-1] == 'coverage'
    assert 'relisten' not in r['graph']['nodes'] and 'arbitrate' not in r['graph']['nodes']
    assert _enriched(r)['x_enriched']['graph']['nodes'] == r['graph']['nodes']
    legacy, _, _ = await _run(pipeline.run_pipeline, rich, silence)
    assert 'graph' not in _enriched(legacy)['x_enriched'], 'у линейного конвейера артефакт прежний'


def test_graph_takes_every_helper_it_needs_from_the_pipeline():
    from graph.deps import Deps
    assert Deps(None).check() == []
