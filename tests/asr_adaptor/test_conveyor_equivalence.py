"""Золотой тест конвейера: на заглушках `conveyor.run.run_conveyor` даёт то же, что давал прежний
линейный `pipeline.run_pipeline` — его выход заморожен в `golden/<случай>.json` перед удалением (29.09).

Равенство трёх вещей: результат (без таймингов и отпечатка), лента событий и строки прогресса
(вид и поля, без `at`/`sec`), вызовы аудио-бэкенда (какие куски и с какой подсказкой слушали).
Наборы флагов покрывают все условные узлы: переслушивание, арбитраж с воротами читателя и третьим
голосом по требованию, защита известных слов с арбитражем звуком, времена слов декодера.

⚠️ Снимок — договор, а не мнение: правка, меняющая выход конвейера на заглушках, обязана либо
остаться под флагом (выключено — снимок тот же), либо обновить снимок осознанно, с объяснением в
коммите. Обновить: `pytest tests/asr_adaptor/test_conveyor_equivalence.py --update-golden`
(флаг — в conftest.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline
from app import _enriched
from fakes import HINTS
from conveyor.run import run_conveyor

GOLDEN = Path(__file__).parent / 'golden'

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
    r.pop('conveyor', None)
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


def snapshot(r: dict, ev: list, calls: list[str]) -> dict:
    order, everything = _norm_events(ev)
    snap = {'result': _norm_result(r), 'order': order, 'events': everything, 'calls': calls,
            'segments': _enriched(r)['segments']}
    return json.loads(json.dumps(snap, ensure_ascii=False, sort_keys=True, default=str))


@pytest.mark.parametrize('case', list(CASES))
async def test_conveyor_equals_the_frozen_linear_pipeline(case, rich, silence, monkeypatch, request):
    for k, v in CASES[case].items():
        monkeypatch.setattr(pipeline.CFG, k, v)
    r, ev, calls = await _run(run_conveyor, rich, silence)
    got = snapshot(r, ev, calls)
    path = GOLDEN / f'{case}.json'
    if request.config.getoption('--update-golden'):
        path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    want = json.loads(path.read_text())
    assert got['result'] == want['result']
    assert got['order'] == want['order'] and got['events'] == want['events']
    assert got['calls'] == want['calls'], 'конвейер слушал не те куски или не с той подсказкой'
    assert got['segments'] == want['segments']
    # набор флагов действительно прошёл своим путём — иначе тест зелёный впустую
    if case == 'relisten':
        assert r.get('relisten'), 'переслушивание не сработало'
    if case.startswith(('second', 'reader')):
        assert r.get('arbitration'), 'арбитраж не дал решений'
        assert any(d['by'] == 'голосование' for d in r['arbitration'])
    if case == 'reader_demand':
        assert r['timing']['arbitrate_gate'].get('третий голос'), 'третий голос не звали'
    if case == 'protect_final_ear':
        assert any(f['why'].startswith('known_term') for f in r.get('fixes', ())), 'ухо финал-раунда не звали'
    if case == 'word_times':
        assert all(s.get('words') for t in r['turns'] for s in t['segments'] if s['text'])


async def test_conveyor_reports_its_nodes_in_the_artifact(rich, silence):
    r, _, _ = await _run(run_conveyor, rich, silence)
    assert r['conveyor']['nodes'][:3] == ['prepare', 'diarize', 'pass1'] and r['conveyor']['nodes'][-1] == 'coverage'
    assert 'relisten' not in r['conveyor']['nodes'] and 'arbitrate' not in r['conveyor']['nodes']
    assert _enriched(r)['x_enriched']['conveyor']['nodes'] == r['conveyor']['nodes']


def test_conveyor_takes_every_helper_it_needs_from_the_pipeline_module():
    from conveyor.deps import Deps
    assert Deps(None).check() == []
