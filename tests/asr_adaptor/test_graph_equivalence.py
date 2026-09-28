"""Золотой тест графа: на одних заглушках `graph.run.run_graph` даёт то же, что `pipeline.run_pipeline`.

Равенство трёх вещей: результат (без таймингов и отпечатка), лента событий и строки прогресса
(вид и поля, без `at`/`sec`), вызовы аудио-бэкенда (какие куски и с какой подсказкой слушали).
Наборы флагов покрывают все условные узлы: переслушивание, арбитраж с воротами читателя и третьим
голосом по требованию, защита известных слов с арбитражем звуком, времена слов декодера.

⚠️ Заглушки — ТЕ ЖЕ, что у конвейера (`pipeline.<имя>`): граф берёт помощники поздним
связыванием, и это единственная причина, по которой один тест гоняет обе реализации.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

import pipeline
from app import _enriched
from graph.run import run_graph
from test_pipeline_recovery import AUDIO_S, PASS1, SILENT_CHUNK, SR

CASES = {
    'default': {},
    'word_times': {'word_times': True},
    'relisten': {'relisten': True},
    'second_always': {'second_model': 'other', 'clean_ear': 'always'},
    'reader_demand': {'second_model': 'other', 'arbitrate_gate': 'reader',
                      'clean_ear': 'demand', 'clean_ear_window': 'window30'},
    'protect_final_ear': {'protect_known': True, 'final_ear': True},
}
HINTS = {'terms': ['Postgres', 'Kubernetes'], 'names': ['Мария Кузнецова'], 'about': 'демо'}


class RichBackend:
    """Бэкенд, у которого есть что арбитрировать: три слова в куске, второе ухо слышит иначе,
    чистое ухо в середине записи соглашается со вторым (голосование), в конце — петля (переслушивание)."""

    def __init__(self) -> None:
        self.slices: dict[str, tuple[float, float]] = {}
        self.calls: list[dict] = []

    def cut(self, wav, a, b, dst):
        self.slices[dst] = (a, b)

    def asr(self, path: str, prompt: str = '', model: str = '', words: bool = False) -> dict:
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = self.slices[path]
        self.calls.append({'a': round(a, 2), 'b': round(b, 2), 'prompt': prompt, 'model': model, 'words': words})
        if SILENT_CHUNK[0] <= a < SILENT_CHUNK[1] and prompt:
            return {'text': '', 'segments': []}
        # Чистое ухо соглашается со вторым в СЕРЕДИНЕ записи — по середине окна, а не по его
        # началу: окно третьего голоса (`window30`) и ухо финал-раунда начинаются раньше куска,
        # а спорное слово обязано звучать ОДИНАКОВО из любого окна, накрывающего то же место.
        mid = (a + b) / 2
        if model or (not prompt and 20 <= mid < 50):
            text = f'начало печь-{int(mid // 15) * 15} конец'
        elif a >= 50 and prompt:
            text = 'ИИИИИИИИИИИИ петля'
        else:
            text = f'начало речь-{a:.0f} конец'
        seg = {'start': 0.0, 'end': b - a, 'text': text, 'avg_logprob': -0.3}
        if words:
            seg['words'] = [{'word': ' ' + w, 'start': 0.1 * k, 'end': 0.1 * k + 0.05, 'probability': 0.9}
                            for k, w in enumerate(text.split())]
        return {'text': text, 'segments': [seg]}


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """Тишина нужной длины: покрытие считается по заголовку wav (как у теста конвейера)."""
    path = tmp_path / 'source.wav'
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b'\0' * (int(AUDIO_S * SR) * 2))
    return path


@pytest.fixture
def rich(monkeypatch, wav) -> RichBackend:
    fake = RichBackend()
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})   # семафоры чужого цикла событий
    monkeypatch.setattr(pipeline, '_to_wav', lambda src, dst: Path(dst).write_bytes(wav.read_bytes()))
    monkeypatch.setattr(pipeline, '_slice', fake.cut)
    monkeypatch.setattr(pipeline.audio_clients, 'asr', fake.asr)
    monkeypatch.setattr(pipeline.audio_clients, 'diarize',
                        lambda p: [{'start': 0.0, 'end': AUDIO_S, 'speaker': 'SPEAKER_00'}])
    monkeypatch.setattr(pipeline.audio_clients, 'campp', lambda p, spans: ({}, {}))
    monkeypatch.setattr(pipeline.registry, 'assign', lambda *a, **kw: {'SPEAKER_00': 'Speaker_0'})
    monkeypatch.setattr(pipeline.registry, 'names', lambda path: {})

    async def nothing(*a, **kw):
        return []

    async def empty_text(*a, **kw):
        return ''

    async def no_names(*a, **kw):
        return {}, []

    async def reader(llm, text, terms):
        return ['речь-20'] if 'речь-20' in text or 'печь-20' in text else []

    async def correct(text, dsum, ctx, canonicals, llm, always=(), recalled='', corpus_desc='',
                      fixes_out=None, protect=()):
        # Замена известного слова, отвергнутая сторожем: её и переслушивает `final_ear`.
        if fixes_out is not None and 'речь-20' in text:
            fixes_out.append({'was': 'речь-20', 'now': 'печь-20', 'ok': False, 'why': 'known_term'})
        return text

    monkeypatch.setattr(pipeline, 'build_glossary', nothing)
    monkeypatch.setattr(pipeline, 'build_hints', nothing)
    monkeypatch.setattr(pipeline, 'doc_summary', empty_text)
    monkeypatch.setattr(pipeline, 'recall_entities', empty_text)
    monkeypatch.setattr(pipeline, 'correct', correct)
    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: 'речь-20' in raw)
    monkeypatch.setattr(pipeline, 'name_speakers', no_names)
    monkeypatch.setattr(pipeline, 'WhisperTokenCounter', lambda model: None)
    monkeypatch.setattr(pipeline, 'build_prompt',
                        lambda terms, counter, budget, always=(), hinted=(): 'каноники')
    monkeypatch.setattr(pipeline.arbitrate_stage, 'reader_flags', reader)
    return fake


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


async def _run(fn, rich: RichBackend, wav) -> tuple[dict, list, list[str]]:
    rich.calls.clear()
    seen: list = []
    r = await fn(str(wav), llm=object(), episode='ep1', title='Демо', hints=HINTS, progress=seen.append)
    calls = sorted(json.dumps(c, sort_keys=True) for c in rich.calls)
    return r, seen, calls


@pytest.mark.parametrize('case', list(CASES))
async def test_graph_equals_pipeline(case, rich, wav, monkeypatch):
    for k, v in CASES[case].items():
        monkeypatch.setattr(pipeline.CFG, k, v)

    legacy, ev_l, calls_l = await _run(pipeline.run_pipeline, rich, wav)
    graph, ev_g, calls_g = await _run(run_graph, rich, wav)

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


async def test_graph_reports_its_nodes_and_keeps_the_artifact_shape(rich, wav):
    r, _, _ = await _run(run_graph, rich, wav)
    assert r['graph']['nodes'][:3] == ['prepare', 'diarize', 'pass1'] and r['graph']['nodes'][-1] == 'coverage'
    assert 'relisten' not in r['graph']['nodes'] and 'arbitrate' not in r['graph']['nodes']
    assert _enriched(r)['x_enriched']['graph']['nodes'] == r['graph']['nodes']
    legacy, _, _ = await _run(pipeline.run_pipeline, rich, wav)
    assert 'graph' not in _enriched(legacy)['x_enriched'], 'у линейного конвейера артефакт прежний'


def test_graph_takes_every_helper_it_needs_from_the_pipeline():
    from graph.deps import Deps
    assert Deps(None).check() == []
