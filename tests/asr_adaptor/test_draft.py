"""Черновик другой школы (`ASR_DRAFT_MODEL`) и второе ухо из его слов (`ASR_SECOND_FROM_DRAFT`), ADR-0033.

Без ручек всё как было: пасс-1 моделью профиля без времён слов, уши слушают бэкенд, `opening` — пасс-1 → …
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import config
from conveyor import nodes
from conveyor.events import Emitter
from conveyor.loop import Place
from conveyor.places import Slices, arbitrate_tools, draft_text, draft_words
from conveyor.proofread import run_proofread
from conveyor.state import State
from stages import arbitrate as A
from stages import coverage
from stages.final_round import apply_fixes

CPU = {'cpu': config.Lane('cpu', 'cpu', 1, ('http://c/v1/audio/transcriptions',))}
WORDS = [{'word': 'раз', 'start': 10.0, 'end': 10.4}, {'word': 'два.', 'start': 10.6, 'end': 11.0},
         {'word': 'Три', 'start': 12.0, 'end': 12.3}, {'word': 'четыре', 'start': 12.4, 'end': 12.9}]
SEGS = [{'id': 0, 'start': 10.0, 'end': 11.0, 'text': 'раз два.', 'words': WORDS[:2]},
        {'id': 1, 'start': 12.0, 'end': 12.9, 'text': 'Три четыре', 'words': WORDS[2:]}]


# --- пасс-1 ---------------------------------------------------------------------------------------------

def _pass1_deps(calls: list, **cfg):
    def asr(path, prompt='', **kw):
        calls.append(kw)
        return {'text': 'раз два. Три четыре', 'segments': SEGS}

    base = dict(whisper_slots=1, asr_urls=['http://w'], hole_min_s=2.0, lanes={}, model_lanes={})
    base.update(cfg)
    return SimpleNamespace(cfg=SimpleNamespace(**base), audio_clients=SimpleNamespace(asr=asr), coverage=coverage,
                           _res=lambda name, slots: asyncio.Semaphore(1), _emit_draft=lambda emit, segs: None)


def _st(tmp_path: Path) -> State:
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav=str(tmp_path / 'x.wav'))
    st.audio_sec = 30.0
    return st


async def test_pass1_without_draft_model_calls_the_backend_the_old_way(tmp_path):
    calls: list = []
    st = _st(tmp_path)
    await nodes.pass1(st, _pass1_deps(calls), Emitter(None, time.monotonic()))
    assert calls == [{}], 'ни модели, ни времён слов — вызов прежней формы'
    assert st.full_text == 'раз два. Три четыре' and len(st.segs) == 2


async def test_pass1_with_draft_model_asks_its_lane_for_words(tmp_path):
    calls: list = []
    names: list = []

    d = _pass1_deps(calls, draft_model='giga', lanes=CPU, model_lanes={'giga': 'cpu'})
    d._res = lambda name, slots: names.append(name) or asyncio.Semaphore(1)
    st = _st(tmp_path)
    await nodes.pass1(st, d, Emitter(None, time.monotonic()))
    assert calls == [{'model': 'giga', 'words': True}]
    assert names == ['cpu'], 'черновик идёт под ресурсом своей полосы, не whisper'
    assert draft_words(st.segs) == [('раз', 10.0, 10.4), ('два.', 10.6, 11.0), ('Три', 12.0, 12.3), ('четыре', 12.4, 12.9)]


def test_draft_words_is_none_when_the_draft_has_no_word_times():
    assert draft_words([{'start': 0.0, 'end': 5.0, 'text': 'без слов'}]) is None
    assert draft_words([]) is None and draft_words(None) is None


def test_draft_text_takes_words_whose_middle_is_inside_the_window():
    draft = draft_words(SEGS)
    assert draft_text(draft, 9.9, 11.5) == 'раз два.'
    assert draft_text(draft, 10.5, 12.2) == 'два. Три'
    assert draft_text(draft, 20.0, 25.0) == '' and draft_text(None, 0, 1) == ''


# --- второе ухо арбитража из черновика --------------------------------------------------------------------

def _arb_deps(tmp_path: Path, asr_calls: list):
    def asr(path, prompt='', model=''):
        asr_calls.append(model)
        return {'text': 'слышно бэкендом', 'segments': []}

    def slice_(wav, a, b, dst):
        Path(dst).write_bytes(b'RIFF')

    cfg = SimpleNamespace(second_model='giga', whisper_slots=1, asr_urls=['http://w'], lanes={}, model_lanes={},
                          arbitrate_ratio=100.0, arbitrate_latin_guard=False, clean_ear='demand')
    return SimpleNamespace(cfg=cfg, arbitrate_stage=A, audio_clients=SimpleNamespace(asr=asr), _slice=slice_,
                           _res=lambda name, slots: asyncio.Semaphore(1), llm=None)


def _place(a: float, b: float) -> Place:
    return Place('chunk:0', 'arbitrate', item={'start': a, 'end': b, 'raw': 'раз два'}, canon=set(),
                 audio_sec=30.0, neighbours=(a, b), index=0)


async def test_second_ear_from_draft_needs_no_backend_and_costs_no_audio(tmp_path):
    calls: list = []
    d = _arb_deps(tmp_path, calls)
    meter: dict = {}
    reg = arbitrate_tools(_place(9.5, 13.0), d, Slices(d, str(tmp_path), 'x.wav', 'a0'), journal=[], meter=meter,
                          draft=draft_words(SEGS))
    heard = await reg.call('listen', ear='second')
    assert heard['text'] == 'раз два. Три четыре' and heard['from'] == 'draft'
    assert calls == [], 'бэкенд не звали'
    assert meter.get('audio_s', 0.0) == 0.0, 'ухо из черновика звука не стоит'
    clean = await reg.call('listen', ear='clean')
    assert clean['text'] == 'слышно бэкендом' and calls == [''] and meter['audio_s'] == 3.5


async def test_second_ear_falls_back_to_the_backend_when_the_draft_is_silent_there(tmp_path):
    calls: list = []
    d = _arb_deps(tmp_path, calls)
    reg = arbitrate_tools(_place(20.0, 25.0), d, Slices(d, str(tmp_path), 'x.wav', 'a1'), journal=[], meter={},
                          draft=draft_words(SEGS))
    heard = await reg.call('listen', ear='second')
    assert heard['text'] == 'слышно бэкендом' and 'from' not in heard and calls == ['giga']


async def test_without_draft_the_second_ear_listens_as_before(tmp_path):
    calls: list = []
    d = _arb_deps(tmp_path, calls)
    reg = arbitrate_tools(_place(9.5, 13.0), d, Slices(d, str(tmp_path), 'x.wav', 'a2'), journal=[], meter={})
    heard = await reg.call('listen', ear='second')
    assert heard['text'] == 'слышно бэкендом' and calls == ['giga']


# --- корректор: второе ухо из черновика -------------------------------------------------------------------

class _LLM:
    def __init__(self):
        self.users = []

    async def complete_json(self, messages, schema=None, schema_name='', max_tokens=None, **kw):
        if schema_name == 'proofread_flags':
            return {'flags': [{'quote': 'материнские оплаты', 'guess': 'материнские платы', 'why': 'не вяжется'}]}
        self.users.append(messages[-1]['content'])
        return {'old_error': 'yes', 'heard': 'платы', 'fix': {'was': 'оплаты', 'now': 'платы'}, 'why': 'тест'}


async def test_proofread_second_ear_reads_the_draft_and_clean_ear_still_listens(tmp_path):
    models: list = []

    def asr(path, prompt='', model=''):
        models.append(model)
        return {'text': 'редко материнские платы такого', 'segments': []}

    cfg = SimpleNamespace(corpus_desc='тестовой записи', always_terms=[], round_concurrency=2, whisper_slots=1,
                          second_model='giga', proofread=True, proofread_ears='clean,second', proofread_min_ears=1,
                          proofread_pad_s=4.0, proofread_max_per_min=100.0, proofread_suspect='',
                          proofread_clean_alone=False, proofread_max_was_words=3, second_from_draft=True)
    llm = _LLM()
    d = SimpleNamespace(cfg=cfg, arbitrate_stage=A, apply_fixes=apply_fixes, audio_clients=SimpleNamespace(asr=asr),
                        _slice=lambda w, a, b, dst: Path(dst).write_bytes(b'RIFF'),
                        _res=lambda name, slots: asyncio.Semaphore(1), llm=llm, _around=lambda turns, i, n=1: '')
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav=str(tmp_path / 'x.wav'))
    st.audio_sec = 20.0
    st.known = ()
    st.turns = [{'cluster': 'SPEAKER_00', 'speaker': 'Speaker_0', 'start': 0.0, 'end': 10.0, 'chunks': [], 'segments': [],
                 'raw': 'У пользователей редко материнские оплаты такого класса стоят.',
                 'final': 'У пользователей редко материнские оплаты такого класса стоят.'}]
    words = [{'word': w, 'start': 1.0 + 0.5 * i, 'end': 1.3 + 0.5 * i}
             for i, w in enumerate('у пользователей редко материнские платы такого класса стоят'.split())]
    st.segs = [{'id': 0, 'start': 1.0, 'end': 5.0, 'text': ' '.join(w['word'] for w in words), 'words': words}]
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    assert models == [''], 'бэкенд слушало только чистое ухо; второе взято из черновика'
    assert 'УШИ[second] (окно ±4 с): «у пользователей редко материнские платы' in llm.users[-1]
    assert 'материнские платы такого' in st.turns[0]['final']


# --- расписание opening ---------------------------------------------------------------------------------

def test_opening_plan_follows_the_lane_of_the_draft_model():
    assert nodes.opening_plan(SimpleNamespace(draft_model='', lanes={}, model_lanes={})) == 'serial'
    assert nodes.opening_plan(SimpleNamespace(draft_model='giga', lanes=CPU, model_lanes={'giga': 'cpu'})) == 'draft-parallel'
    gpu = {'g2': config.Lane('g2', 'gpu', 1, ('http://g2',))}
    assert nodes.opening_plan(SimpleNamespace(draft_model='giga', lanes=gpu, model_lanes={'giga': 'g2'})) == 'serial', \
        'черновик на GPU-полосе — рядом с диаризацией не кладём (GPU на GPU)'
    assert nodes.opening_plan(SimpleNamespace(draft_model='giga', lanes={}, model_lanes={}, whisper_slots=1, asr_urls=[])) == 'serial'


async def test_opening_runs_the_draft_beside_diarization_only_on_a_non_gpu_lane(monkeypatch, tmp_path):
    order: list = []

    def node(name: str, dt: float):
        async def run(st, d, ev):
            order.append(('start', name))
            await asyncio.sleep(dt)
            order.append(('end', name))
        return run

    monkeypatch.setattr(nodes, 'pass1', node('pass1', 0.05))
    monkeypatch.setattr(nodes, 'diarize', node('diarize', 0.1))
    monkeypatch.setattr(nodes, 'glossary', node('glossary', 0.01))

    async def dsum(text, llm):
        return ''

    st = _st(tmp_path)
    ev = Emitter(None, time.monotonic())
    d = SimpleNamespace(cfg=SimpleNamespace(draft_model='giga', lanes=CPU, model_lanes={'giga': 'cpu'}),
                        prefetch={}, doc_summary=dsum, llm=None)
    await nodes.opening(st, d, ev)
    assert order[:2] == [('start', 'diarize'), ('start', 'pass1')] or order[:2] == [('start', 'pass1'), ('start', 'diarize')]
    assert order.index(('end', 'pass1')) < order.index(('start', 'glossary')) < order.index(('end', 'diarize'))
    assert 'opening_s' in st.tm and 'dsum' in d.prefetch

    order.clear()
    d.cfg = SimpleNamespace(draft_model='', lanes={}, model_lanes={})
    d.prefetch = {}
    await nodes.opening(st, d, ev)
    assert order[:2] == [('start', 'pass1'), ('end', 'pass1')], 'без черновика — прежний порядок: пасс-1 первым'


def test_stage_events_mark_parallel_stages_only_while_the_flag_is_set():
    got: list = []
    ev = Emitter(got.append, time.monotonic())
    ev.stage('pass1')
    ev.parallel = True
    ev.stage('diarize')
    ev.stage_done('diarize', 1.0)
    ev.parallel = False
    ev.stage_done('pass1', 2.0)
    marks = [(e['t'], e['stage'], e.get('parallel')) for e in got if isinstance(e, dict) and e['t'].startswith('stage.')]
    assert marks == [('stage.start', 'pass1', None), ('stage.start', 'diarize', True), ('stage.end', 'diarize', True),
                     ('stage.end', 'pass1', None)]


# --- черновик-ухо: whisper режет, GigaAM слушает рядом и кормит только второе ухо (S2b) -------------------------

async def test_draft_ear_runs_on_its_lane_and_feeds_the_second_ear(tmp_path):
    calls: list = []
    names: list = []

    def asr(path, prompt='', **kw):
        calls.append(kw)
        return {'text': 'раз два. Три четыре', 'segments': SEGS}

    d = SimpleNamespace(cfg=SimpleNamespace(draft_ear_model='giga', whisper_slots=1, asr_urls=['http://w'], lanes=CPU,
                                            model_lanes={'giga': 'cpu'}),
                        audio_clients=SimpleNamespace(asr=asr), _res=lambda name, slots: names.append(name) or asyncio.Semaphore(1))
    st = _st(tmp_path)
    await nodes.draft_ear(st, d, Emitter(None, time.monotonic()))
    assert calls == [{'model': 'giga', 'words': True}] and names == ['cpu']
    assert st.ear_draft == [('раз', 10.0, 10.4), ('два.', 10.6, 11.0), ('Три', 12.0, 12.3), ('четыре', 12.4, 12.9)]
    assert 'draft_ear_s' in st.tm and st.segs == [], 'сегменты пасса-1 (границы кусков) черновик-ухо не трогает'
    from conveyor.places import draft_of
    assert draft_of(st) == st.ear_draft
    st.ear_draft = []
    st.segs = SEGS
    assert draft_of(st) == draft_words(SEGS), 'без черновика-уха — слова пасса-1, если они есть'


async def test_opening_runs_whisper_pass1_beside_the_draft_ear(monkeypatch, tmp_path):
    order: list = []

    def node(name: str, dt: float):
        async def run(st, d, ev):
            order.append(('start', name))
            await asyncio.sleep(dt)
            order.append(('end', name))
        return run

    for name, dt in (('pass1', 0.05), ('draft_ear', 0.08), ('diarize', 0.02), ('glossary', 0.01)):
        monkeypatch.setattr(nodes, name, node(name, dt))

    async def dsum(text, llm):
        return ''

    st = _st(tmp_path)
    d = SimpleNamespace(cfg=SimpleNamespace(draft_model='', draft_ear_model='giga', lanes=CPU, model_lanes={'giga': 'cpu'},
                                            voices_early=False), prefetch={}, doc_summary=dsum, llm=None)
    await nodes.opening(st, d, Emitter(None, time.monotonic()))
    assert order.index(('start', 'draft_ear')) < order.index(('end', 'pass1')), 'черновик-ухо стартует, пока идёт пасс-1'
    assert order.index(('end', 'draft_ear')) < order.index(('start', 'diarize')), 'диаризация — после обоих'
    assert 'opening_s' in st.tm


# --- чистое ухо из черновика пасса-1 (ASR_CLEAN_FROM_DRAFT) ------------------------------------------------------

async def test_pass1_with_clean_from_draft_keeps_words_for_the_ear_but_not_for_chunking(tmp_path):
    calls: list = []

    def asr(path, prompt='', **kw):
        calls.append(kw)
        return {'text': 'раз два. Три четыре', 'segments': [dict(s, words=list(s['words'])) for s in SEGS]}

    d = _pass1_deps(calls, clean_from_draft=True)
    d.audio_clients = SimpleNamespace(asr=asr)
    st = _st(tmp_path)
    await nodes.pass1(st, d, Emitter(None, time.monotonic()))
    assert calls == [{'words': True}]
    assert st.clean_draft == [('раз', 10.0, 10.4), ('два.', 10.6, 11.0), ('Три', 12.0, 12.3), ('четыре', 12.4, 12.9)]
    assert all('words' not in s for s in st.segs), 'разрезу на куски — сегменты, не слова'


async def test_clean_ear_from_draft_needs_no_backend(tmp_path):
    calls: list = []
    d = _arb_deps(tmp_path, calls)
    reg = arbitrate_tools(_place(9.5, 13.0), d, Slices(d, str(tmp_path), 'x.wav', 'a3'), journal=[], meter={},
                          clean_draft=draft_words(SEGS))
    heard = await reg.call('listen', ear='clean')
    assert heard['text'] == 'раз два. Три четыре' and heard['from'] == 'draft' and calls == []
    second = await reg.call('listen', ear='second')
    assert second['text'] == 'слышно бэкендом' and calls == ['giga'], 'второе ухо без своего черновика слушает бэкенд'
