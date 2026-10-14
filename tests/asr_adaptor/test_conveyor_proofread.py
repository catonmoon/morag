"""Корректор после распознавания (conveyor/proofread.py): модель предлагает, звук подтверждает, код применяет.

Сценарии — с замера на «Капитанском мостике» 14.10.2026: «материнские оплаты» → «платы» (оба уха
слышат «платы»), замена без свидетеля отбрасывается, порог двух ушей, подозрительное слово остаётся,
когда уши его тоже слышат.
"""
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conveyor.events import Emitter
from conveyor.proofread import batches, locate, run_proofread, suspect_flags, time_of
from conveyor.state import State
from stages import arbitrate as A
from stages.final_round import apply_fixes


def _turns():
    return [
        {'cluster': 'SPEAKER_00', 'speaker': 'Speaker_0', 'start': 0.0, 'end': 10.0, 'chunks': [], 'segments': [],
         'raw': 'У пользователей редко материнские оплаты такого класса стоят.',
         'final': 'У пользователей редко материнские оплаты такого класса стоят.'},
        {'cluster': 'SPEAKER_01', 'speaker': 'Speaker_1', 'start': 10.0, 'end': 20.0, 'chunks': [], 'segments': [],
         'raw': 'Ну блин, это сильно, да.', 'final': 'Ну блин, это сильно, да.'},
    ]


class FakeLLM:
    """Детектор помечает «материнские оплаты»; решатель отдаёт то, что ему сказали в `fix_for`."""

    def __init__(self, fix_for: dict, old_error: str = 'yes'):
        self.fix_for, self.old_error, self.calls = fix_for, old_error, []

    async def complete_json(self, messages, schema=None, schema_name='', max_tokens=None, **kw):
        self.calls.append(schema_name)
        if schema_name == 'proofread_flags':
            return {'flags': [{'quote': 'материнские оплаты', 'guess': 'материнские платы', 'why': 'не вяжется'}]}
        user = messages[-1]['content']
        for key, fix in self.fix_for.items():
            if key in user:
                return {'old_error': self.old_error, 'heard': fix['now'] if fix else '', 'fix': fix, 'why': 'тест'}
        return {'old_error': 'no', 'heard': '', 'fix': None, 'why': 'тест'}


def _deps(tmp_path: Path, ears: dict, llm, **cfg):
    def asr(path, prompt='', model=''):
        return {'text': ears['second' if model else 'clean'], 'segments': []}

    def slice_(wav, a, b, dst):
        Path(dst).write_bytes(b'RIFF')

    base = dict(corpus_desc='тестовой записи', always_terms=[], round_concurrency=2, whisper_slots=1,
                second_model='other-ear', proofread=True, proofread_ears='clean,second', proofread_min_ears=1,
                proofread_pad_s=4.0, proofread_max_per_min=100.0, proofread_suspect='')
    base.update(cfg)
    return SimpleNamespace(cfg=SimpleNamespace(**base), arbitrate_stage=A, apply_fixes=apply_fixes,
                           audio_clients=SimpleNamespace(asr=asr), _slice=slice_,
                           _res=lambda name, slots: asyncio.Semaphore(1), llm=llm,
                           _around=lambda turns, i, n=1: '')


def _state(tmp_path: Path) -> State:
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav=str(tmp_path / 'x.wav'))
    st.audio_sec = 20.0
    st.turns = _turns()
    st.known = ()
    return st


async def test_fix_heard_by_both_ears_is_applied(tmp_path):
    llm = FakeLLM({'«материнские оплаты»': {'was': 'оплаты', 'now': 'платы'}})
    d = _deps(tmp_path, {'clean': 'редко материнские платы такого', 'second': 'материнские платы такого класса'}, llm)
    st = _state(tmp_path)
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    assert 'материнские платы такого' in st.turns[0]['final']
    ok = [r for r in st.round_log if r.get('ok')]
    assert len(ok) == 1 and ok[0]['why'] == 'proofread:clean+second'
    assert st.tm['n_proofread_accepted'] == 1
    assert llm.calls == ['proofread_flags', 'proofread_fix']


async def test_fix_without_witness_is_refused(tmp_path):
    llm = FakeLLM({'«материнские оплаты»': {'was': 'оплаты', 'now': 'платы'}})
    d = _deps(tmp_path, {'clean': 'редко материнские оплаты такого', 'second': 'что-то совсем другое'}, llm)
    st = _state(tmp_path)
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    assert 'оплаты' in st.turns[0]['final']
    assert [r['why'] for r in st.round_log] == ['proofread:no_witness']


async def test_two_ears_required_when_configured(tmp_path):
    llm = FakeLLM({'«материнские оплаты»': {'was': 'оплаты', 'now': 'платы'}})
    d = _deps(tmp_path, {'clean': 'материнские платы', 'second': 'ничего похожего'}, llm, proofread_min_ears=2)
    st = _state(tmp_path)
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    assert 'оплаты' in st.turns[0]['final']
    assert st.round_log[0]['why'] == 'proofread:no_witness'


async def test_suspect_word_stays_when_ears_hear_it_too(tmp_path):
    """Брань из списка подозрительных: уши тоже её слышат — говорящий так сказал, текст не трогаем."""
    llm = FakeLLM({}, old_error='no')
    d = _deps(tmp_path, {'clean': 'ну блин это сильно', 'second': 'ну блин это сильно да'}, llm, proofread_suspect='блин')
    st = _state(tmp_path)
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    assert st.turns[1]['final'] == 'Ну блин, это сильно, да.'
    flagged = [r for r in st.round_log if r['quote'].startswith('блин')]
    assert flagged and flagged[0]['why'] == 'proofread:not_error'


def test_locate_and_time_fall_back_to_token_share():
    turns = _turns()
    loc = locate('материнские оплаты', turns, [0, 1])
    assert loc == (0, 3, 5)
    t0, t1 = time_of(turns[0], 3, 5)
    assert 0.0 < t0 < t1 <= 10.0


def test_batches_and_suspects():
    turns = _turns()
    assert batches(turns, max_chars=10) == [[0], [1]]
    assert [f['quote'] for f in suspect_flags(turns, ['блин'])] == ['блин,']


def test_node_is_off_by_default():
    from conveyor import nodes
    entry = next(e for e in nodes.NODES if e[0] == 'proofread')
    assert entry[2](SimpleNamespace(proofread=False), None) is False
    names = [e[0] for e in nodes.NODES]
    assert names.index('final_round') < names.index('proofread') < names.index('speakers')
