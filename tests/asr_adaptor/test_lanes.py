"""Полосы: модель → свой процесс бэкенда и свой семафор (`config.Lane`, `lane_for`); без `ASR_LANES` — прежний пул.

Зачем полосы: GigaAM (torch, CPU) в одном процессе с mlx-whisper стоит в той же очереди инференса — второе ухо
на CPU ничего не выигрывало (16.10: корректор 598 с на пяти записях — очередь двух ушей, не LLM).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import audio_clients
import config
from conveyor.events import Emitter
from conveyor.proofread import run_proofread
from conveyor.state import State
from stages import arbitrate as A
from stages.final_round import apply_fixes


def test_no_lanes_means_the_old_whisper_pool(monkeypatch):
    monkeypatch.delenv('ASR_LANES', raising=False)
    monkeypatch.delenv('ASR_MODEL_LANES', raising=False)
    cfg = config.Config()
    ln = config.lane_for(cfg, 'any-model')
    assert (ln.name, ln.device, ln.slots, list(ln.urls)) == ('whisper', 'gpu', cfg.whisper_slots, cfg.asr_urls)
    assert cfg.lanes == {} and cfg.model_lanes == {}


def test_lane_spec_and_model_map(monkeypatch):
    monkeypatch.setenv('ASR_LANES', 'cpu=cpu,1,http://127.0.0.1:8223; '
                                    'far=remote,2,http://gpu-box:9000/v1/audio/transcriptions,http://gpu-box:9010')
    monkeypatch.setenv('ASR_MODEL_LANES', 'gigaam-v3-e2e-rnnt=cpu, big=far')
    cfg = config.Config()
    assert set(cfg.lanes) == {'cpu', 'far'}
    assert cfg.lanes['cpu'].urls == ('http://127.0.0.1:8223/v1/audio/transcriptions',), 'путь дописывается к базе'
    assert cfg.lanes['far'].slots == 2 and len(cfg.lanes['far'].urls) == 2
    assert config.lane_for(cfg, 'gigaam-v3-e2e-rnnt').name == 'cpu'
    assert config.lane_for(cfg, 'big').device == 'remote'
    assert config.lane_for(cfg, cfg.asr_model).name == 'whisper', 'модель без записи — пул whisper'
    assert config.lane_for(cfg).name == 'whisper'


@pytest.mark.parametrize('lanes, models, msg', [
    ('cpu=cpu,1', '', 'нужно имя=устройство'),
    ('cpu=ane,1,http://x', '', 'устройство'),
    ('whisper=gpu,1,http://x', '', 'прежними ручками'),
    ('cpu=cpu,1,http://x', 'm=gpu2', 'не описана'),
])
def test_bad_specs_refuse_loudly(monkeypatch, lanes, models, msg):
    monkeypatch.setenv('ASR_LANES', lanes)
    monkeypatch.setenv('ASR_MODEL_LANES', models)
    with pytest.raises(ValueError, match=msg):
        config.Config()


def test_lane_for_works_on_a_bare_namespace():
    """cfg узлов в тестах — SimpleNamespace без полей полос: ответ — прежний ресурс whisper."""
    ln = config.lane_for(SimpleNamespace(whisper_slots=3, asr_urls=['u']), 'x')
    assert (ln.name, ln.slots, ln.urls) == ('whisper', 3, ('u',))


def test_request_of_a_laned_model_goes_to_its_own_process(monkeypatch, tmp_path):
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    seen = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {'text': 'ок', 'segments': []}

    monkeypatch.setattr(audio_clients.requests, 'post', lambda url, **kw: seen.append((url, kw['data']['model'])) or R())
    monkeypatch.setattr(audio_clients.CFG, 'asr_urls', ['http://w/v1/audio/transcriptions'])
    monkeypatch.setattr(audio_clients.CFG, 'asr_model', 'turbo')
    monkeypatch.setattr(audio_clients.CFG, 'lanes',
                        {'cpu': config.Lane('cpu', 'cpu', 1, ('http://c/v1/audio/transcriptions',))})
    monkeypatch.setattr(audio_clients.CFG, 'model_lanes', {'giga': 'cpu'})
    monkeypatch.setattr(audio_clients, '_INFLIGHT', {})
    audio_clients.asr(str(wav), '')
    audio_clients.asr(str(wav), '', model='giga')
    assert seen == [('http://w/v1/audio/transcriptions', 'turbo'), ('http://c/v1/audio/transcriptions', 'giga')]
    assert set(audio_clients._INFLIGHT.values()) == {0}


# --- уши корректора рядом, не по очереди ------------------------------------------------------------------

class _LLM:
    def __init__(self):
        self.users = []

    async def complete_json(self, messages, schema=None, schema_name='', max_tokens=None, **kw):
        if schema_name == 'proofread_flags':
            return {'flags': [{'quote': 'материнские оплаты', 'guess': 'материнские платы', 'why': 'не вяжется'}]}
        self.users.append(messages[-1]['content'])
        return {'old_error': 'yes', 'heard': 'платы', 'fix': {'was': 'оплаты', 'now': 'платы'}, 'why': 'тест'}


def _deps(ear_s: float, res_names: list, **cfg):
    def asr(path, prompt='', model=''):
        time.sleep(ear_s)
        return {'text': 'материнские платы такого', 'segments': []}

    def res(name, slots):
        res_names.append(name)
        return asyncio.Semaphore(1)

    base = dict(corpus_desc='тестовой записи', always_terms=[], round_concurrency=2, whisper_slots=1,
                second_model='other-ear', proofread=True, proofread_ears='clean,second', proofread_min_ears=1,
                proofread_pad_s=4.0, proofread_max_per_min=100.0, proofread_suspect='', proofread_clean_alone=False,
                proofread_max_was_words=3)
    base.update(cfg)
    llm = _LLM()
    return SimpleNamespace(cfg=SimpleNamespace(**base), arbitrate_stage=A, apply_fixes=apply_fixes,
                           audio_clients=SimpleNamespace(asr=asr), _slice=lambda w, a, b, dst: Path(dst).write_bytes(b'RIFF'),
                           _res=res, llm=llm, _around=lambda turns, i, n=1: ''), llm


def _state(tmp_path: Path) -> State:
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav=str(tmp_path / 'x.wav'))
    st.audio_sec = 20.0
    st.turns = [{'cluster': 'SPEAKER_00', 'speaker': 'Speaker_0', 'start': 0.0, 'end': 10.0, 'chunks': [], 'segments': [],
                 'raw': 'У пользователей редко материнские оплаты такого класса стоят.',
                 'final': 'У пользователей редко материнские оплаты такого класса стоят.'}]
    st.known = ()
    return st


async def test_proofread_ears_listen_side_by_side_on_their_lanes_and_keep_order(tmp_path):
    names: list = []
    d, llm = _deps(0.25, names, lanes={'cpu': config.Lane('cpu', 'cpu', 1, ('http://c',))},
                   model_lanes={'other-ear': 'cpu'})
    st = _state(tmp_path)
    t0 = time.monotonic()
    await run_proofread(st, d, Emitter(None, time.monotonic()))
    wall = time.monotonic() - t0
    assert wall < 0.42, f'два уха по 0.25 с шли по очереди: {wall:.2f} с'
    assert set(names) == {'whisper', 'cpu'}, 'чистое ухо — пул whisper, второе — своя полоса'
    user = llm.users[-1]
    assert user.index('УШИ[clean]') < user.index('УШИ[second]'), 'порядок ушей в промпте — прежний'
    assert st.turns[0]['final'].startswith('У пользователей редко материнские платы')
    for key in ('proofread_detect_s', 'proofread_ears_s', 'proofread_llm_s'):
        assert key in st.tm and key.endswith('_s'), key
    assert st.tm['proofread_ears_s'] < 0.42


async def test_proofread_without_lanes_uses_the_whisper_resource_only(tmp_path):
    names: list = []
    d, _ = _deps(0.0, names)
    await run_proofread(_state(tmp_path), d, Emitter(None, time.monotonic()))
    assert set(names) == {'whisper'}
