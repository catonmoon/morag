"""Ранние голоса (`ASR_VOICES_EARLY`, ADR-0033 §5): кластеры узнаются после диаризации, имена — в подсказки и промпты.

Три уровня остаются в данных: метка кластера → голос реестра → имя; имя живёт в слое промптов и показа.
Без флага — прежний порядок узлов и артефакт байт в байт (золотые снимки).
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pipeline
from conveyor import nodes
from conveyor.events import Emitter
from conveyor.proofread import render
from conveyor.run import run_conveyor
from conveyor.state import State
from fakes import HINTS


def _assign_with_cos(cos: float):
    """Заглушка реестра, которая отчитывается решением (`out`) с косинусом матча — его читает порог имени."""
    def assign(cents, air, episode, path, threshold, max_centroids, out=None, **kw):
        if out is not None:
            out.append({'cluster': 'SPEAKER_00', 'action': 'matched', 'label': 'Speaker_0', 'cos': cos})
        return {'SPEAKER_00': 'Speaker_0'}
    return assign


async def test_early_voices_name_turns_feed_hints_and_the_artifact(rich, monkeypatch, silence):
    monkeypatch.setattr(pipeline.CFG, 'voices_early', True)
    monkeypatch.setattr(pipeline.registry, 'assign', _assign_with_cos(0.91))
    monkeypatch.setattr(pipeline.registry, 'names', lambda path: {'Speaker_0': 'Голос Реестра'})
    hints = {**HINTS, 'speaker_names': {'Speaker_0': 'Мария Кузнецова'}}
    r = await run_conveyor(str(silence), llm=object(), episode='ep1', hints=hints)
    order = r['conveyor']['nodes']
    assert order.index('diarize') < order.index('voices') < order.index('pass1'), order
    assert 'voices_s' in r['timing'] and 'speakers_s' in r['timing']
    assert r['speaker_names'] == {'Speaker_0': 'Мария Кузнецова'}, 'карта вызывающего старше имени реестра'
    turns = r['turns']
    assert turns and all(t['speaker_id'] == 'Speaker_0' and t['name'] == 'Мария Кузнецова' and t['cluster'] == 'SPEAKER_00'
                         for t in turns), turns[0]
    assert all(t['speaker'] == 'Speaker_0' for t in turns), 'текст и markdown несут голос реестра, имя — отдельным полем'


async def test_without_the_flag_turns_carry_no_name_and_voices_node_is_absent(rich, silence):
    r = await run_conveyor(str(silence), llm=object(), episode='ep1',
                           hints={**HINTS, 'speaker_names': {'Speaker_0': 'Мария Кузнецова'}})
    assert 'voices' not in r['conveyor']['nodes']
    assert all('name' not in t and 'cluster' not in t for t in r['turns'])
    assert r['speaker_names'] == {}


async def test_recognised_names_join_the_hint_names(rich, monkeypatch, silence):
    seen: dict = {}

    async def build_hints(full_text, llm, *, terms=(), names=(), about=''):
        seen['names'] = list(names)
        return []

    monkeypatch.setattr(pipeline, 'build_hints', build_hints)
    monkeypatch.setattr(pipeline.CFG, 'voices_early', True)
    monkeypatch.setattr(pipeline.registry, 'assign', _assign_with_cos(0.8))
    await run_conveyor(str(silence), llm=object(), episode='ep1',
                       hints={**HINTS, 'names': ['Пётр Ковалёв'], 'speaker_names': {'Speaker_0': 'Мария Кузнецова'}})
    assert seen['names'] == ['Пётр Ковалёв', 'Мария Кузнецова'], 'имя узнанного голоса — к именам подсказки, после данных'


async def test_late_speakers_recognise_only_fresh_clusters_without_the_ticket(tmp_path):
    campp_spans: list = []

    def campp(wav, spans):
        campp_spans.append([s['speaker'] for s in spans])
        return {'SPEAKER_01': [0.1]}, {'SPEAKER_01': 30.0}

    async def no_ticket(*a):
        raise AssertionError('билет реестра поздним узлом после ранних голосов не берётся')

    d = SimpleNamespace(cfg=SimpleNamespace(voices_early=True, campp_slots=1, record_guard=False, naming_trust_cos=0,
                                            registry_path='reg.json', match_threshold=0.7, max_centroids=8),
                        audio_clients=SimpleNamespace(campp=campp),
                        registry=SimpleNamespace(assign=lambda *a, **kw: {'SPEAKER_01': 'Speaker_7'}),
                        _res=lambda name, slots: asyncio.Semaphore(1), _wait_turn=no_ticket,
                        _release_turn=lambda t: None, _project=lambda cents: {})
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav='x.wav')
    st.mapping = {'SPEAKER_00': 'Speaker_0'}
    st.names = {'Speaker_0': 'Мария Кузнецова'}
    st.spans = [{'start': 0.0, 'end': 10.0, 'speaker': 'SPEAKER_00'}, {'start': 10.0, 'end': 40.0, 'speaker': 'SPEAKER_01'}]
    st.turns = [{'cluster': 'SPEAKER_00', 'start': 0.0, 'chunks': [{'start': 0.0, 'end': 10.0}]},
                {'cluster': 'SPEAKER_01', 'start': 10.0, 'chunks': [{'start': 10.0, 'end': 40.0}]}]
    await nodes.speakers(st, d, Emitter(None, time.monotonic()))
    assert campp_spans == [['SPEAKER_01']], 'центроиды — только по отрезкам нового кластера'
    assert st.mapping == {'SPEAKER_00': 'Speaker_0', 'SPEAKER_01': 'Speaker_7'}
    assert [t['speaker'] for t in st.turns] == ['Speaker_0', 'Speaker_7']
    assert st.turns[0]['name'] == 'Мария Кузнецова' and 'name' not in st.turns[1]

    # Все кластеры уже узнаны — CAM++ и реестр не зовутся вовсе.
    campp_spans.clear()
    st.turns = st.turns[:1]
    await nodes.speakers(st, d, Emitter(None, time.monotonic()))
    assert campp_spans == []


def test_prompt_contexts_show_the_label_then_fall_back_to_old_fields():
    turns = [{'cluster': 'SPEAKER_00', 'speaker': 'Speaker_0', 'name': 'Мария Кузнецова', 'label': 'Мария Кузнецова',
              'start': 0.0, 'raw': 'раз', 'final': 'раз'},
             {'cluster': 'SPEAKER_01', 'speaker': 'Speaker_1', 'label': 'Участник 2', 'start': 5.0, 'raw': 'два', 'final': 'два'},
             {'cluster': 'SPEAKER_02', 'start': 9.0, 'raw': 'три', 'final': 'три'}]
    assert pipeline._around(turns, 1, 1).splitlines() == ['[Мария Кузнецова] раз', '[SPEAKER_02] три'], \
        'без подписи (прежний режим) — прежние поля'
    assert render(turns, [0, 1]).splitlines()[0].startswith('[Мария Кузнецова · ')
    assert render(turns, [1]).startswith('[Участник 2 ')


async def test_unnamed_voices_get_neutral_labels_in_order_of_appearance(tmp_path):
    st = State(audio_path='x.mp3', tmp=str(tmp_path), wav='x.wav')
    st.mapping = {'SPEAKER_00': 'Speaker_5', 'SPEAKER_01': 'Speaker_9', 'SPEAKER_02': 'Speaker_12'}
    st.names = {'Speaker_9': 'Мария Кузнецова'}
    st.turns = [{'cluster': c, 'start': float(i), 'chunks': [{'start': float(i), 'end': i + 1.0}]}
                for i, c in enumerate(['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02', 'SPEAKER_00'])]
    nodes._label_turns(st)
    assert [t['label'] for t in st.turns] == ['Участник 1', 'Мария Кузнецова', 'Участник 2', 'Участник 1']
    assert [t['speaker'] for t in st.turns] == ['Speaker_5', 'Speaker_9', 'Speaker_12', 'Speaker_5'], 'данные несут голос реестра'


async def test_unsure_match_keeps_the_voice_number_but_withholds_the_name(rich, monkeypatch, silence):
    """Порог имени (`ASR_VOICES_NAME_COS`, 0.75) строже порога матча: cos 0.62 — голос узнан, имя не подсказываем."""
    monkeypatch.setattr(pipeline.CFG, 'voices_early', True)
    monkeypatch.setattr(pipeline.registry, 'assign', _assign_with_cos(0.62))
    r = await run_conveyor(str(silence), llm=object(), episode='ep1',
                           hints={**HINTS, 'speaker_names': {'Speaker_0': 'Мария Кузнецова'}})
    assert all(t['speaker_id'] == 'Speaker_0' and 'name' not in t for t in r['turns'])
    assert r['speaker_names'] == {}
    assert r['voices']['names_withheld'] == [{'speaker': 'Speaker_0', 'cos': 0.62}]


async def test_match_without_a_reported_cos_gets_no_name(rich, monkeypatch, silence):
    """Реестр не отчитался решением (чужая реализация) — косинус неизвестен, имя не подсказываем."""
    monkeypatch.setattr(pipeline.CFG, 'voices_early', True)
    r = await run_conveyor(str(silence), llm=object(), episode='ep1',
                           hints={**HINTS, 'speaker_names': {'Speaker_0': 'Мария Кузнецова'}})
    assert all('name' not in t for t in r['turns']) and r['speaker_names'] == {}
