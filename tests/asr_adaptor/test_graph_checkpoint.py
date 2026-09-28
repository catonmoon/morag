"""Чекпойнты графа: состояние едет в JSON без биометрии, прогон возобновляется с узла."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pipeline
from fakes import HINTS
from graph.run import run_graph
from graph.state import State


def test_state_round_trip_keeps_types_and_drops_centroids_and_tokenizer(tmp_path):
    st = State(audio_path='a.wav', hint_set=frozenset({'kafka'}), known=('Postgres',),
               cents={'SPEAKER_00': [0.1, 0.2]}, counter=object(), chunks=[{'start': 0.0, 'raw': 'x'}])
    data = st.to_json()
    assert 'cents' not in data and 'counter' not in data, 'биометрия и объекты в чекпойнт не едут'
    back = State.from_json(json.loads(json.dumps(data)))
    assert back.hint_set == frozenset({'kafka'}) and back.known == ('Postgres',) and back.chunks == st.chunks
    path = tmp_path / 'ck' / 'ep1.json'
    st.save(path)
    assert oct(os.stat(path).st_mode & 0o777) == '0o600'
    assert State.load(path).hint_set == frozenset({'kafka'})


def _norm(r: dict) -> dict:
    r = json.loads(json.dumps(r, ensure_ascii=False, default=str))
    r.pop('env', None)
    r.pop('graph', None)
    r['timing'] = {k: v for k, v in r['timing'].items() if not k.endswith('_s') and k != 'resumed_from'}
    return r


async def test_run_writes_checkpoints_and_resumes_after_a_crash_without_redoing_the_audio(rich, silence, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.CFG, 'graph_checkpoints', str(tmp_path / 'ck'))
    full = await run_graph(str(silence), llm=object(), episode='ep1', hints=HINTS)
    ck = Path(full['graph']['checkpoint'])
    assert ck.exists() and oct(os.stat(ck).st_mode & 0o777) == '0o600'
    data = json.loads(ck.read_text(encoding='utf-8'))
    assert data['done'][-1] == 'coverage' and 'cents' not in data

    # «Упали» после пасса-2: чекпойнт знает куски, но не реплики.
    data['done'] = data['done'][:data['done'].index('pass2') + 1]
    crashed = tmp_path / 'crashed.json'
    crashed.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    def no_diarize(p):
        raise AssertionError('диаризацию при возобновлении звать нельзя')

    monkeypatch.setattr(pipeline.audio_clients, 'diarize', no_diarize)
    rich.calls.clear()
    again = await run_graph(str(silence), llm=object(), episode='ep1', hints=HINTS, resume=str(crashed))
    assert _norm(again) == _norm(full), 'возобновлённый прогон даёт тот же результат'
    assert again['timing']['resumed_from'] == str(crashed)
    assert again['graph']['nodes'] == full['graph']['nodes']
    assert not any(c['prompt'] for c in rich.calls), 'пасс-2 заново не гонялся'


async def test_without_a_checkpoint_dir_nothing_is_written(rich, silence, monkeypatch):
    monkeypatch.setattr(pipeline.CFG, 'graph_checkpoints', '')
    r = await run_graph(str(silence), llm=object(), episode='ep1', hints=HINTS)
    assert 'checkpoint' not in r['graph']
