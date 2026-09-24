"""Состав ленты событий на сквозном прогоне с поддельными бэкендами.

Проверяем не «событие пришло», а то, ради чего канал заведён: что показать работу МОЖНО —
диаризация даёт ленту голосов, каждый кусок пасса-2 отчитывается вместе с контекстом, который
ушёл в whisper, а решение по голосу несёт близость и ветку. И что показывать при этом нечего
лишнего: событие обязано быть сериализуемым и не содержать биометрии.
"""
from __future__ import annotations

import json

import pytest
from test_pipeline_recovery import AUDIO_S, backend, wav  # noqa: F401 — фикстуры переиспользуем

import pipeline


async def _events(wav_path, **kw) -> list[dict]:
    seen: list[dict] = []
    seq = [0]

    def progress(m):
        if isinstance(m, str):
            return
        seq[0] += 1
        m['i'] = seq[0]
        seen.append(m)

    await pipeline.run_pipeline(str(wav_path), llm=None, episode='ep1', progress=progress, **kw)
    return seen


def _of(events, kind):
    return [e for e in events if e['t'] == kind]


async def test_the_stream_describes_the_whole_run(backend, wav):  # noqa: F811
    events = await _events(wav)

    assert events, 'лента не должна быть пустой'
    assert [e['i'] for e in events] == list(range(1, len(events) + 1)), 'нумерация сплошная'
    assert all(isinstance(e.get('at'), (int, float)) for e in events), 'у события есть время'

    meta = _of(events, 'job.meta')
    assert len(meta) == 1 and meta[0]['audio_sec'] == pytest.approx(AUDIO_S, abs=0.2)
    assert meta[0]['env'], 'шкала и отпечаток стенда открывают ленту'

    started = [e['stage'] for e in _of(events, 'stage.start')]
    ended = [e['stage'] for e in _of(events, 'stage.end')]
    assert started[:4] == ['diarize', 'pass1', 'glossary', 'pass2']
    assert set(ended) <= set(started), 'стадия не может кончиться, не начавшись'
    assert all(isinstance(e['sec'], (int, float)) for e in _of(events, 'stage.end'))


async def test_diarization_comes_out_as_a_ribbon(backend, wav):  # noqa: F811
    spans = _of(await _events(wav), 'diar.spans')
    assert len(spans) == 1
    assert spans[0]['speakers'] == ['SPEAKER_00']
    ribbon = spans[0]['spans']
    assert ribbon and all(len(s) == 3 for s in ribbon), '[начало, конец, индекс голоса]'
    assert all(a < b for a, b, _ in ribbon) and ribbon == sorted(ribbon)


async def test_every_chunk_reports_itself_with_the_context_given_to_whisper(backend, wav):  # noqa: F811
    events = await _events(wav)
    started, done = _of(events, 'chunk.start'), _of(events, 'chunk.done')

    assert started and len(started) == len(done), 'на каждый начатый кусок есть законченный'
    assert [e['i_'] if 'i_' in e else e['i'] for e in started] != [], 'куски пронумерованы'
    first = started[0]
    assert first['n'] == len(started) and first['from'] < first['to']
    # ⚠️ Ради этого поля событие и стоит ровно здесь: дальше цикл затирает промпт, и показать
    # контекст, ушедший в whisper, было бы уже неоткуда.
    assert first['prompt'] == 'каноники'
    assert 'terms' in first and isinstance(first['terms'], list)
    assert all('raw' in e or 'error' in e for e in done)


async def test_speaker_decisions_carry_closeness_but_never_biometrics(backend, wav, monkeypatch):  # noqa: F811
    """⚠️⚠️ 192 числа — это отпечаток живого голоса. Наружу едет только решение и пара координат."""
    import numpy as np

    rng = np.random.default_rng(7)
    cents = {'SPEAKER_00': (rng.normal(size=192) / 13.8).astype(np.float32).tolist(),
             'SPEAKER_01': (rng.normal(size=192) / 13.8).astype(np.float32).tolist()}
    monkeypatch.setattr(pipeline.audio_clients, 'campp',
                        lambda p, spans: (cents, {'SPEAKER_00': 700.0, 'SPEAKER_01': 20.0}))

    def fake_assign(c, air, ep, path, thr=0.55, cap=8, out=None):
        if out is not None:
            out.append({'cluster': 'SPEAKER_00', 'air': 700.0, 'label': 'Speaker_0',
                        'best': 'Speaker_0', 'cos': 0.94, 'action': 'matched'})
            out.append({'cluster': 'SPEAKER_01', 'air': 20.0, 'label': 'Speaker_9',
                        'best': 'Speaker_0', 'cos': 0.31, 'action': 'new'})
        return {'SPEAKER_00': 'Speaker_0', 'SPEAKER_01': 'Speaker_9'}

    monkeypatch.setattr(pipeline.registry, 'assign', fake_assign)

    vecs = _of(await _events(wav), 'spk.vec')
    assert len(vecs) == 2
    assert [v['action'] for v in vecs] == ['matched', 'new']
    assert vecs[0]['cos'] == 0.94 and vecs[1]['best'] == 'Speaker_0'
    for v in vecs:
        assert len(v['xy']) == 2 and all(abs(c) <= 1.0001 for c in v['xy']), 'проекция нормирована'
        assert not any(isinstance(x, list) and len(x) > 8 for x in v.values()), 'вектора в событии нет'


async def test_every_event_survives_json(backend, wav):  # noqa: F811
    """Лента едет по HTTP и пишется в файл — несериализуемое поле обрушило бы весь опрос."""
    events = await _events(wav)
    blob = json.dumps(events, ensure_ascii=False)
    assert json.loads(blob) == events
    assert max(len(json.dumps(e, ensure_ascii=False)) for e in events) < 60_000, \
        'одно событие не должно весить как страница'
