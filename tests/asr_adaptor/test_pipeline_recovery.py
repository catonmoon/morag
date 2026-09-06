"""Сквозной прогон конвейера с поддельными бэкендами: потерянная речь возвращается в транскрипт.

Аудио-бэкенды (Мак), LLM и ffmpeg подменены — проверяем ровно то, что нельзя увидеть в юнитах:
чанк, добранный из дыры пасса-1, и чанк, переспрошенный после пустого ответа, доходят до .md,
а отчёт о покрытии сходится с тем, что реально распознано.
"""
from __future__ import annotations

import wave
from pathlib import Path

import pytest

import pipeline
from app import _enriched

SR = 16000
AUDIO_S = 90.0

# Пасс-1 «услышал» два куска и промолчал между 20-й и 50-й секундой: чанков там не возникло бы,
# и пасс-2 этот звук никогда бы не услышал — это второй механизм потери речи.
PASS1 = [{'start': 0.0, 'end': 20.0, 'text': 'начало разговора'},
         {'start': 50.0, 'end': 90.0, 'text': 'продолжение разговора'}]
SILENT_CHUNK = (0.0, 20.0)  # на этот кусок пасс-2 ответит пусто, пока его спрашивают с промптом


class FakeBackend:
    """Подменяет audio_clients: помнит, какой кусок звука у какого файла, и что на него ответил."""

    def __init__(self) -> None:
        self.slices: dict[str, tuple[float, float]] = {}
        self.calls: list[dict] = []

    def cut(self, wav, a, b, dst):
        self.slices[dst] = (a, b)

    def asr(self, path: str, prompt: str = '') -> dict:
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = self.slices[path]
        self.calls.append({'a': a, 'b': b, 'prompt': prompt})
        if SILENT_CHUNK[0] <= a < SILENT_CHUNK[1] and prompt:
            return {'text': '', 'segments': []}
        text = f'речь-{a:.0f}'
        return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """Тишина нужной длины: покрытие считается по заголовку wav."""
    path = tmp_path / 'source.wav'
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b'\0' * (int(AUDIO_S * SR) * 2))
    return path


@pytest.fixture
def backend(monkeypatch, wav) -> FakeBackend:
    fake = FakeBackend()

    monkeypatch.setattr(pipeline, '_to_wav', lambda src, dst: Path(dst).write_bytes(wav.read_bytes()))
    monkeypatch.setattr(pipeline, '_slice', fake.cut)
    monkeypatch.setattr(pipeline.audio_clients, 'asr', fake.asr)
    monkeypatch.setattr(pipeline.audio_clients, 'diarize',
                        lambda p: [{'start': 0.0, 'end': AUDIO_S, 'speaker': 'SPEAKER_00'}])
    monkeypatch.setattr(pipeline.audio_clients, 'campp', lambda p, spans: ({}, {}))
    monkeypatch.setattr(pipeline.registry, 'assign', lambda *a, **kw: {'SPEAKER_00': 'Speaker_0'})
    monkeypatch.setattr(pipeline.registry, 'names', lambda path: {})

    # LLM-стадии: содержательно они здесь ни при чём
    async def nothing(*a, **kw):
        return []

    async def empty_text(*a, **kw):
        return ''

    async def no_names(*a, **kw):
        return {}, []

    monkeypatch.setattr(pipeline, 'build_glossary', nothing)
    monkeypatch.setattr(pipeline, 'doc_summary', empty_text)
    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: False)
    monkeypatch.setattr(pipeline, 'name_speakers', no_names)
    monkeypatch.setattr(pipeline, 'WhisperTokenCounter', lambda model: None)
    monkeypatch.setattr(pipeline, 'build_prompt',
                        lambda terms, counter, budget, always=(), hinted=(): 'каноники')
    return fake


async def test_gap_left_by_pass1_is_recovered_into_the_transcript(backend, wav):
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    # Отпечаток установки едет в артефакте: через год видно, чем сделана эта расшифровка.
    assert r['env'].get('host') and r['env'].get('packages')

    cov = r['coverage']
    assert cov['pass1_holes'] == [[20.0, 50.0]]
    assert cov['recovered_chunks'] == 2  # 30 с не влезают в один чанк ≤28 с
    assert cov['lost_sec'] == 0.0
    # добранное — не только в отчёте: речь дошла до расшифровки
    assert 'речь-20' in r['markdown'] and 'речь-35' in r['markdown']


async def test_empty_answer_is_retried_without_prompt_and_with_padding(backend, wav):
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    retries = [c for c in backend.calls if not c['prompt']]
    assert len(retries) == 1
    assert retries[0]['a'] == pytest.approx(SILENT_CHUNK[0] - pipeline.PAD_S, abs=0.01) or \
        retries[0]['a'] == 0.0  # у куска в начале файла паддинг слева упирается в ноль
    assert retries[0]['b'] > SILENT_CHUNK[1]  # справа паддинг есть всегда
    assert r['coverage']['retried_chunks'] == 1
    assert r['coverage']['empty_chunks'] == []  # повтор закрыл пустоту
    assert 'речь-0' in r['markdown']


async def test_hole_inside_a_chunk_is_not_recovered(backend, wav, monkeypatch):
    """Пасс-2 переслушивает чанк ЦЕЛИКОМ, поэтому дыра пасса-1 внутри чанка уже покрыта.

    Добор по сегментам пасса-1 дублировал бы текст — поймано на живом ep2-10: фраза «этот сам
    термин недоопределён…» пришла дважды, во второй раз хуже («изобретать» → «изобили»).
    """
    # Короткий сегмент 16-17 склеится с соседом (`_merge_short`), и чанк 0-17 накроет дыру 10-16.
    inside = [{'start': 0.0, 'end': 10.0, 'text': 'начало'},
              {'start': 16.0, 'end': 17.0, 'text': 'ага'},
              {'start': 18.0, 'end': AUDIO_S, 'text': 'продолжение'}]

    def asr(path, prompt=''):
        if path.endswith('in.wav'):
            return {'text': 'начало ага продолжение', 'segments': inside}
        return backend.asr(path, prompt)

    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)

    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    assert r['coverage']['pass1_holes'] == [[10.0, 16.0]]  # дыра видна в отчёте
    assert r['coverage']['unheard'] == []                  # но пасс-2 её услышит сам
    assert r['coverage']['recovered_chunks'] == 0


async def test_music_gap_is_not_invented(backend, wav, monkeypatch):
    """Пасс-1 молчит на заставке — но и диаризация там речи не слышит, добирать нечего.

    Без этого страховка от потери речи сама дописывала бы в транскрипт фантазии Whisper на музыке.
    """
    monkeypatch.setattr(pipeline.audio_clients, 'diarize',
                        lambda p: [{'start': 0.0, 'end': 20.0, 'speaker': 'SPEAKER_00'},
                                   {'start': 50.0, 'end': AUDIO_S, 'speaker': 'SPEAKER_00'}])

    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    assert r['coverage']['recovered_chunks'] == 0
    assert 'речь-20' not in r['markdown']
    assert r['coverage']['pass1_holes'] == [[20.0, 50.0]]  # дыра при этом остаётся видимой


async def test_turns_carry_real_segment_times(backend, wav):
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    for t in r['turns']:
        assert t['end'] > t['start']
        assert t['segments'] and all(s['end'] > s['start'] for s in t['segments'])

    # в стандартном verbose_json теперь настоящие сегменты, а не заглушка start == end
    segments = _enriched(r)['segments']
    assert segments and all(s['end'] > s['start'] for s in segments)
    assert [s['id'] for s in segments] == list(range(len(segments)))


async def test_alignment_failure_does_not_break_the_transcript(backend, wav):
    """Торча в этом окружении нет — стадия обязана упасть мягко, транскрипт остаётся."""
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    assert r['words'] is None
    assert r['markdown'] and r['turns']
