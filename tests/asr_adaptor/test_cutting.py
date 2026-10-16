"""Разрез длинного звука по тихой точке и сборка сегментов из слов (`backends/transcribe/cutting.py`, чистый numpy)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / 'services' / 'asr-adaptor' / 'backends' / 'transcribe' / 'cutting.py'
spec = importlib.util.spec_from_file_location('cutting', _SRC)
cutting = importlib.util.module_from_spec(spec)
sys.modules['cutting'] = cutting
spec.loader.exec_module(cutting)

SR = 16000


def _tone_with_silence(seconds: float, quiet_at: list[tuple[float, float]]) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for a, b in quiet_at:
        audio[int(a * SR):int(b * SR)] = 0.0
    return audio


def test_short_audio_is_one_piece():
    assert cutting.quiet_cuts(_tone_with_silence(10.0, []), SR) == [(0.0, 10.0)]


def test_cut_falls_into_the_silence_of_the_window_and_pieces_cover_everything():
    # Тишина на 22.0–22.5 с и на 44.0–44.3 с — разрезы обязаны лечь в неё, а не на сетку 24 с.
    audio = _tone_with_silence(60.0, [(22.0, 22.5), (44.0, 44.3)])
    pieces = cutting.quiet_cuts(audio, SR)
    assert len(pieces) == 3
    assert 22.0 <= pieces[0][1] <= 22.5 and 44.0 <= pieces[1][1] <= 44.3
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 60.0
    assert all(b - a <= 24.0 + 1e-6 for a, b in pieces[:-1]), 'каждая часть не длиннее 24 с'
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1)), 'части стыкуются без дыр'


def test_no_silence_still_cuts_within_the_window():
    pieces = cutting.quiet_cuts(_tone_with_silence(50.0, []), SR)
    assert len(pieces) == 3 and all(20.0 <= b - a <= 24.0 + 1e-6 for a, b in pieces[:-1])


def test_segments_split_at_sentence_end_and_long_pause_and_carry_words():
    words = [{'word': 'Спасибо.', 'start': 0.5, 'end': 1.0}, {'word': 'Женя', 'start': 1.2, 'end': 1.4},
             {'word': 'придёт', 'start': 1.5, 'end': 1.8}, {'word': 'позже', 'start': 1.9, 'end': 2.3},
             {'word': 'так', 'start': 3.5, 'end': 3.7}, {'word': 'начнём', 'start': 3.8, 'end': 4.1}]
    segs = cutting.words_to_segments(words, pause=0.6)
    assert [s['text'] for s in segs] == ['Спасибо.', 'Женя придёт позже', 'так начнём']
    assert [(s['start'], s['end']) for s in segs] == [(0.5, 1.0), (1.2, 2.3), (3.5, 4.1)]
    assert [s['id'] for s in segs] == [0, 1, 2]
    assert segs[1]['words'][0] == {'word': 'Женя', 'start': 1.2, 'end': 1.4}, 'слова — ключи word/start/end, как у mlx'


def test_verbose_json_has_the_whisper_shape():
    words = [{'word': 'раз', 'start': 0.1, 'end': 0.3}, {'word': 'два.', 'start': 0.4, 'end': 0.6}]
    out = cutting.verbose_json(words, 'ru', 1.234)
    assert set(out) == {'task', 'language', 'duration', 'text', 'segments'}
    assert out['text'] == 'раз два.' and out['duration'] == 1.234
    assert cutting.verbose_json([], 'ru', 2.0) == {'task': 'transcribe', 'language': 'ru', 'duration': 2.0,
                                                   'text': '', 'segments': []}
