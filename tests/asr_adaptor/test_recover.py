"""Склеенный диалог: два голоса по фразам — только когда разделение доказано (stages/recover.py)."""
import numpy as np

from stages import recover as rc


def _v(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    # шум — в долях единичного вектора (базовый нормирован), иначе он заглушает голос
    v = (np.asarray(base) if base is not None else rng.normal(size=192)) \
        + noise * rng.normal(size=192) / np.sqrt(192)
    return (v / np.linalg.norm(v)).tolist()


def test_lone_voice_needs_exactly_one_substantial_cluster():
    assert rc.lone_voice([{'start': 0, 'end': 100, 'speaker': 'A'},
                          {'start': 100, 'end': 105, 'speaker': 'B'}]) == 'A'
    assert rc.lone_voice([{'start': 0, 'end': 100, 'speaker': 'A'},
                          {'start': 100, 'end': 200, 'speaker': 'B'}]) is None


def test_two_people_are_split():
    a, b = _v(1), _v(2)
    vecs = [_v(10 + k, 0.3, a) for k in range(5)] + [_v(20 + k, 0.3, b) for k in range(5)]
    labels, info = rc.split_two(vecs, [2.0] * 10)
    assert labels is not None and info['separation'] >= rc.MIN_SEPARATION
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5]


def test_one_person_is_not_split():
    a = _v(1)
    vecs = [_v(10 + k, 0.3, a) for k in range(10)]
    labels, info = rc.split_two(vecs, [2.0] * 10)
    assert labels is None and info['why'] == 'голоса не разошлись'


def test_relabel_gives_chunks_the_majority_voice():
    chunks = [{'start': 0.0, 'end': 4.0, 'speaker': 'A'}, {'start': 4.0, 'end': 8.0, 'speaker': 'A'}]
    phr = [{'start': 0.5, 'end': 3.5}, {'start': 4.5, 'end': 7.5}, {'start': 8.5, 'end': 12.0}]
    spans = rc.relabel(chunks, [{'start': 0, 'end': 12, 'speaker': 'A'}], phr, [0, 1, 0], 'A')
    assert [c['speaker'] for c in chunks] == ['A', 'A_B']
    assert [s['speaker'] for s in spans] == ['A', 'A_B', 'A']
