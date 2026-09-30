"""Шов с соседями: окно с запасом отдаёт куску только его слова (stages/seams.py)."""
from stages import seams


def _seg(start, end, words):
    return {'start': start, 'end': end, 'text': ' '.join(w for w, *_ in words),
            'avg_logprob': -0.2,
            'words': [{'word': w, 'start': a, 'end': b} for w, a, b in words]}


def test_keep_inside_drops_neighbour_words_by_midpoint():
    # Вырезка с 8.5 с (запас 1.5 с), кусок [10, 14]: «конец соседа» звучит до 10-й секунды.
    segs = [_seg(0.0, 4.5, [('конец', 0.1, 0.6), ('соседа', 0.7, 1.4),
                            ('наше', 1.6, 2.2), ('слово', 2.3, 3.0)]),
            _seg(4.6, 7.0, [('чужое', 6.0, 6.9)])]
    out = seams.keep_inside(segs, 8.5, 10.0, 14.0)
    assert seams.text_of(out) == 'наше слово'
    assert out[0]['start'] == 10.1 and out[0]['avg_logprob'] == -0.2


def test_keep_inside_without_word_times_uses_segment_midpoint():
    segs = [{'start': 0.0, 'end': 1.2, 'text': 'сосед'},
            {'start': 1.5, 'end': 4.0, 'text': 'своё'}]
    assert seams.text_of(seams.keep_inside(segs, 8.5, 10.0, 14.0)) == 'своё'


def test_dedupe_head_needs_three_words_in_a_row():
    assert seams.dedupe_head('и вот мы пришли к тому', 'пришли к тому, что всё') == ('что всё', 3)
    # Одиночное совпадение — живая речь («да, да»), не шов.
    assert seams.dedupe_head('ну да', 'да, конечно')[1] == 0


def test_dedupe_chunks_only_touches_padded_chunks():
    prev = {'start': 0.0, 'end': 5.0, 'raw': 'мы пришли к тому'}
    plain = {'start': 5.0, 'end': 8.0, 'raw': 'пришли к тому снова'}
    padded = {'start': 5.0, 'end': 8.0, 'raw': 'пришли к тому снова', 'retried': True,
              'segments': [{'start': 5.0, 'end': 8.0, 'text': 'пришли к тому снова'}]}
    assert seams.dedupe_chunks([prev, dict(plain)]) == []
    log = seams.dedupe_chunks([prev, padded])
    assert padded['raw'] == 'снова' and padded['segments'][0]['text'] == 'снова'
    assert log[0]['dropped'] == 3
