"""Голоса по словам: реплика режется по смене говорящего (stages/resplit.py)."""
from stages import resplit as rs

LABELS = {'A': ('Speaker_1', 'Ковалёв'), 'B': ('Speaker_2', 'Speaker_2')}


def label_of(cl):
    return LABELS.get(cl)


def _turn(sid, name, start, end, words):
    text = ' '.join(w for w, *_ in words)
    return ({'speaker': name, 'speaker_id': sid, 'start': start, 'end': end, 'text': text,
             'raw': text, 'segments': [{'start': start, 'end': end, 'text': text}]},
            {'start': start, 'end': end, 'speaker': name, 'words': [list(w) for w in words]})


def test_single_voice_turn_is_untouched():
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 3.0, [('один', 0.1, 0.5), ('два', 0.6, 1.0)])
    spans = [{'start': 0.0, 'end': 3.0, 'speaker': 'A'}]
    out, wout, log = rs.resplit([t], [w], spans, label_of)
    assert out[0] is t and wout[0] is w and log == {}


def test_backchannel_over_host_speech_goes_to_the_short_span():
    # Ковалёв говорит 0–10 с, Б вставляет «ага» на 4.0–4.6 поверх него: слово внутри обоих отрезков.
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 10.0,
                 [('мы', 1.0, 1.3), ('сделали', 1.4, 2.0), ('ага', 4.1, 4.5), ('так', 6.0, 6.3)])
    spans = [{'start': 0.0, 'end': 10.0, 'speaker': 'A'}, {'start': 4.0, 'end': 4.6, 'speaker': 'B'}]
    out, wout, log = rs.resplit([t], [w], spans, label_of)
    assert [x['speaker_id'] for x in out] == ['Speaker_1', 'Speaker_2', 'Speaker_1']
    assert [x['text'] for x in out] == ['мы сделали', 'ага', 'так']
    assert [len(x['words']) for x in wout] == [2, 1, 1]
    assert log['n_moved'] == 1 and log['moments'][0]['at'] == 4.1


def test_tail_of_other_voice_joins_the_next_turn_of_that_voice():
    t1, w1 = _turn('Speaker_1', 'Ковалёв', 0.0, 5.0, [('вопрос', 0.5, 1.0), ('да', 4.2, 4.6)])
    t2, w2 = _turn('Speaker_2', 'Speaker_2', 5.0, 8.0, [('конечно', 5.2, 6.0)])
    spans = [{'start': 0.0, 'end': 4.0, 'speaker': 'A'}, {'start': 4.0, 'end': 8.0, 'speaker': 'B'}]
    out, wout, _ = rs.resplit([t1, t2], [w1, w2], spans, label_of)
    assert [x['text'] for x in out] == ['вопрос', 'да конечно']
    assert out[1]['start'] == 4.2 and [x[0] for x in wout[1]['words']] == ['да', 'конечно']


def test_uncovered_words_keep_the_turn_voice_and_unknown_clusters_prove_nothing():
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 6.0, [('раз', 0.5, 0.9), ('два', 3.0, 3.4)])
    spans = [{'start': 0.0, 'end': 1.0, 'speaker': 'A'}, {'start': 2.9, 'end': 3.5, 'speaker': 'C'}]
    out, _, log = rs.resplit([t], [w], spans, label_of)
    assert len(out) == 1 and log == {}


def test_raw_is_cut_where_the_text_is_cut():
    words = [('первое', 0.1, 0.5), ('слово', 0.6, 1.0), ('ответ', 3.1, 3.5)]
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 4.0, words)
    t['raw'] = 'первое славо ответ'
    spans = [{'start': 0.0, 'end': 2.0, 'speaker': 'A'}, {'start': 3.0, 'end': 4.0, 'speaker': 'B'}]
    out, _, _ = rs.resplit([t], [w], spans, label_of)
    assert [x['raw'] for x in out] == ['первое славо', 'ответ']
