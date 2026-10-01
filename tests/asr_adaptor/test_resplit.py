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
    out, wout, log = rs.resplit([t], [w], spans, label_of, backchannel={'ага'})
    assert [x['speaker_id'] for x in out] == ['Speaker_1', 'Speaker_2', 'Speaker_1']
    assert [x['text'] for x in out] == ['мы сделали', 'ага', 'так']
    assert [len(x['words']) for x in wout] == [2, 1, 1]
    assert log['n_moved'] == 1 and log['moments'][0]['at'] == 4.1


def test_host_word_under_a_short_interjection_stays_with_the_host():
    # кто-то сказал «да» поверх лектора, а whisper записал слово ЛЕКТОРА — оно не уходит перебившему
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 10.0,
                 [('объясните,', 1.0, 1.6), ('пожалуйста,', 1.7, 2.3), ('мне,', 4.1, 4.5), ('кто', 6.0, 6.3)])
    spans = [{'start': 0.0, 'end': 10.0, 'speaker': 'A'}, {'start': 4.0, 'end': 4.6, 'speaker': 'B'}]
    out, _, log = rs.resplit([t], [w], spans, label_of, backchannel={'ага', 'да'})
    assert len(out) == 1 and log == {}


def test_witness_returns_a_run_that_sounds_like_its_neighbour():
    # мелкий кластер B забрал целую фразу лектора: по голосу она — лектор
    a_voice, b_voice = [1.0, 0.0], [0.0, 1.0]
    words = [('раз', 0.0, 1.5), ('два.', 1.6, 3.0), ('Так,', 3.2, 3.6), ('вот', 3.7, 4.0),
             ('видно.', 4.1, 5.0), ('три', 5.2, 6.5), ('четыре.', 6.6, 8.0)]
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 8.0, words)
    spans = [{'start': 0.0, 'end': 3.1, 'speaker': 'A'}, {'start': 3.1, 'end': 5.1, 'speaker': 'B'},
             {'start': 5.1, 'end': 8.0, 'speaker': 'A'}]

    def lecturer(sp):
        return [a_voice for _ in sp]                       # всё звучит как лектор
    out, _, log = rs.resplit([t], [w], spans, label_of, embed=lecturer)
    assert len(out) == 1 and log['witness'][0]['why'] == 'голос'

    def honest(sp):
        return [b_voice if 3.1 <= a <= 5.1 else a_voice for a, b in sp]   # там и правда другой голос
    out, _, _ = rs.resplit([t], [w], spans, label_of, embed=honest)
    assert [x['speaker_id'] for x in out] == ['Speaker_1', 'Speaker_2', 'Speaker_1']


def test_tail_of_other_voice_joins_the_next_turn_of_that_voice():
    t1, w1 = _turn('Speaker_1', 'Ковалёв', 0.0, 5.0, [('вопрос', 0.5, 1.0), ('да', 4.2, 4.6)])
    t2, w2 = _turn('Speaker_2', 'Speaker_2', 5.0, 8.0, [('конечно', 5.2, 6.0)])
    spans = [{'start': 0.0, 'end': 4.0, 'speaker': 'A'}, {'start': 4.0, 'end': 8.0, 'speaker': 'B'}]
    out, wout, _ = rs.resplit([t1, t2], [w1, w2], spans, label_of, backchannel={'да'})
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


def test_pinned_words_take_the_voice_on_their_left():
    # «опление» — задвоение на шве, которое правка человека удаляет: своего голоса у него нет
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 6.0,
                 [('отличное', 1.0, 1.5), ('выступление.', 1.6, 2.4), ('опление', 4.1, 5.2)])
    spans = [{'start': 0.0, 'end': 3.0, 'speaker': 'A'}, {'start': 4.0, 'end': 5.0, 'speaker': 'B'}]
    out, _, _ = rs.resplit([t], [w], spans, label_of)
    assert [x['speaker_id'] for x in out] == ['Speaker_1', 'Speaker_2']
    out, _, log = rs.resplit([t], [w], spans, label_of, pinned={(0, 2)})
    assert len(out) == 1 and log == {}


def test_snap_moves_a_mid_sentence_boundary_to_the_sentence_end():
    # «… продиагностируем. Ещё | вопрос.» — диаризация отдала «Ещё» прежнему голосу
    toks = ['давай', 'продиагностируем.', 'Ещё', 'вопрос.']
    assert rs.snap(['A', 'A', 'A', 'B'], toks) == ['A', 'A', 'B', 'B']
    # «…конференции, но | вопросы действительно... Есть! Спасибо» — граница уезжает вправо
    toks = ['лучший', 'доклад', 'на', 'конференции,', 'но', 'вопросы', 'действительно...', 'Есть!',
            'Спасибо,', 'давайте']
    assert rs.snap(list('AAAAABBBBB'), toks) == list('AAAAAAABBB')


def test_a_word_without_proven_voice_takes_its_neighbours_not_the_old_turn_label():
    # старая реплика подписана докладчиком (ошибка), по словам это зритель; «Вот, и» — в слабом кластере
    t, w = _turn('Speaker_1', 'Ковалёв', 0.0, 9.0,
                 [('конференции.', 1.0, 1.8), ('Вот,', 2.0, 2.3), ('и', 2.4, 2.5), ('такой', 2.6, 3.0)])
    spans = [{'start': 0.0, 'end': 1.9, 'speaker': 'B'}, {'start': 1.9, 'end': 2.55, 'speaker': 'C'},
             {'start': 2.55, 'end': 9.0, 'speaker': 'B'}]
    out, _, _ = rs.resplit([t], [w], spans, label_of)          # 'C' — неизвестный/слабый голос
    assert [x['speaker_id'] for x in out] == ['Speaker_2'] and out[0]['text'].startswith('конференции.')


def test_snap_crosses_old_turn_boundaries():
    t1, w1 = _turn('Speaker_1', 'Ковалёв', 0.0, 3.0,
                   [('давай', 0.1, 0.3), ('это', 0.4, 0.5), ('продиагностируем.', 0.6, 1.5), ('Ещё', 2.0, 2.4)])
    t2, w2 = _turn('Speaker_1', 'Ковалёв', 3.0, 6.0, [('вопрос.', 3.1, 3.6), ('Спасибо', 4.0, 4.5)])
    spans = [{'start': 0.0, 'end': 2.5, 'speaker': 'A'}, {'start': 2.9, 'end': 6.0, 'speaker': 'B'}]
    out, _, _ = rs.resplit([t1, t2], [w1, w2], spans, label_of)
    assert [(x['speaker_id'], x['text']) for x in out] == \
        [('Speaker_1', 'давай это продиагностируем.'), ('Speaker_2', 'Ещё вопрос. Спасибо')]


def test_snap_keeps_a_backchannel_and_grows_it_by_one_word_at_most():
    toks = ['мы', 'сделали,', 'ага,', 'так', 'и', 'идём']          # конца предложения рядом нет
    assert rs.snap(['A', 'A', 'B', 'A', 'A', 'A'], toks) == ['A', 'A', 'B', 'A', 'A', 'A']
    toks = ['мы', 'сделали.', 'Ну', 'вот', 'ага', 'так']            # двух слов «Ну вот» не отнимает
    assert rs.snap(['A', 'A', 'A', 'A', 'B', 'A'], toks) == ['A', 'A', 'A', 'A', 'B', 'A']
