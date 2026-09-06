"""Пословное выравнивание: то, что проверяется без торча и без звука.

Сам forced alignment (MMS_FA) проверяется прогоном на реальном выпуске — см. ADR-0019.
"""
from stages import align


def test_module_imports_without_torch():
    """Импорт стадии не должен тянуть торч: в Docker-варианте его нет, а pipeline её импортирует."""
    assert align.FORMAT == 'morag-words-v1'


def test_enforce_order_pulls_back_words_that_go_backwards():
    """Караоке ищет текущее слово двоичным поиском и на неотсортированном врёт молча."""
    turns = [{'words': [['раз', 0.0, 1.0], ['два', 0.5, 0.9]]},
             {'words': [['три', 0.2, 2.0]]}]

    fixed = align.enforce_order(turns)

    assert fixed == 2
    starts = [w[1] for t in turns for w in t['words']]
    assert starts == sorted(starts)


def test_enforce_order_counts_across_turn_boundary():
    """Счётчик сквозной: собеседники перебивают друг друга, и слова разъезжаются на стыке реплик."""
    turns = [{'words': [['раз', 0.0, 5.0]]}, {'words': [['два', 1.0, 2.0]]}]

    assert align.enforce_order(turns) == 1
    assert turns[1]['words'][0] == ['два', 5.0, 5.0]


def test_enforce_order_leaves_sorted_words_alone():
    turns = [{'words': [['раз', 0.0, 1.0], ['два', 1.5, 2.0]]}]

    assert align.enforce_order(turns) == 0
    assert turns[0]['words'] == [['раз', 0.0, 1.0], ['два', 1.5, 2.0]]


class _FakeAligner:
    """Вместо MMS_FA: раскладывает слова по окну поровну — проверяем обвязку, не акустику."""

    device = 'cpu'

    def __init__(self):
        self.calls = []

    def turn(self, wave, t0, t1, words):
        self.calls.append((t0, t1, tuple(words)))
        step = (t1 - t0) / max(len(words), 1)
        return [[w, round(t0 + i * step, 2), round(t0 + (i + 1) * step, 2)]
                for i, w in enumerate(words)]


def _patch(monkeypatch, aligner):
    monkeypatch.setattr(align, '_device', lambda explicit='': 'cpu')
    monkeypatch.setattr(align, '_get_aligner', lambda device: aligner)
    monkeypatch.setattr(align, '_read_wav', lambda path: None)


def test_align_turns_builds_words_document(monkeypatch):
    fake = _FakeAligner()
    _patch(monkeypatch, fake)
    turns = [{'start': 0.0, 'speaker': 'Первый', 'text': 'раз два'},
             {'start': 10.0, 'speaker': 'Второй', 'text': 'три'}]

    doc = align.align_turns('in.wav', turns, 30.0, episode='ep29')

    assert doc['format'] == 'morag-words-v1'
    assert doc['episode'] == 'ep29' and doc['duration_sec'] == 30.0
    assert doc['words_total'] == 3
    assert doc['turns'][0]['words'][0][0] == 'раз'
    assert [t['speaker'] for t in doc['turns']] == ['Первый', 'Второй']


def test_align_turns_uses_window_to_the_next_turn(monkeypatch):
    """Конца у реплики нет: окно идёт до следующей, последнее — до конца звука. По этой же
    конвенции считается покрытие, иначе потеря в хвосте реплики спряталась бы."""
    fake = _FakeAligner()
    _patch(monkeypatch, fake)
    turns = [{'start': 0.0, 'speaker': 'A', 'text': 'раз'},
             {'start': 10.0, 'speaker': 'B', 'text': 'два'}]

    doc = align.align_turns('in.wav', turns, 30.0)

    assert [(t['start'], t['end']) for t in doc['turns']] == [(0.0, 10.0), (10.0, 30.0)]
    assert [c[:2] for c in fake.calls] == [(0.0, 10.0), (10.0, 30.0)]


def test_align_turns_skips_empty_turns(monkeypatch):
    fake = _FakeAligner()
    _patch(monkeypatch, fake)
    turns = [{'start': 0.0, 'speaker': 'A', 'text': ''},
             {'start': 10.0, 'speaker': 'B', 'text': 'два'}]

    doc = align.align_turns('in.wav', turns, 20.0)

    assert doc['turns'][0]['words'] == []
    assert len(fake.calls) == 1  # пустую реплику модели не отдаём


def _turn(pauses, n=60, seg=10.0, per_seg=3, gap=0.2):
    """Реплика из n сегментов по seg секунд; pauses = {индекс сегмента: пауза ПОСЛЕ него}.

    Слова уникальны («с3-1»), поэтому по индексу видно, на каком сегменте прошёл разрез.
    """
    segs, t = [], 0.0
    for i in range(n):
        segs.append({'start': round(t, 2), 'end': round(t + seg, 2),
                     'text': ' '.join(f'с{i}-{k}' for k in range(per_seg))})
        t += seg + pauses.get(i, gap)
    return {'start': 0.0, 'end': round(t - gap, 2), 'segments': segs,
            'text': ' '.join(s['text'] for s in segs)}


def test_split_turn_leaves_short_turn_alone():
    """Реплики короче капа не трогаем: у подкаста медиана 61 с, его выравнивание должно
    остаться прежним до последнего знака."""
    turn = _turn({}, n=10)
    words = turn['text'].split()

    assert align.split_turn(turn, 0.0, 102.0, words) == [(0.0, 102.0, 0, len(words))]


def test_split_turn_cuts_in_the_middle_of_the_longest_pause():
    """Режем по самой длинной паузе в досягаемости, а границу ставим в её СЕРЕДИНУ: погрешность
    времён сегментов не должна отрезать начало или хвост слова."""
    turn = _turn({5: 3.0, 20: 5.0})           # 20-й сегмент кончается на 214-й секунде — в капе
    words = turn['text'].split()
    end = turn['end']

    pieces = align.split_turn(turn, 0.0, end, words)

    cut = turn['segments'][20]['end'] + 5.0 / 2
    assert pieces[0] == (0.0, cut, 0, 21 * 3)  # слова 21 сегмента — в первом куске
    assert pieces[1][0] == cut


def test_split_turn_ignores_pauses_too_close_to_the_start():
    """Кусок короче эмиссионного (CHUNK) резать незачем: модель не примет огрызок, и слова
    остались бы без времён вовсе."""
    turn = _turn({1: 8.0, 20: 5.0})           # длиннейшая пауза — на 28-й секунде, ближе CHUNK
    words = turn['text'].split()

    pieces = align.split_turn(turn, 0.0, turn['end'], words)

    assert pieces[0][1] == turn['segments'][20]['end'] + 2.5


def test_split_turn_covers_the_whole_turn_without_holes():
    """Куски идут встык и по звуку, и по словам: пропущенное слово — потерянная подсветка."""
    turn = _turn({7: 1.0, 20: 5.0, 33: 2.0, 44: 4.0}, n=60)
    words = turn['text'].split()
    end = turn['end']

    pieces = align.split_turn(turn, 0.0, end, words)

    assert len(pieces) > 1
    assert pieces[0][0] == 0.0 and pieces[-1][1] == end
    assert pieces[0][2] == 0 and pieces[-1][3] == len(words)
    for (_, b, _, j), (a2, _, i2, _) in zip(pieces, pieces[1:]):
        assert (b, j) == (a2, i2)


def test_split_turn_maps_word_index_through_final_round_edits():
    """Паузы известны про СЫРОЙ текст, а режем финальный: финал-раунд правит слова
    («постгрез» → «Postgres»), и от этого индексы разъезжаются."""
    turn = _turn({20: 5.0}, n=40)
    raw = turn['text'].split()
    final = list(raw)
    final[4] = 'Postgres'                      # правка слова — длина та же
    final.insert(9, 'ещё')                     # вставка — всё дальнейшее сдвинулось на слово
    turn['text'] = ' '.join(final)

    pieces = align.split_turn(turn, 0.0, turn['end'], final)

    assert pieces[0][3] == 21 * 3 + 1          # +1: вставленное слово тоже слева от разреза
    assert final[pieces[0][3] - 1] == raw[21 * 3 - 1]


def test_split_turn_without_segments_keeps_whole_turn():
    """Чужой вызов без сегментов резать не по чему: пусть будет медленно, но верно."""
    turn = {'text': 'раз два три', 'start': 0.0}

    assert align.split_turn(turn, 0.0, 900.0, ['раз', 'два', 'три']) == [(0.0, 900.0, 0, 3)]


def test_split_turn_merges_short_tail_into_previous_piece():
    """Огрызок в хвосте короче CHUNK модель не примет — слова остались бы без времён вовсе."""
    turn = _turn({22: 5.0}, n=25)              # единственная пауза — почти у самого конца
    words = turn['text'].split()
    end = turn['end']

    pieces = align.split_turn(turn, 0.0, end, words)

    assert end - (turn['segments'][22]['end'] + 2.5) < align.CHUNK   # огрызок вышел бы коротким
    assert pieces == [(0.0, end, 0, len(words))]                     # поэтому его вернули назад


def test_align_turns_splits_long_turn_and_concatenates_words(monkeypatch):
    """Разрез — деталь реализации: наружу реплика уезжает одним списком слов по порядку."""
    fake = _FakeAligner()
    _patch(monkeypatch, fake)
    turn = _turn({20: 5.0}, n=60)
    turn['speaker'] = 'A'

    doc = align.align_turns('in.wav', [turn], turn['end'])

    assert len(fake.calls) > 1                                     # реплика ушла кусками
    assert doc['words_total'] == len(turn['text'].split())         # но слова все на месте
    starts = [w[1] for w in doc['turns'][0]['words']]
    assert starts == sorted(starts)
