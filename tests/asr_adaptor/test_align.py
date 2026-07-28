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
