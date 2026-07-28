"""Покрытие звука расшифровкой: арифметика, которая делает потерю речи видимой."""
from stages import coverage as C


def test_merge_collapses_overlaps():
    # сегменты соседних кусков налезают друг на друга (повтор с паддингом) — двойной счёт завысил
    # бы покрытие и спрятал дыру
    assert C.merge([(5.0, 10.0), (0.0, 6.0), (20.0, 21.0)]) == [(0.0, 10.0), (20.0, 21.0)]


def test_covered_clips_to_window():
    assert C.covered([(0.0, 10.0), (20.0, 30.0)], 5.0, 25.0) == 10.0


def test_holes_finds_gap_in_the_middle():
    assert C.holes([(0.0, 20.0), (50.0, 70.0)], 0.0, 70.0, 5.0) == [(20.0, 50.0)]


def test_holes_finds_gap_at_both_ends():
    assert C.holes([(10.0, 20.0)], 0.0, 40.0, 5.0) == [(0.0, 10.0), (20.0, 40.0)]


def test_holes_ignores_gaps_below_threshold():
    assert C.holes([(0.0, 20.0), (23.0, 40.0)], 0.0, 40.0, 5.0) == []


def test_holes_empty_when_fully_covered():
    assert C.holes([(0.0, 40.0)], 0.0, 40.0, 5.0) == []


def test_turn_windows_run_to_the_next_turn():
    turns = [{'start': 0.0}, {'start': 30.0}, {'start': 90.0}]
    assert C.turn_windows(turns, 120.0) == [(0.0, 30.0), (30.0, 90.0), (90.0, 120.0)]


def _chunk(start, end, segments, **kw):
    return {'start': start, 'end': end, 'raw': ' '.join(s['text'] for s in segments),
            'segments': segments, **kw}


def _seg(start, end, text='речь'):
    return {'start': start, 'end': end, 'text': text}


def test_summarize_counts_pass1_holes_and_recovery():
    """Дыру пасса-1 добрали чанком — она видна и как дыра, и как возвращённая речь."""
    p1 = [_seg(0.0, 40.0), _seg(60.0, 100.0)]
    chunks = [
        _chunk(0.0, 40.0, [_seg(0.0, 40.0)]),
        _chunk(40.0, 60.0, [_seg(41.0, 58.0)], recovered=True),
        _chunk(60.0, 100.0, [_seg(60.0, 100.0)]),
    ]
    turns = [{'start': 0.0, 'speaker': 'Кто-то', 'chunks': chunks}]

    cov = C.summarize(100.0, p1, chunks, turns, 5.0)

    assert cov['pass1_holes'] == [[40.0, 60.0]]
    assert cov['pass1_hole_sec'] == 20.0
    assert cov['recovered_chunks'] == 1
    assert cov['recovered_sec'] == 17.0
    assert cov['lost_sec'] == 0.0  # добор закрыл дыру


def test_summarize_reports_hole_inside_turn():
    chunks = [_chunk(0.0, 20.0, [_seg(0.0, 20.0)]), _chunk(50.0, 70.0, [_seg(50.0, 70.0)])]
    turns = [{'start': 0.0, 'speaker': 'Кто-то', 'chunks': chunks}]

    cov = C.summarize(100.0, [_seg(0.0, 100.0)], chunks, turns, 5.0)

    assert cov['lost_sec'] == 30.0
    assert cov['lost_spots'][0] == {'start': 20.0, 'sec': 30.0, 'speaker': 'Кто-то'}


def test_summarize_does_not_blame_trailing_music():
    """Хвост звука после последнего слова — заставка, а не потерянная речь."""
    chunks = [_chunk(0.0, 20.0, [_seg(0.0, 20.0)])]
    turns = [{'start': 0.0, 'speaker': 'Кто-то', 'chunks': chunks}]

    cov = C.summarize(300.0, [_seg(0.0, 20.0)], chunks, turns, 5.0)

    assert cov['lost_sec'] == 0.0


def test_summarize_lists_empty_chunks():
    chunks = [_chunk(0.0, 20.0, [_seg(0.0, 20.0)]), _chunk(20.0, 40.0, [])]
    turns = [{'start': 0.0, 'speaker': 'Кто-то', 'chunks': chunks}]

    cov = C.summarize(40.0, [_seg(0.0, 40.0)], chunks, turns, 5.0)

    assert cov['empty_chunks'] == [[20.0, 40.0]]
    assert cov['lost_sec'] == 0.0  # окно последней реплики обрезано по последнему слову
