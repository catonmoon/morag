"""Добор чанков на промежутки, которые пасс-1 не покрыл."""
from stages import chunking


def test_gap_chunks_split_by_max_and_take_speaker_from_diarization():
    spans = [{'start': 0.0, 'end': 100.0, 'speaker': 'SPEAKER_01'}]

    out = chunking.gap_chunks([(10.0, 50.0)], spans)

    assert len(out) == 2  # 40 с не влезают в один чанк ≤28 с
    assert out[0]['start'] == 10.0
    assert out[-1]['end'] == 50.0
    assert all(c['end'] - c['start'] <= chunking.MAX_S for c in out)
    assert all(c['speaker'] == 'SPEAKER_01' and c['text'] == '' and c['recovered'] for c in out)


def test_gap_chunks_cover_the_hole_without_holes_of_their_own():
    out = chunking.gap_chunks([(0.0, 90.0)], [])

    assert [c['start'] for c in out[1:]] == [c['end'] for c in out[:-1]]
    assert all(c['speaker'] == 'unknown' for c in out)  # диаризация не отработала — фильтровать нечем


def test_gap_chunks_keep_short_hole_intact():
    out = chunking.gap_chunks([(5.0, 12.0)], [])

    assert len(out) == 1 and out[0]['start'] == 5.0 and out[0]['end'] == 12.0


def test_gap_chunks_skip_where_diarization_hears_no_speech():
    """Заставка и музыка — тоже непокрытый промежуток, а Whisper на них фантазирует. Добираем
    только то, что диаризация считает речью, иначе страховка сама дописала бы небывшее."""
    spans = [{'start': 60.0, 'end': 80.0, 'speaker': 'SPEAKER_00'}]

    out = chunking.gap_chunks([(0.0, 100.0)], spans)

    assert [(c['start'], c['end']) for c in out] == [(60.0, 80.0)]


def test_gap_chunks_split_speech_parts_separately():
    """Две реплики в дыре с музыкой между ними → два куска, музыка между ними не добирается."""
    spans = [{'start': 10.0, 'end': 20.0, 'speaker': 'A'}, {'start': 50.0, 'end': 60.0, 'speaker': 'B'}]

    out = chunking.gap_chunks([(0.0, 70.0)], spans)

    assert [(c['start'], c['end'], c['speaker']) for c in out] == [(10.0, 20.0, 'A'), (50.0, 60.0, 'B')]


def test_gap_chunks_merge_into_the_stream_by_time():
    """Добранные встают в общий поток по времени — pipeline сортирует, реплики группируются как обычно."""
    base = [{'start': 0.0, 'end': 10.0, 'speaker': 'A', 'text': 'раз'},
            {'start': 40.0, 'end': 50.0, 'speaker': 'A', 'text': 'два'}]

    merged = sorted(base + chunking.gap_chunks([(10.0, 40.0)], []), key=lambda c: c['start'])

    assert [c['start'] for c in merged] == [0.0, 10.0, 25.0, 40.0]
