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


# --- вырожденные чанки --------------------------------------------------------
# ⚠️ Куплено живым переносом: 115-минутный доклад умер после девяти минут работы на
# `ffmpeg -ss 222.36 -to 222.36` — границы чанка схлопнулись при печати с двумя знаками.
# Пасс-2 на таком куске всё равно вернул бы пусто, а прогон он ронял целиком.


def test_degenerate_word_does_not_become_a_chunk():
    """Пасс-1 иногда отдаёт сегмент нулевой длины — чанка из него быть не должно."""
    words = [{'text': 'раз', 'start': 0.0, 'end': 2.0},
             {'text': 'глюк', 'start': 222.36, 'end': 222.36},
             {'text': 'два', 'start': 400.0, 'end': 402.0}]

    out = chunking.chunk(words, [{'start': 0.0, 'end': 500.0, 'speaker': 'SPEAKER_00'}])

    assert all(c['end'] - c['start'] >= chunking.MIN_S for c in out), \
        'чанк без звука уронит ffmpeg на нарезке'
    assert any('раз' in c['text'] for c in out) and any('два' in c['text'] for c in out), \
        'нормальные слова должны остаться'


def test_rounding_cannot_collapse_chunk_boundaries():
    """Ровно тот случай, что упал: длина есть, но при печати %.2f она исчезает."""
    words = [{'text': 'а', 'start': 222.361, 'end': 222.364}]

    out = chunking.chunk(words, [])

    assert not [c for c in out if f"{c['start']:.2f}" == f"{c['end']:.2f}"]


def test_gap_chunks_ignore_empty_holes():
    assert chunking.gap_chunks([(10.0, 10.0)], []) == []


def test_normal_chunks_survive():
    words = [{'text': 'обычная', 'start': 0.0, 'end': 1.0},
             {'text': 'речь', 'start': 1.0, 'end': 3.0}]

    out = chunking.chunk(words, [])

    assert len(out) == 1 and out[0]['text'] == 'обычная речь'


def test_short_merge_glues_only_the_same_voice_and_is_off_by_default():
    """`ASR_CHUNK_SHORT_S`: короткие куски одного голоса клеятся до порога; чужой голос — никогда; без ручки — как было."""
    from stages.chunking import chunk
    segs = [{'start': 0.0, 'end': 4.0, 'text': 'раз'}, {'start': 4.5, 'end': 8.0, 'text': 'два'},
            {'start': 8.5, 'end': 11.0, 'text': 'три'}, {'start': 11.5, 'end': 20.0, 'text': 'четыре'},
            {'start': 20.5, 'end': 24.0, 'text': 'пять'}]
    spans = [{'start': 0.0, 'end': 8.2, 'speaker': 'A'}, {'start': 8.2, 'end': 11.2, 'speaker': 'B'},
             {'start': 11.2, 'end': 24.5, 'speaker': 'A'}]
    plain = chunk({'segments': segs}, spans)
    assert [(c['speaker'], round(c['end'] - c['start'], 1)) for c in plain] == [('A', 8.0), ('B', 2.5), ('A', 12.5)]
    merged = chunk({'segments': segs}, spans, short_s=10.0)
    assert [(c['speaker'], round(c['end'] - c['start'], 1)) for c in merged] == [('A', 8.0), ('B', 2.5), ('A', 12.5)], \
        'короткий кусок A (8 с) не клеится к B (2.5 с) и не достаёт через B до второго A'
    segs2 = [{'start': 0.0, 'end': 4.0, 'text': 'раз'}, {'start': 4.5, 'end': 8.0, 'text': 'два'},
             {'start': 8.5, 'end': 15.0, 'text': 'три'}]
    spans2 = [{'start': 0.0, 'end': 15.5, 'speaker': 'A'}]
    # один голос, пауза 0.5 с между сегментами: штатный разрез режет ≤ 28 с → один кусок и так; проверим на разрыве
    spans3 = [{'start': 0.0, 'end': 8.2, 'speaker': 'A'}, {'start': 8.2, 'end': 15.5, 'speaker': 'A'}]
    assert len(chunk({'segments': segs2}, spans3)) == 1
