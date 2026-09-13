"""TranscriptChunker: подсказки границ и привязки (ADR-0027) на синтетическом транскрипте.

Реплики по 30 с из трёх предложений близкой длины → предложение ≈ 10 с по интерполяции.
"""
from unittest.mock import AsyncMock

import pytest

from morag.indexing.chunker import TranscriptChunker
from morag.indexing.token_counter import TiktokenCounter

counter = TiktokenCounter()

S = [
    'Первое предложение про очереди.', 'Второе предложение про брокер.', 'Третье предложение про топики.',
    'Четвёртое про партиции данных.', 'Пятое про репликацию узлов.', 'Шестое про подтверждения.',
    'Седьмое про потребителей.', 'Восьмое про смещения чтения.', 'Девятое про группы клиентов.',
]
TEXT = (
    f'[Лектор] <!-- t:0.0 --> {S[0]} {S[1]} {S[2]}\n\n'
    f'[Лектор] <!-- t:30.0 --> {S[3]} {S[4]} {S[5]}\n\n'
    f'[Лектор] <!-- t:60.0 --> {S[6]} {S[7]} {S[8]}\n\n'
    f'[Лектор] <!-- t:90.0 --> Десятое, последнее.'
)


def _chunker(mock, **kw) -> TranscriptChunker:
    kw.setdefault('max_tokens', 10_000)
    kw.setdefault('boundary_min_tokens', 1)
    return TranscriptChunker(mock, counter, boundary_hints=True, **kw)


def _starts(results):
    return [r.start_sec for r in results]


def _firsts(results):
    """Первое предложение каждого чанка (без `[Лектор] `)."""
    return [r.text.split('] ', 1)[1].split('.')[0] + '.' for r in results]


# --- байт в байт без подсказок ----------------------------------------------------------------

async def test_no_hints_is_single_llm_call_as_before():
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 2, 'topic': 'a'},
                                                    {'start': 3, 'end': 4, 'topic': 'b'}]}
    res = await _chunker(mock).chunk_with_metadata(TEXT)
    assert mock.complete_json.call_count == 1
    numbered = mock.complete_json.call_args[0][0][1]['content']
    assert numbered.startswith('[1] [Лектор] Первое') and '[4] [Лектор] Десятое' in numbered
    assert _starts(res) == [0.0, 60.0]


async def test_empty_hints_equal_no_hints():
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 4, 'topic': 'a'}]}
    res = await _chunker(mock).chunk_with_metadata(TEXT, boundaries=[], pins=[])
    assert mock.complete_json.call_count == 1 and len(res) == 1


# --- притяжение -------------------------------------------------------------------------------

async def test_hint_inside_turn_snaps_to_nearest_sentence_start():
    mock = AsyncMock()
    res = await _chunker(mock).chunk_with_metadata(TEXT, boundaries=[41.0])
    assert mock.complete_json.call_count == 0            # отрезки короче max_tokens — без LLM
    assert _firsts(res) == ['Первое предложение про очереди.', 'Пятое про репликацию узлов.']
    assert 38.0 <= res[1].start_sec <= 42.0               # интерполяция ≈ 40 с
    assert res[0].end_sec == res[1].start_sec


async def test_hint_near_turn_start_snaps_to_measured_turn_start():
    res = await _chunker(AsyncMock()).chunk_with_metadata(TEXT, boundaries=[63.0])
    assert _starts(res) == [0.0, 60.0]
    assert _firsts(res)[1] == 'Седьмое про потребителей.'


async def test_hint_without_candidate_in_window_is_dropped():
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 4, 'topic': 'a'}]}
    res = await _chunker(mock, boundary_window_sec=2.0).chunk_with_metadata(TEXT, boundaries=[45.0])
    assert mock.complete_json.call_count == 1 and len(res) == 1  # как без подсказок


async def test_hint_at_document_start_makes_no_cut():
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 4, 'topic': 'a'}]}
    res = await _chunker(mock).chunk_with_metadata(TEXT, boundaries=[0.0])
    assert len(res) == 1


# --- отрезки: клейка коротких, LLM по длинным -------------------------------------------------

async def test_short_stretch_is_glued_to_shorter_neighbour():
    # подсказки 30 и 40 → отрезок из одного предложения (4-е); соседи: 3 предложения слева, 5 справа
    short = counter.count(S[3])
    res = await _chunker(AsyncMock(), boundary_min_tokens=short + 1).chunk_with_metadata(
        TEXT, boundaries=[30.0, 40.0])
    assert _firsts(res) == ['Первое предложение про очереди.', 'Пятое про репликацию узлов.']


async def test_long_stretch_goes_to_llm_with_local_indices():
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 1, 'topic': 'a'},
                                                    {'start': 2, 'end': 3, 'topic': 'b'}]}
    # отрезок после разреза на 30 с — три юнита: реплика 2, реплика 3, реплика 4
    tail = counter.count(f'{S[3]} {S[4]} {S[5]}') + counter.count(f'{S[6]} {S[7]} {S[8]}') \
        + counter.count('Десятое, последнее.')
    res = await _chunker(mock, max_tokens=tail - 1).chunk_with_metadata(TEXT, boundaries=[30.0])
    assert mock.complete_json.call_count == 1
    numbered = mock.complete_json.call_args[0][0][1]['content']
    assert numbered.startswith('[1] [Лектор] Четвёртое')      # нумерация — внутри отрезка
    assert '[3] [Лектор] Десятое' in numbered and 'Первое' not in numbered
    # сегменты сдвинуты на начало отрезка: реплика 1 | реплика 2 | реплики 3-4
    assert _starts(res) == [0.0, 30.0, 60.0]


# --- привязки сильнее окна ----------------------------------------------------------------------

async def test_pin_backward_moves_cut_past_the_referring_sentence():
    """Слайд сменился на 60 с, но 7-е предложение (60-70 с) договаривает про прежний экран."""
    res = await _chunker(AsyncMock()).chunk_with_metadata(TEXT, boundaries=[60.0], pins=[(63.0, 30.0)])
    assert _firsts(res)[1] == 'Восьмое про смещения чтения.'
    assert 68.0 <= res[1].start_sec <= 72.0


async def test_pin_forward_moves_cut_before_the_announcing_sentence():
    """6-е предложение (50-60 с) анонсирует следующий экран → открывает его чанк."""
    res = await _chunker(AsyncMock()).chunk_with_metadata(TEXT, boundaries=[60.0], pins=[(55.0, 60.0)])
    assert _firsts(res)[1] == 'Шестое про подтверждения.'
    assert 48.0 <= res[1].start_sec <= 52.0


async def test_pin_wins_over_window():
    """В окне 3 с единственный кандидат (60.0) рвёт привязку → ближайший удовлетворяющий вне окна."""
    res = await _chunker(AsyncMock(), boundary_window_sec=3.0).chunk_with_metadata(
        TEXT, boundaries=[60.0], pins=[(75.0, 30.0)])
    assert _firsts(res)[1] == 'Девятое про группы клиентов.'


async def test_pin_to_the_same_switch_does_not_block_the_cut():
    """Обращение к ТЕКУЩЕМУ экрану после смены — не помеха разрезу на этой смене."""
    res = await _chunker(AsyncMock()).chunk_with_metadata(TEXT, boundaries=[60.0], pins=[(65.0, 60.0)])
    assert _starts(res) == [0.0, 60.0]


# --- признак для pipeline ------------------------------------------------------------------------

def test_supports_boundaries_flag_follows_config():
    assert TranscriptChunker(AsyncMock(), counter).supports_boundaries is False
    assert _chunker(AsyncMock()).supports_boundaries is True


@pytest.mark.parametrize('kw', [{}, {'boundaries': [41.0]}])
async def test_speakers_and_offsets_survive(kw):
    mock = AsyncMock()
    mock.complete_json.return_value = {'segments': [{'start': 1, 'end': 2, 'topic': 'a'},
                                                    {'start': 3, 'end': 4, 'topic': 'b'}]}
    res = await _chunker(mock).chunk_with_metadata(TEXT, **kw)
    assert len(res) == 2
    for r in res:
        assert r.speakers == ['Лектор'] and r.text.startswith('[Лектор] ')
    assert res[0].char_offset == 0 < res[1].char_offset < len(TEXT)
