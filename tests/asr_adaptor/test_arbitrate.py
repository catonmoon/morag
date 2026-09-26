"""Арбитраж (ADR-0030): правила по отдельности и стадия целиком, на синтетических словах."""
from __future__ import annotations

import pytest

import pipeline
from stages import arbitrate as A
from test_pipeline_recovery import PASS1, backend, wav  # noqa: F401 — фикстуры


def test_key_and_sound_ignore_case_punctuation_and_alphabet():
    assert A.key('«Postgres»,') == 'postgres'
    assert A.sound('Postgres') == A.sound('постгрес')
    assert A.sound('Kafka') == A.sound('кафка')


def test_only_one_to_one_swaps_inside_the_chunk_are_touched(monkeypatch):
    # Правило под тестом — КАНОН; частотник выключен, иначе латинское слово, известное ему с
    # крошечной частотой, забирает решение себе (порядок правил: частота раньше канона).
    monkeypatch.setattr(A, 'freq', lambda w, lang: None)
    canon = {A.sound('Postgres')}
    # «пастгрес» — гарбл с искажённой основой: похоже, но не то же; канон подтверждает второе ухо
    # (гарбл, отличающийся ПОСЛЕДНЕЙ буквой, по правилу формы слова неотличим от падежа — и это
    # намеренно: «поставил / поставила» разбирает голосование, а не канон)
    text, dec = A.arbitrate('база пастгрес тут', 'база Postgres тут', None, canon)
    assert text == 'база Postgres тут' and [d['by'] for d in dec] == ['канон']
    # а «постгрес» звучит ТАК ЖЕ — это не спор: канонизация не заповедь (решение владельца)
    text, dec = A.arbitrate('база постгрес тут', 'база Postgres тут', None, canon)
    assert text == 'база постгрес тут' and dec == []
    # край куска — артефакт границы: не трогаем
    text, dec = A.arbitrate('пастгрес тут', 'Postgres тут', None, canon)
    assert text == 'пастгрес тут' and dec == []
    # развал на два слова против одного — не «слово не расслышано»
    text, dec = A.arbitrate('база пастгрес тут', 'база пост грес тут', None, canon)
    assert text == 'база пастгрес тут' and dec == []


def test_same_sound_is_not_a_dispute_and_inflection_is_not_canon():
    canon = {A.sound('Kafka')}
    # «кафка» против «Kafka» — то же слово иначе записано: канонизация не заповедь
    text, dec = A.arbitrate('пишем в кафка сейчас', 'пишем в Kafka сейчас', None, canon)
    assert dec == []
    # «Кафкой» — форма того же слова, канон её не «чинит» на именительный
    text, dec = A.arbitrate('пишем кафкой сейчас', 'пишем Kafka сейчас', None, canon)
    assert text == 'пишем кафкой сейчас'


def test_frequency_rule_fixes_a_garble_into_an_ordinary_word():
    pytest.importorskip('wordfreq')
    text, dec = A.arbitrate('это прегресс тут', 'это регресс тут', None, set())
    assert text == 'это регресс тут' and dec[0]['by'] == 'частота'
    # обычное слово на обычное — отношение частот не даёт права менять
    text, dec = A.arbitrate('он поставил задачу', 'он поставила задачу', None, set())
    assert text == 'он поставил задачу'


def test_voting_takes_the_ear_but_canon_vetoes_the_majority():
    # чистое ухо и вторая модель услышали одно, кусок — другое: большинство берёт
    text, dec = A.arbitrate('он поставил задачу', 'он поставила задачу', 'он поставила задачу', set())
    assert text == 'он поставила задачу' and dec[0]['by'] == 'голосование'
    # …но слово куска знает канон — большинство не указ, и отказ виден в журнале
    canon = {A.sound('SQL')}
    text, dec = A.arbitrate('пишем SQL запрос', 'пишем скуль запрос', 'пишем скуль запрос', canon)
    assert text == 'пишем SQL запрос'
    assert dec == [{'i': 1, 'was': 'SQL', 'now': 'скуль', 'by': 'вето', 'taken': False}]


def test_apply_keeps_punctuation_and_patches_segments_and_decoder_words():
    chunk = {'raw': 'база, пастгрес. тут',
             'segments': [{'start': 0.0, 'end': 1.0, 'text': 'база, пастгрес.',
                           'words': [{'word': ' пастгрес.', 'start': 0.4, 'end': 0.9}]}]}
    dec = A.apply(chunk, 'база Postgres тут', None, {A.sound('Postgres')})
    assert chunk['raw'] == 'база, Postgres. тут' and chunk['arbitrated'] is True
    assert chunk['segments'][0]['text'] == 'база, Postgres.'
    assert chunk['segments'][0]['words'][0]['word'] == ' Postgres.'
    assert dec[0]['taken'] and dec[0]['now'] == 'Postgres.'


async def test_stage_is_absent_without_a_second_model(backend, wav):
    assert pipeline.CFG.second_model == ''
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')
    assert not r.get('arbitration')


async def test_stage_asks_the_second_model_and_journals_the_swap(backend, wav, monkeypatch):
    pytest.importorskip('wordfreq')
    calls: list[str] = []

    def asr(path, prompt='', model='', **kw):
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        calls.append(model)
        a, b = backend.slices[path]
        text = 'это регресс тут' if model == 'other' else 'это прегресс тут'
        return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}

    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)
    monkeypatch.setattr(pipeline.CFG, 'second_model', 'other')
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    assert 'other' in calls and '' in calls                    # вторая модель спрошена, первая — тоже
    assert r['arbitration'] and all(d['by'] == 'частота' and d['taken'] for d in r['arbitration'])
    # …и журнал ДОЕЗЖАЕТ до артефакта: x_enriched собирается явным перечнем полей
    from app import _enriched
    assert _enriched(r)['x_enriched']['arbitration'] == r['arbitration']
    assert all('регресс' in t['raw'] and 'прегресс' not in t['raw'] for t in r['turns'] if t['raw'])


def test_canon_takes_spellings_for_verification_only():
    canon = A.canon_from({'terms': ['Postgres'], 'names': ['Мария Кузнецова'],
                          'spellings': ['Kafka', 'Отдел Разметки']},
                         [{'canonicals': ['Redis']}])
    for word in ('Postgres', 'Кузнецова', 'Мария', 'Kafka', 'Разметки', 'Redis'):
        assert A.sound(word) in canon
    assert A.canon_from(None, None) == set()
