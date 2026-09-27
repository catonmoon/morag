"""Арбитраж (ADR-0030): правила по отдельности и стадия целиком, на синтетических словах."""
from __future__ import annotations

import pytest
from pathlib import Path

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


def test_vote_words_only_guard_is_optional_and_blocks_non_words(monkeypatch):
    """Страж измерен и по умолчанию выключен; включённый — не верит большинству за не-слово."""
    # обе формы — обычные слова, иначе правило частоты забирает решение раньше голосования
    monkeypatch.setattr(A, 'freq', lambda w, lang: 1e-5 if w.startswith('постави') else 0.0)
    both = 'он поставила задачу'
    assert A.arbitrate('он поставил задачу', both, both, set())[1][0]['by'] == 'голосование'
    garble = 'он ннн задачу'
    assert A.arbitrate('он инн задачу', garble, garble, set())[1]              # без стража берёт
    assert A.arbitrate('он инн задачу', garble, garble, set(), vote_words_only=True)[1] == []


async def test_reader_gate_listens_only_where_the_reader_points(backend, wav, monkeypatch):
    """Ворота читателя: второе ухо зовётся только для кусков, которые читатель счёл невменяемыми;
    остальные не слушаются вовсе — это и есть экономия прохода."""
    pytest.importorskip('wordfreq')
    asked: list[str] = []
    second_calls: list[str] = []

    class Reader:
        async def complete_json(self, messages, schema, **kw):
            text = messages[-1]['content'].rsplit('Текст:', 1)[-1].strip()
            asked.append(text)
            return {'suspicious': ['прегресс'] if 'прегресс' in text else []}

    def asr(path, prompt='', model='', **kw):
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = backend.slices[path]
        if model == 'other':
            second_calls.append(path)
            return {'text': 'это регресс тут', 'segments': [{'start': 0.0, 'end': b - a, 'text': 'это регресс тут'}]}
        text = 'это прегресс тут' if a < 30 else 'это чисто тут'
        return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}

    # ⚠️ Реестр семафоров конвейера переживает тест и держит семафор, привязанный к ЧУЖОМУ циклу
    # событий: под нагрузкой (читатель + два уха разом) это «bound to a different event loop».
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)
    monkeypatch.setattr(pipeline.CFG, 'second_model', 'other')
    monkeypatch.setattr(pipeline.CFG, 'arbitrate_gate', 'reader')
    monkeypatch.setattr(pipeline.CFG, 'clean_ear', '')
    r = await pipeline.run_pipeline(str(wav), llm=Reader(), episode='ep1')

    assert asked, 'читателя не спросили'
    gate = r['timing']['arbitrate_gate']
    assert gate['слушали'] >= 1 and gate['пропущено'] >= 1
    assert len(second_calls) == gate['слушали']                 # второе ухо — только по пометке
    assert all(d['by'] == 'частота' for d in r['arbitration'])


def test_prompt_guard_catches_both_failure_modes_of_a_prompt():
    clean = {'avg_logprob': -0.25, 'compression_ratio': 2.0}
    assert A.prompt_guard(clean, {'avg_logprob': -0.27, 'compression_ratio': 2.1}) == ''
    # модель «ослепла» от контекста в подсказке — logprob просел, сжатие даже упало
    assert A.prompt_guard(clean, {'avg_logprob': -1.4, 'compression_ratio': 1.6}).startswith('ослепла')
    # петля: сжатие взлетело, а logprob ВЫРОС — уверенное зацикливание, logprob его не видит
    assert A.prompt_guard(clean, {'avg_logprob': 0.04, 'compression_ratio': 22.0}).startswith('петля')
    assert A.prompt_guard({}, {}) == ''                          # метрик нет — не судим


async def test_reader_does_not_flag_the_known_spellings_it_was_given():
    """Пометка, совпадающая с известным написанием, — не пометка: читатель бывало переписывал
    в подозрительные весь список, который ему дали."""
    class Echo:
        async def complete_json(self, messages, schema, **kw):
            return {'suspicious': ['Postgres', 'Кузнецова', 'прегресс', 'kafka']}

    flags = await A.reader_flags(Echo(), 'это прегресс тут', ['Postgres', 'Мария Кузнецова', 'Kafka'])
    assert flags == ['прегресс']


def test_unresolved_dispute_is_journaled_not_decided():
    """Спор без правила и свидетеля остаётся черновиком, но виден в журнале — по нему зовут третий голос."""
    text, dec = A.arbitrate('он поставил задачу', 'он поставила задачу', None, set())
    assert text == 'он поставил задачу'
    assert dec == [{'i': 1, 'was': 'поставил', 'now': 'поставила', 'by': 'спорно', 'taken': False}]


def test_ear_window_never_cuts_a_word_out_of_context():
    chunk = {'start': 100.0, 'end': 112.0, 'raw': 'один два три четыре',
             'segments': [{'start': 100.0, 'end': 112.0, 'text': 'один два три четыре',
                           'words': [{'word': ' три', 'start': 107.0, 'end': 107.5}]}]}
    disputes = [{'i': 2, 'was': 'три', 'now': 'тры', 'by': 'спорно'}]
    assert A.ear_window(chunk, disputes, 'chunk', 3000.0) == (100.0, 112.0)
    assert A.ear_window(chunk, disputes, 'window30', 3000.0) == (92.0, 122.0)       # 30 с вокруг слова
    a, b = A.ear_window(chunk, disputes, 'neighbours', 3000.0, (70.0, 150.0))
    assert a <= 100.0 and b >= 112.0 and b - a <= A.NEIGHBOURS_MAX
    assert A.dispute_time({'start': 0.0, 'end': 10.0, 'raw': 'a b c d'}, [{'i': 3}]) == 8.75   # доля токена


async def test_clean_ear_on_demand_is_called_only_where_a_dispute_remains(backend, wav, monkeypatch):
    """Режим demand: чистое ухо зовётся только для кусков, где второе ухо оставило нерешённый спор."""
    clean_calls: list[str] = []

    def asr(path, prompt='', model='', **kw):
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = backend.slices[path]
        if model == 'other':
            text = 'он поставила задачу' if a < 30 else 'это чисто тут'   # спор только в первом куске
        elif Path(path).stem.startswith('e'):                             # третий голос — своим окном
            clean_calls.append((path, a, b)); text = 'он поставила задачу'
            return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}
        else:
            text = 'он поставил задачу' if a < 30 else 'это чисто тут'
        return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}

    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)
    monkeypatch.setattr(pipeline.CFG, 'second_model', 'other')
    monkeypatch.setattr(pipeline.CFG, 'clean_ear', 'demand')
    monkeypatch.setattr(pipeline.CFG, 'clean_ear_window', 'window30')
    monkeypatch.setattr(pipeline.CFG, 'arbitrate_gate', '')
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    gate = r['timing']['arbitrate_gate']
    assert gate['третий голос'] >= 1 and len(clean_calls) == gate['третий голос']
    assert all(abs((b - a) - A.EAR_WINDOW) < 1e-6 or a == 0.0 for _, a, b in clean_calls)   # окно, не слово
    assert gate['секунд третьего голоса'] > 0
    assert all(d['by'] in ('голосование', 'спорно') for d in r['arbitration'])
    assert any(d['by'] == 'голосование' and d['taken'] for d in r['arbitration'])


async def test_protect_known_runs_end_to_end(backend, wav, monkeypatch):
    """Защита известных слов собирается в run_pipeline из подсказок и постоянных терминов — путь,
    которого юнит-тесты _final_round не проходят (ловилось вживую: NameError на финал-раунде)."""
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.CFG, 'protect_known', True)
    monkeypatch.setattr(pipeline.CFG, 'final_ear', True)
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1',
                                    hints={'terms': ['Postgres'], 'names': ['Мария Кузнецова']})
    assert r['turns'] and r['markdown']


async def test_always_terms_stay_protected_when_known_word_protection_is_off(backend, wav, monkeypatch):
    """Регрессия на затенение: при выключенной защите известных слов прежнее вето по постоянным
    терминам (`_term_survives`) обязано работать как раньше."""
    from stages.final_round import apply_fixes

    async def proposing(raw, dsum, csum, canon, llm, always=(), recalled='', fixes_out=None, **kw):
        fixed, _, _ = apply_fixes(raw, [{'was': 'Kubernetes', 'now': 'Cabernet'}], canon, always, log_to=fixes_out)
        return fixed

    def asr(path, prompt='', **kw):
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        a, b = backend.slices[path]
        return {'text': 'ставим Kubernetes сюда', 'segments': [{'start': 0.0, 'end': b - a, 'text': 'ставим Kubernetes сюда'}]}

    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)
    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', proposing)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])

    async def nothing(*a, **kw):
        return ''

    monkeypatch.setattr(pipeline, 'recall_entities', nothing)
    monkeypatch.setattr(pipeline.CFG, 'always_terms', ('Kubernetes',))
    monkeypatch.setattr(pipeline.CFG, 'protect_known', False)
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    verdicts = [f for f in r.get('fixes') or [] if f['was'] == 'Kubernetes']
    assert verdicts and all(f['why'] == 'breaks_term' for f in verdicts)
    assert all('Cabernet' not in (t.get('text') or '') for t in r['turns'])


async def test_backend_death_during_arbitration_is_counted_in_the_artifact(backend, wav, monkeypatch):
    """Бэкенд упал посреди стадии: запись не роняем, но число кусков без арбитража едет в артефакт."""
    def asr(path, prompt='', model='', **kw):
        if path.endswith('in.wav'):
            return {'text': ' '.join(s['text'] for s in PASS1), 'segments': PASS1}
        if model == 'other':
            raise ConnectionError('Max retries exceeded')          # второе ухо — бэкенда больше нет
        a, b = backend.slices[path]
        text = 'он поставил задачу' if a < 30 else 'это чисто тут'
        return {'text': text, 'segments': [{'start': 0.0, 'end': b - a, 'text': text}]}

    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.audio_clients, 'asr', asr)
    monkeypatch.setattr(pipeline.CFG, 'second_model', 'other')
    monkeypatch.setattr(pipeline.CFG, 'clean_ear', '')
    monkeypatch.setattr(pipeline.CFG, 'arbitrate_gate', '')
    r = await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')

    assert not r.get('arbitration')                            # журнал пуст — ключа нет
    assert r['timing']['arbitrate_failed'] >= 1

