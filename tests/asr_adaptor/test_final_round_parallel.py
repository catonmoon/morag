"""Параллельный финал-раунд: реплики независимы, срыв одной не роняет выпуск.

Стадия занимала 8-12 минут из 15-18 на выпуск при том, что аудио-стадии укладываются в пять.
А срыв на ней стоил нам за день двух выпусков целиком (кончились кредиты, отбил прокси) — при
том, что сырые реплики к этому моменту уже готовы.
"""
import asyncio

import pipeline


def _turns(n: int) -> list[dict]:
    return [{'start': float(i), 'raw': f'реплика номер {i} про NVIDIA'} for i in range(n)]


async def test_turns_are_corrected_in_parallel(monkeypatch):
    """Порядок вызовов не важен, важно что они идут одновременно, а не гуськом."""
    inflight, peak = 0, 0

    async def slow_correct(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.02)
        inflight -= 1
        return raw.replace('NVIDIA', 'NVIDIA!')

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', slow_correct)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = _turns(12)

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 6, lambda m: None)

    assert (n, failed) == (12, 0)
    assert peak > 1, 'реплики шли последовательно — параллельности нет'
    assert all(t['final'].endswith('NVIDIA!') for t in turns)


async def test_concurrency_is_capped(monkeypatch):
    """Потолок соблюдается: провайдер не должен получать залп."""
    inflight, peak = 0, 0

    async def slow_correct(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.02)
        inflight -= 1
        return raw

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', slow_correct)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))

    await pipeline._final_round(_turns(20), 'сводка', [], None, 3, lambda m: None)

    assert peak <= 3


async def test_one_failed_turn_does_not_kill_the_episode(monkeypatch):
    """Реплика остаётся сырой, остальные считаются — вместо потери всей аудио-работы."""
    async def flaky(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        if 'номер 2 ' in raw:
            raise RuntimeError('402 кончились кредиты')
        return raw + ' [правлено]'

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', flaky)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = _turns(5)

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 4, lambda m: None,
                                            sweep_delay=0)

    assert (n, failed) == (5, 1)
    assert turns[2].get('correction_failed') is True
    assert turns[2]['final'] == turns[2]['raw']              # сорвавшаяся — сырой текст
    assert all(t['final'].endswith('[правлено]') for i, t in enumerate(turns) if i != 2)


async def test_short_and_signalless_turns_skip_the_llm(monkeypatch):
    called = 0

    async def counting(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        nonlocal called
        called += 1
        return raw

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: 'NVIDIA' in raw)
    monkeypatch.setattr(pipeline, 'correct', counting)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = [{'start': 0.0, 'raw': 'да'},                       # короткая
             {'start': 1.0, 'raw': 'обычная речь без сущностей'},  # нет сигнала
             {'start': 2.0, 'raw': 'а вот тут про NVIDIA речь'}]

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 4, lambda m: None)

    assert (n, failed, called) == (1, 0, 1)
    assert turns[0]['final'] == 'да'


async def test_context_gives_neighbouring_turns(monkeypatch):
    """Правке нужен разговор вокруг: по нему видно, что WeChat — второй игрок, а не описка.

    Раньше вместо контекста шёл пересказ ЭТОГО ЖЕ фрагмента, сделанный той же моделью.
    """
    seen = []

    async def capture(raw, dsum, context, canon, llm, always=(), recalled='', **kw):
        seen.append(context)
        return raw

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', capture)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = [{'start': 0.0, 'raw': 'сначала про Alipay говорили'},
             {'start': 1.0, 'raw': 'а потом про WeChat подробно'},
             {'start': 2.0, 'raw': 'и закончили на Visa'}]

    await pipeline._final_round(turns, 'сводка', [], None, 4, lambda m: None)

    assert 'Alipay' in seen[1] and 'Visa' in seen[1]      # виден и предыдущий, и следующий
    assert 'WeChat' not in seen[1]                        # сам фрагмент передаётся отдельно
    assert seen[1].count('\n') == 1                       # ровно две соседние реплики, целиком


def test_context_at_the_edges_is_empty_not_broken():
    assert pipeline._around([{'raw': 'одна единственная реплика'}], 0) == ''


def test_context_keeps_whole_turns_and_speakers():
    """Реплика целиком и с меткой говорящего: обрезка по символам рвала бы фразу."""
    turns = [{'raw': 'первая ' * 60, 'speaker': 'Малых'},
             {'raw': 'вторая', 'speaker': 'Колодезев'},
             {'raw': 'третья ' * 60, 'speaker': 'Малых'}]

    ctx = pipeline._around(turns, 1, n=1)

    assert ctx.startswith('[Малых] первая') and ctx.rstrip().endswith('третья')
    assert 'вторая' not in ctx


def _async(value):
    async def _inner():
        return value
    return _inner()


async def test_failed_turn_is_retried_once_and_second_try_wins(monkeypatch):
    """Деген даёт битый JSON, второй заход обычно чистый — как у батчей глоссария.

    НЕ «до успеха»: систематическая ошибка (402/403 — ловили обе) зависла бы навсегда.
    """
    calls = {'n': 0}

    async def flaky_once(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ValueError('LLM returned invalid JSON')
        return raw + ' [правлено]'

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', flaky_once)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = [{'start': 0.0, 'raw': 'одна реплика про NVIDIA'}]

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 2, lambda m: None)

    assert (n, failed) == (1, 0)
    assert turns[0]['final'].endswith('[правлено]')
    assert calls['n'] == 2


async def test_sweep_pass_heals_transient_failure(monkeypatch):
    """Спайк сети/нагрузки: оба захода на месте упали, добивочный проход спустя паузу — прошёл."""
    calls = {'n': 0}

    async def transient(raw, dsum, csum, canon, llm, always=(), recalled='', **kw):
        calls['n'] += 1
        if calls['n'] <= 1:                       # ретраи ВЫЗОВА — в RetryingLLM, стадия зовёт раз
            raise ValueError('LLM returned invalid JSON')
        return raw + ' [правлено]'

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', transient)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = [{'start': 0.0, 'raw': 'одна реплика про NVIDIA'}]

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 2, lambda m: None,
                                            sweep_delay=0)

    assert (n, failed) == (1, 0)
    assert turns[0]['final'].endswith('[правлено]')
    assert not turns[0].get('correction_failed')


async def test_verdicts_are_kept_even_without_an_event_channel(monkeypatch):
    """Вердикты замен собираются всегда, не только при включённой ленте (ADR-0030, наблюдаемость)."""
    async def judging(raw, dsum, csum, canon, llm, always=(), recalled='', fixes_out=None, **kw):
        if fixes_out is not None:
            fixes_out.append({'was': 'пост грес', 'now': 'Postgres', 'ok': True, 'why': 'canonical'})
            fixes_out.append({'was': 'наш', 'now': 'ваш', 'ok': False, 'why': 'common word'})
        return raw.replace('пост грес', 'Postgres')

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', judging)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = _turns(3)
    for t in turns:
        t['raw'] = 'у нас пост грес и наш кэш'
    await pipeline._final_round(turns, 'сводка', [], None, 6, None)   # ленты НЕТ

    for t in turns:
        assert [f['ok'] for f in t['fixes']] == [True, False]
        assert t['final'] == 'у нас Postgres и наш кэш'


def test_known_words_of_the_record_are_not_replaced_by_different_sounding_ones():
    """«МММ» из метки поста финал-раунд менял на «MCP»; «PDN» → «PLN». Известное слово записи может
    смениться только косметически (другой алфавит) или формой — не другим словом."""
    from stages.final_round import apply_fixes

    text = 'кто знает про МММ? а PDN в проде? и Postgres тоже'
    fixes = [{'was': 'МММ', 'now': 'MCP'}, {'was': 'PDN', 'now': 'PLN'},
             {'was': 'Postgres', 'now': 'Постгрес'}]
    log = []
    out, applied, skipped = apply_fixes(text, fixes, log_to=log, protect=['МММ', 'PDN', 'Postgres'])
    assert [v['why'] for v in log] == ['known_term', 'known_term', '']
    assert 'МММ' in out and 'PDN' in out and 'Постгрес' in out          # косметика прошла
    # без protect — прежнее поведение: все три замены применяются
    out2, applied2, _ = apply_fixes(text, fixes)
    assert 'MCP' in out2 and 'PLN' in out2 and applied2 == 3
