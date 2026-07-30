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

    async def slow_correct(raw, dsum, csum, canon, llm, always=(), recalled=''):
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

    async def slow_correct(raw, dsum, csum, canon, llm, always=(), recalled=''):
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
    async def flaky(raw, dsum, csum, canon, llm, always=(), recalled=''):
        if 'номер 2 ' in raw:
            raise RuntimeError('402 кончились кредиты')
        return raw + ' [правлено]'

    monkeypatch.setattr(pipeline, 'has_entity_signal', lambda raw, gloss: True)
    monkeypatch.setattr(pipeline, 'correct', flaky)
    monkeypatch.setattr(pipeline, 'relevant', lambda *a: [])
    monkeypatch.setattr(pipeline, 'recall_entities', lambda *a, **kw: _async(''))
    turns = _turns(5)

    n, failed = await pipeline._final_round(turns, 'сводка', [], None, 4, lambda m: None)

    assert (n, failed) == (5, 1)
    assert turns[2]['final'] == turns[2]['raw']              # сорвавшаяся — сырой текст
    assert all(t['final'].endswith('[правлено]') for i, t in enumerate(turns) if i != 2)


async def test_short_and_signalless_turns_skip_the_llm(monkeypatch):
    called = 0

    async def counting(raw, dsum, csum, canon, llm, always=(), recalled=''):
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

    async def capture(raw, dsum, context, canon, llm, always=(), recalled=''):
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
