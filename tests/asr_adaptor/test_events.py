"""Лента событий стадий: включается по просьбе и НЕ существует, пока её не просили.

⚠️⚠️ Первый тест здесь — главный. На этом адаптере живёт не один продукт, и тот, который ленту не
просит, обязан не заметить вообще ничего: ни лишнего ключа в ответе, ни изменившегося типа у
`progress`, ни аллокации. Канал показа работы не имеет права стоить чужому клиенту ни байта.
"""
import asyncio

import jobs
import pytest


async def _run(factory, *, events=False):
    job_id = jobs.submit(factory, events=events)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if jobs.get(job_id)['status'] in ('done', 'error'):
            break
    return job_id, jobs.get(job_id)


async def test_without_asking_there_is_no_ring_and_no_new_fields():
    async def factory(progress):
        progress('diarize')                       # строка — как раньше
        progress({'t': 'chunk.done', 'i': 1})     # словарь — некому слушать, уходит в никуда
        progress('pass2 20/195')
        return {'ok': True}

    job_id, job = await _run(factory)
    assert job['status'] == 'done'
    assert 'events' not in job and 'seq' not in job, 'ленты не просили — кольца быть не должно'
    assert job['progress'] == 'pass2 20/195', 'строка прогресса прежняя и по-прежнему строка'
    assert isinstance(job['progress'], str), 'словарь не смеет попасть в поле, которое читают как строку'
    assert jobs.since(job, 0) == ([], 0, 0)


async def test_the_no_op_callback_of_sync_mode_still_works():
    """Синхронный режим передаёт `lambda _: None` — контракт обязан это переживать."""
    async def factory(progress):
        progress('diarize')
        progress({'t': 'stage.start', 'stage': 'diarize'})
        return 'ok'

    assert await factory(lambda _: None) == 'ok'


async def test_events_are_numbered_and_served_by_cursor():
    async def factory(progress):
        for i in range(5):
            progress({'t': 'chunk.done', 'i_chunk': i, 'say': f'pass2 {i}/5'})
        return {}

    _, job = await _run(factory, events=True)
    items, cursor, dropped = jobs.since(job, 0)
    assert [e['i'] for e in items] == [1, 2, 3, 4, 5], 'нумерация сплошная и монотонная'
    assert cursor == 5 and dropped == 0
    assert jobs.since(job, 3)[0] == items[3:], 'курсор отдаёт только то, чего клиент не видел'
    assert job['progress'] == 'pass2 4/5', '`say` заодно обновляет человеческую строку'


async def test_a_lagging_client_is_told_how_much_it_missed(monkeypatch):
    """⚠️ Пропуск СЧИТАЕМ и говорим. Промолчать — значит показать плавную картинку с дырой
    посередине, и никто не поймёт, почему у записи «потерялись» куски."""
    monkeypatch.setattr(jobs, 'RING', 4)
    monkeypatch.setattr(jobs, 'TAIL', 4)

    async def factory(progress):
        for i in range(10):
            progress({'t': 'chunk.done', 'i_chunk': i})
        return {}

    _, job = await _run(factory, events=True)
    items, cursor, dropped = jobs.since(job, 0)
    assert [e['i'] for e in items] == [7, 8, 9, 10]
    assert dropped == 6 and cursor == 10


async def test_the_tail_survives_the_end_of_the_job(monkeypatch):
    """Последний опрос клиента приходит уже после конца прогона — концовка обязана его дождаться."""
    monkeypatch.setattr(jobs, 'TAIL', 3)

    async def factory(progress):
        for i in range(20):
            progress({'t': 'chunk.done', 'i_chunk': i})
        return {}

    _, job = await _run(factory, events=True)
    assert [e['i'] for e in jobs.since(job, 0)[0]] == [18, 19, 20]


async def test_a_broken_event_never_reaches_the_ring_as_a_string():
    async def factory(progress):
        progress('final-round')
        progress({'t': 'turn.fix', 'was': 'эйр флоу', 'now': 'Airflow', 'ok': True})
        return {}

    _, job = await _run(factory, events=True)
    assert job['progress'] == 'final-round'
    assert [e['t'] for e in jobs.since(job, 0)[0]] == ['turn.fix']


@pytest.fixture(autouse=True)
def _clean():
    yield
    jobs._JOBS.clear()
