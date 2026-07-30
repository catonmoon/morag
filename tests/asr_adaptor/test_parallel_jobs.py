"""Параллельные выпуски: реестр строго в порядке поступления, ресурсы гейтятся.

Выигрыш параллели — конвейеризация стадий (GPU выпуска B при LLM выпуска A), а детерминизм
нумерации Speaker_N держится на цепочке билетов: даже если B обогнал A по стадиям, к реестру
он подойдёт после A.
"""
import asyncio

import pipeline


async def test_registry_order_follows_submission_order(monkeypatch):
    order = []

    async def fake_episode(name, slow):
        t = pipeline._take_ticket()
        try:
            if slow:
                await asyncio.sleep(0.05)      # A медленный — B обгоняет по стадиям
            await pipeline._wait_turn(t)
            order.append(name)
        finally:
            pipeline._release_turn(t)

    await asyncio.gather(fake_episode('A', slow=True), fake_episode('B', slow=False))

    assert order == ['A', 'B']


async def test_crashed_episode_does_not_block_the_queue():
    async def crasher():
        t = pipeline._take_ticket()
        try:
            raise RuntimeError('упал до реестра')
        finally:
            pipeline._release_turn(t)

    async def follower():
        t = pipeline._take_ticket()
        try:
            await asyncio.wait_for(pipeline._wait_turn(t), timeout=1.0)
            return 'прошёл'
        finally:
            pipeline._release_turn(t)

    results = await asyncio.gather(crasher(), follower(), return_exceptions=True)

    assert isinstance(results[0], RuntimeError)
    assert results[1] == 'прошёл'


async def test_resource_semaphore_caps_concurrency():
    inflight, peak = 0, 0

    async def use(name):
        nonlocal inflight, peak
        async with pipeline._res('тест-ресурс', 1):
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.01)
            inflight -= 1

    await asyncio.gather(*(use(f'j{i}') for i in range(4)))

    assert peak == 1


async def test_slow_episode_survives_neighbours_finishing_first():
    """Гонка, стоившая ep3 получаса: релиз на стадии реестра прошёл, соседи убежали вперёд и в
    старой версии прибирали его билет (pop(t-2)) — повторный релиз из finally падал на KeyError,
    перекрывая ГОТОВЫЙ результат. Теперь релиз идемпотентен, чужие билеты никто не трогает."""
    t0, t1, t2, t3 = (pipeline._take_ticket() for _ in range(4))

    pipeline._release_turn(t0)
    await pipeline._wait_turn(t1)
    pipeline._release_turn(t1)      # стадия реестра медленного выпуска пройдена
    await pipeline._wait_turn(t2)
    pipeline._release_turn(t2)      # быстрые соседи финишируют и прибирают за собой
    await pipeline._wait_turn(t3)
    pipeline._release_turn(t3)

    pipeline._release_turn(t1)      # finally медленного: не должен упасть и никого не ждёт
