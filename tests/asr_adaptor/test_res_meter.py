"""Прибор ресурсов: ожидание и занятость бэкендов — в `timing` своего выпуска, раздельно."""
from __future__ import annotations

import asyncio

import pipeline
from conveyor.run import run_conveyor
from fakes import HINTS


async def test_run_meters_waiting_and_holding_of_each_backend(rich, silence):
    r = await run_conveyor(str(silence), llm=object(), episode='ep1', hints=HINTS)
    tm = r['timing']
    for name in ('diarize', 'whisper', 'campp'):
        assert f'busy_{name}_s' in tm and f'wait_{name}_s' in tm, name
    assert 'speakers_s' in tm and 'ticket_wait_s' in tm
    assert pipeline.RES_METER.get() is None, 'счётчик выпуска не утёк в вызывающий контекст'


async def test_meter_counts_queueing_separately_from_work():
    tm: dict = {}
    pipeline.RES_METER.set(tm)

    async def use():
        async with pipeline._res('тест-счёт', 1):
            await asyncio.sleep(0.05)

    await asyncio.gather(use(), use())
    assert tm['busy_тест-счёт_s'] >= 0.09, 'занятость обоих'
    assert 0.04 <= tm['wait_тест-счёт_s'] < 0.09, 'ждал только второй'
