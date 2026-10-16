"""Общий замок GPU (`ASR_GPU_GATE`, ADR-0033 §6): ресурсы на GPU разных задач не пересекаются, CPU-полоса свободна."""
from __future__ import annotations

import asyncio

import config
import pipeline


async def _peak(names: list[str], hold: float = 0.05) -> tuple[int, dict]:
    """Пик одновременных держателей ресурсов `names` и счётчик выпуска."""
    tm: dict = {}
    pipeline.RES_METER.set(tm)
    state = {'now': 0, 'peak': 0}

    async def use(name: str):
        async with pipeline._res(name, 1):
            state['now'] += 1
            state['peak'] = max(state['peak'], state['now'])
            await asyncio.sleep(hold)
            state['now'] -= 1

    await asyncio.gather(*(use(n) for n in names))
    pipeline.RES_METER.set(None)
    return state['peak'], tm


async def test_gate_serialises_gpu_resources_of_different_tasks(monkeypatch):
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.CFG, 'gpu_gate', True)
    monkeypatch.setattr(pipeline.CFG, 'gpu_slots', 1)
    peak, tm = await _peak(['whisper', 'diarize', 'align'])
    assert peak == 1, 'whisper, диаризация и выравнивание — по одному на GPU'
    assert tm['wait_gpu_s'] > 0.08, 'двое ждали родителя'
    assert 'gpu' in pipeline._RES_SEMS


async def test_gate_leaves_cpu_lane_and_campp_alone(monkeypatch):
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.CFG, 'gpu_gate', True)
    monkeypatch.setattr(pipeline.CFG, 'lanes', {'cpu': config.Lane('cpu', 'cpu', 1, ('http://c',)),
                                                 'g2': config.Lane('g2', 'gpu', 1, ('http://g',))})
    peak, _ = await _peak(['whisper', 'cpu', 'campp'])
    assert peak == 3, 'CPU-полоса и CAM++ под замок не попадают'
    peak, _ = await _peak(['whisper', 'g2'])
    assert peak == 1, 'полоса с устройством gpu — под замком'


async def test_without_gate_resources_are_independent(monkeypatch):
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.CFG, 'gpu_gate', False)
    peak, tm = await _peak(['whisper', 'diarize'])
    assert peak == 2 and 'wait_gpu_s' not in tm and 'gpu' not in pipeline._RES_SEMS


async def test_gate_with_two_slots_and_pass2_concurrency_does_not_deadlock(monkeypatch):
    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline.CFG, 'gpu_gate', True)
    monkeypatch.setattr(pipeline.CFG, 'gpu_slots', 2)
    tm: dict = {}
    pipeline.RES_METER.set(tm)

    async def chunk():
        async with pipeline._res('whisper', 2):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(asyncio.gather(*(chunk() for _ in range(6))), timeout=2.0)
    pipeline.RES_METER.set(None)
    assert tm['busy_whisper_s'] >= 0.05
