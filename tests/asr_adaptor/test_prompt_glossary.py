"""Выключатель свободного глоссария в подсказке пасса-2 доезжает до сборщика подсказки."""
from __future__ import annotations

import pipeline
from conveyor.run import run_conveyor
from test_pipeline_recovery import PASS1, backend, wav  # noqa: F401 — фикстуры


async def test_glossary_switch_reaches_the_prompt_builder(backend, wav, monkeypatch):
    seen: list[dict] = []

    def fake_build(canon, counter, budget, always=(), hinted=frozenset(), **kw):
        seen.append(dict(kw))
        return ''

    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline, 'build_prompt', fake_build)
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', 'none')
    await run_conveyor(str(wav), llm=None, episode='ep1')
    assert seen and all(kw.get('free_latin') is False for kw in seen)

    seen.clear()
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', 'clean')
    await run_conveyor(str(wav), llm=None, episode='ep1')
    assert seen and all(kw.get('conflict_free') is True and 'free_latin' not in kw for kw in seen)

    seen.clear()
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', 'substitute')
    await run_conveyor(str(wav), llm=None, episode='ep1', hints={'terms': ['Постгрес']})
    assert seen and all(isinstance(kw.get('substitute'), dict) and 'постгрес' in kw['substitute'] for kw in seen)

    seen.clear()
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', 'all')
    await run_conveyor(str(wav), llm=None, episode='ep1')
    assert seen and all(not ({'free_latin', 'conflict_free', 'substitute'} & set(kw)) for kw in seen)   # умолчание — прежняя сигнатура
