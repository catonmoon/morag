"""Выключатель свободного глоссария в подсказке пасса-2 доезжает до сборщика подсказки."""
from __future__ import annotations

import pipeline
from test_pipeline_recovery import PASS1, backend, wav  # noqa: F401 — фикстуры


async def test_glossary_switch_reaches_the_prompt_builder(backend, wav, monkeypatch):
    seen: list[dict] = []

    def fake_build(canon, counter, budget, always=(), hinted=frozenset(), **kw):
        seen.append(dict(kw))
        return ''

    monkeypatch.setattr(pipeline, '_RES_SEMS', {})
    monkeypatch.setattr(pipeline, 'build_prompt', fake_build)
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', False)
    await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')
    assert seen and all(kw.get('free_latin') is False for kw in seen)

    seen.clear()
    monkeypatch.setattr(pipeline.CFG, 'prompt_glossary', True)
    await pipeline.run_pipeline(str(wav), llm=None, episode='ep1')
    assert seen and all('free_latin' not in kw for kw in seen)      # умолчание — прежняя сигнатура
