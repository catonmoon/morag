"""Постоянные термины записи (`hints.always`): в подсказке пасса-2 каждого куска, рядом с терминами корпуса."""
from __future__ import annotations

from types import SimpleNamespace

from fakes import HINTS
from conveyor.nodes import always_of
from conveyor.run import run_conveyor


def test_always_of_merges_corpus_and_record_terms_without_duplicates():
    cfg = SimpleNamespace(always_terms=('Kafka', 'Postgres'))
    assert always_of(cfg, {'always': ['Postgres', 'FRIDA', ' ', 7]}) == ['Kafka', 'Postgres', 'FRIDA']
    assert always_of(cfg, {}) == ['Kafka', 'Postgres']
    assert always_of(cfg, None) == ['Kafka', 'Postgres']


async def test_record_terms_reach_every_pass2_prompt(rich, silence, monkeypatch):
    import pipeline
    seen: list[tuple] = []
    monkeypatch.setattr(pipeline, 'build_prompt',
                        lambda terms, counter, budget, always=(), hinted=(): seen.append(tuple(always)) or 'каноники')
    await run_conveyor(str(silence), llm=object(), episode='ep1', hints={**HINTS, 'always': ['FRIDA']})
    assert seen and all('FRIDA' in a for a in seen), 'термин записи — в подсказке каждого куска'


async def test_without_record_terms_prompts_are_unchanged(rich, silence):
    await run_conveyor(str(silence), llm=object(), episode='ep1', hints=HINTS)
    before = [c['prompt'] for c in rich.calls]
    rich.calls.clear()
    await run_conveyor(str(silence), llm=object(), episode='ep1', hints={**HINTS, 'always': []})
    assert [c['prompt'] for c in rich.calls] == before
