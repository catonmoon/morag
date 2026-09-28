"""Политика правил и цикл по месту: точная последовательность действий, бюджет, отказы."""
from __future__ import annotations

from types import SimpleNamespace

from graph.loop import Action, Finish, Place, decide_place
from graph.policy import RulePolicy
from graph.tools import Registry, Tool, ToolError


def _cfg(**over):
    base = dict(arbitrate_gate='', second_model='other', clean_ear='', clean_ear_window='chunk',
                arbitrate_ratio=100.0, whisper_slots=1, graph_place_steps=8)
    base.update(over)
    return SimpleNamespace(**base)


def _place() -> Place:
    return Place('chunk:1', 'arbitrate', chunk={'raw': 'это прегресс тут', 'start': 20.0, 'end': 35.0},
                 canon=set(), gate_terms=[], audio_sec=90.0, neighbours=(0.0, 50.0))


class Fake:
    """Реестр-заглушка: помнит порядок вызовов, отвечает по сценарию."""

    def __init__(self, *, suspicious=(), disputes=0):
        self.calls: list[tuple[str, dict]] = []
        self.suspicious = list(suspicious)
        self.disputes = disputes
        self.meter: dict = {}

    async def call(self, name, **args):
        self.calls.append((name, args))
        if name == 'reader_flags':
            return {'suspicious': self.suspicious}
        if name == 'listen':
            return {'ear': args['ear'], 'text': 'это регресс тут', 'a': 20.0, 'b': 35.0}
        if name == 'arbitrate_rules':
            return {'text': 'это регресс тут', 'decisions': [], 'disputes': self.disputes, 'taken': 0}
        if name == 'finish':
            return dict(args)
        raise ToolError(f'нет инструмента «{name}»')


async def _run(cfg, fake):
    place = _place()
    fin = await decide_place(place, RulePolicy(cfg), fake, cfg.graph_place_steps)
    return fin, [n if not a else (n, a) for n, a in fake.calls], place


async def test_rules_without_gate_listen_second_then_apply():
    fin, calls, _ = await _run(_cfg(), Fake())
    assert calls == [('listen', {'ear': 'second', 'window': 'same'}), 'arbitrate_rules']
    assert fin.decision == 'apply'


async def test_rules_with_clean_ear_always_listen_both_ears_on_the_same_window():
    fin, calls, _ = await _run(_cfg(clean_ear='always'), Fake())
    assert calls == [('listen', {'ear': 'second', 'window': 'same'}),
                     ('listen', {'ear': 'clean', 'window': 'same'}), 'arbitrate_rules']
    assert fin.decision == 'apply'


async def test_reader_gate_skips_a_clean_chunk_and_listens_to_a_flagged_one():
    fin, calls, _ = await _run(_cfg(arbitrate_gate='reader'), Fake(suspicious=[]))
    assert calls == ['reader_flags'] and fin.decision == 'skip' and 'читатель' in fin.why
    fin, calls, _ = await _run(_cfg(arbitrate_gate='reader'), Fake(suspicious=['прегресс']))
    assert calls[0] == 'reader_flags' and calls[1] == ('listen', {'ear': 'second', 'window': 'same'})
    assert fin.decision == 'apply'


async def test_third_voice_on_demand_only_when_a_dispute_remains_and_with_the_configured_window():
    fin, calls, _ = await _run(_cfg(clean_ear='demand', clean_ear_window='window30'), Fake(disputes=0))
    assert calls == [('listen', {'ear': 'second', 'window': 'same'}), 'arbitrate_rules']
    fin, calls, _ = await _run(_cfg(clean_ear='demand', clean_ear_window='window30'), Fake(disputes=1))
    assert calls == [('listen', {'ear': 'second', 'window': 'same'}), 'arbitrate_rules',
                     ('listen', {'ear': 'clean', 'window': 'window30'})]
    assert fin.decision == 'apply'


async def test_budget_exhaustion_leaves_the_place_as_is_and_is_counted():
    class Loop:
        kind = 'rule'

        async def next(self, place):
            return Action('listen', {'ear': 'second'})     # никогда не заканчивает

        def fallback(self, place):
            return Finish('skip', 'бюджет')

    fake = Fake()
    place = _place()
    fin = await decide_place(place, Loop(), fake, 3)
    assert fin.decision == 'skip' and place.exhausted and len(fake.calls) == 3
    assert fake.meter['budget_exhausted'] == 1


async def test_a_tool_error_becomes_an_observation_and_the_policy_can_recover():
    seen = []

    class Recovering:
        kind = 'x'

        async def next(self, place):
            last = place.history[-1][2] if place.history else None
            seen.append(last)
            if last is None:
                return Action('listen', {'ear': 'fourth'})       # неверный аргумент
            if isinstance(last, dict) and 'error' in last:
                return Action('listen', {'ear': 'second'})       # поправилась по подсказке
            return Finish('apply')

        def fallback(self, place):
            return Finish('skip')

    async def listen(ear: str) -> dict:
        return {'ear': ear, 'text': 'x', 'a': 0.0, 'b': 1.0}

    reg = Registry([Tool('listen', 'слушать', listen,
                         schema={'properties': {'ear': {'type': 'string', 'enum': ['second', 'clean']}},
                                 'required': ['ear']})], journal=[], meter={})
    fin = await decide_place(_place(), Recovering(), reg, 5)
    assert fin.decision == 'apply'
    assert 'error' in seen[1] and 'second, clean' in seen[1]['hint']


async def test_finish_tool_call_ends_the_place_with_its_decision():
    class ViaTool:
        kind = 'x'

        async def next(self, place):
            return Action('finish', {'decision': 'skip', 'why': 'нечего делать'})

        def fallback(self, place):
            return Finish('apply')

    fin = await decide_place(_place(), ViaTool(), Fake(), 4)
    assert fin.decision == 'skip' and fin.why == 'нечего делать'
