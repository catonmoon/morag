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
    return Place('chunk:1', 'arbitrate', item={'raw': 'это прегресс тут', 'start': 20.0, 'end': 35.0},
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


# --- финал-раунд: правила ------------------------------------------------------------------------

def _final_place() -> Place:
    turn = {'raw': 'это прегресс тут', 'start': 20.0, 'segments': [{'start': 20.0, 'end': 35.0, 'text': 'это прегресс тут'}]}
    return Place('turn:0', 'final', item=turn, canon=set(), gate_terms=['Postgres'], audio_sec=90.0)


class FakeFinal:
    def __init__(self, place: Place, *, said='was', fixes=True):
        self.place, self.said, self.fixes = place, said, fixes
        self.calls: list = []
        self.meter: dict = {}

    def schemas(self):
        return []

    async def call(self, name, **args):
        self.calls.append(name if not args else (name, args))
        f = self.place.fixes
        if name == 'recall':
            return {'recalled': 'регресс'}
        if name == 'correct_turn':
            self.place.fixes = ([{'was': 'прегресс', 'now': 'регресс', 'ok': False, 'why': 'known_term'}]
                                if self.fixes else [])
            return {'text': 'это прегресс тут', 'fixes': self.place.fixes, 'changed': False}
        if name == 'listen':
            return {'word': args['word'], 'found': True, 'text': 'это регресс тут', 'a': 5.0, 'b': 35.0}
        if name == 'sound_prefers':
            if self.said != 'now':
                f[0]['why'] = f'known_term (ear: {self.said})'
            return {'said': self.said}
        if name == 'apply_fix':
            f[0].update(ok=True, why='known_term→ear:now')
            return {'applied': True}
        if name == 'finish':
            return dict(args)
        raise ToolError(f'нет инструмента «{name}»')


async def test_final_rules_recall_then_correct_then_apply():
    place = _final_place()
    fake = FakeFinal(place)
    fin = await decide_place(place, RulePolicy(_cfg(protect_known=False, final_ear=False)), fake, 12)
    assert fake.calls == ['recall', 'correct_turn'] and fin.decision == 'apply'


async def test_final_rules_listen_to_a_rejected_known_word_and_apply_only_when_the_sound_agrees():
    cfg = _cfg(protect_known=True, final_ear=True)
    place = _final_place()
    fake = FakeFinal(place, said='now')
    fin = await decide_place(place, RulePolicy(cfg), fake, 12)
    assert fake.calls == ['recall', 'correct_turn', ('listen', {'word': 'прегресс'}),
                          ('sound_prefers', {'was': 'прегресс', 'now': 'регресс'}),
                          ('apply_fix', {'was': 'прегресс', 'now': 'регресс'})]
    assert fin.decision == 'apply' and place.fixes[0]['why'] == 'known_term→ear:now'

    place = _final_place()
    fake = FakeFinal(place, said='was')
    fin = await decide_place(place, RulePolicy(cfg), fake, 12)
    assert fake.calls[-1] == ('sound_prefers', {'was': 'прегресс', 'now': 'регресс'})
    assert fin.decision == 'apply' and place.fixes[0]['why'] == 'known_term (ear: was)'

    place = _final_place()
    fake = FakeFinal(place, fixes=False)
    fin = await decide_place(place, RulePolicy(cfg), fake, 12)
    assert fake.calls == ['recall', 'correct_turn'] and fin.decision == 'apply'


# --- LLM-оркестратор ------------------------------------------------------------------------------

from graph.policy import LLMPolicy  # noqa: E402


def _tc(name, args, cid='c1'):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': args}}


class ScriptedLLM:
    """Отвечает по сценарию: список ответов модели, по одному на вызов."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    async def complete_with_tools(self, messages, tools, **kw):
        self.seen.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {'choices': [{'message': item, 'finish_reason': 'tool_calls' if item.get('tool_calls') else 'stop'}]}


async def test_llm_policy_drives_the_tools_it_chose_and_feeds_results_back():
    import json
    llm = ScriptedLLM([
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('reader_flags', '{}', 'a')]},
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('listen', '{"ear": "second", "window": "same"}', 'b')]},
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('finish', '{"decision": "apply", "why": "ок"}', 'c')]},
    ])
    fake = Fake(suspicious=['прегресс'])
    fake.schemas = lambda: [{'type': 'function', 'function': {'name': 'x'}}]
    place = _place()
    fin = await decide_place(place, LLMPolicy(_cfg(), llm), fake, 8)
    assert fin.decision == 'apply' and fin.why == 'ок'
    assert [c if isinstance(c, str) else c[0] for c in fake.calls] == ['reader_flags', 'listen', 'finish']
    # второй вызов модели видит результат первого инструмента под его id
    second = llm.seen[1]
    assert second[0]['role'] == 'system' and 'оркестратор' in second[0]['content']
    assert 'кусок' in second[1]['content'] and 'прегресс' in second[1]['content']
    assert second[-1]['role'] == 'tool' and second[-1]['tool_call_id'] == 'a'
    assert json.loads(second[-1]['content'])['suspicious'] == ['прегресс']
    assert fake.meter['orchestrator_calls'] == 3


async def test_llm_policy_reports_bad_arguments_and_asks_again_then_gives_up_softly():
    llm = ScriptedLLM([
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('listen', '{"ear": ', 'a')]},   # битый JSON
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('finish', '{"decision": "skip"}', 'b')]},
    ])
    fake = Fake()
    fake.schemas = lambda: []
    fin = await decide_place(_place(), LLMPolicy(_cfg(), llm), fake, 8)
    assert fin.decision == 'skip'
    assert fake.meter['orchestrator_bad_args'] == 1
    err = llm.seen[1][-1]
    assert err['role'] == 'tool' and 'не JSON' in err['content']


async def test_llm_policy_without_a_tool_call_or_with_a_dead_model_leaves_the_place_as_is():
    fake = Fake()
    fake.schemas = lambda: []
    fin = await decide_place(_place(), LLMPolicy(_cfg(), ScriptedLLM([{'role': 'assistant', 'content': 'готово'}])), fake, 8)
    assert fin.decision == 'skip' and 'готово' in fin.why and fake.calls == []
    fin = await decide_place(_place(), LLMPolicy(_cfg(), ScriptedLLM([RuntimeError('шлюз')])), fake, 8)
    assert fin.decision == 'skip' and 'RuntimeError' in fin.why


async def test_llm_policy_bad_tool_name_becomes_an_observation_not_a_crash():
    llm = ScriptedLLM([
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('guess', '{}', 'a')]},
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('finish', '{"decision": "skip", "why": "нет такого"}', 'b')]},
    ])
    fake = Fake()
    fake.schemas = lambda: []
    place = _place()
    fin = await decide_place(place, LLMPolicy(_cfg(), llm), fake, 8)
    assert fin.decision == 'skip'
    assert 'error' in place.history[0][2] and 'guess' in place.history[0][2]['error']
