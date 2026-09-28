"""Политика решения по месту: что звать и когда остановиться.

`RulePolicy` — сегодняшняя последовательность конвейера, записанная явными шагами (ворота
читателя → второе ухо → [чистое ухо] → правила → [третий голос по требованию] → применить).
Граф с ней обязан совпадать с линейным конвейером байт в байт — это держит золотой тест.
`LLMPolicy` (шаг 3) — оркестратор через function calling: выбирает те же инструменты сам.

⚠️ Ни одна политика не пишет текст. Замена ложится только из `arbitrate_rules`/`apply_swaps`
под вето в коде — модель может ПРЕДЛАГАТЬ, подтверждать вправе только звук, канон и экран.
"""
from __future__ import annotations

from typing import Protocol

from graph.loop import Action, Finish, Place


class Policy(Protocol):
    kind: str

    async def next(self, place: Place) -> Action | Finish: ...

    def fallback(self, place: Place) -> Finish: ...


class RulePolicy:
    """Правила конвейера как явные шаги. Читает конфиг стадии, ничего не помнит между местами."""

    kind = 'rule'

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    async def next(self, place: Place) -> Action | Finish:
        if place.kind == 'arbitrate':
            return self._arbitrate(place)
        return Finish('skip', f'политика правил не знает место «{place.kind}»')

    def fallback(self, place: Place) -> Finish:
        return Finish('skip', 'бюджет шагов исчерпан')

    def _arbitrate(self, place: Place) -> Action | Finish:
        cfg = self.cfg
        if cfg.arbitrate_gate == 'reader':
            gate = place.last('reader_flags')
            if gate is None:
                return Action('reader_flags')
            if not (gate or {}).get('suspicious'):
                return Finish('skip', 'читатель: кусок чист')
        if not place.ok('listen', ear='second'):
            return Action('listen', {'ear': 'second', 'window': 'same'})
        if cfg.clean_ear == 'always' and not place.ok('listen', ear='clean'):
            return Action('listen', {'ear': 'clean', 'window': 'same'})
        rules = place.last('arbitrate_rules')
        if rules is None:
            return Action('arbitrate_rules')
        if (cfg.clean_ear == 'demand' and (rules or {}).get('disputes')
                and not place.ok('listen', ear='clean')):
            # Третий голос по требованию: только если после второго уха остались споры, не
            # решённые ни правилом, ни свидетелем. Окном, не словом (ADR-0030).
            return Action('listen', {'ear': 'clean', 'window': cfg.clean_ear_window})
        return Finish('apply', 'правила')


def make_policy(cfg, llm=None) -> Policy:
    """`ASR_GRAPH_POLICY=rule|llm`. LLM-оркестратор появится шагом 3; до него — правила."""
    return RulePolicy(cfg)
