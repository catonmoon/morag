"""Политика решения по месту: что звать и когда остановиться.

`RulePolicy` — сегодняшняя последовательность конвейера, записанная явными шагами: арбитраж —
ворота читателя → второе ухо → [чистое ухо] → правила → [третий голос по требованию] →
применить; финал-раунд — вспомнить сущности → предложить замены → [для отвергнутых замен
известных слов: послушать → рассудить звуком → применить]. С ней конвейер обязан совпадать с
прежним линейным байт в байт — это держит золотой тест (снимок `tests/asr_adaptor/golden/`).

LLM-оркестратор по месту (один вызов модели — один шаг) был и снят 29.09 (ADR-0031): решения по
тексту модель принимает только в узле редактора, страницей, а не куском.

⚠️ Ни одна политика не пишет текст. Замена ложится только из правил и свидетелей (звук, канон)
под вето в коде — модель может ПРЕДЛАГАТЬ, подтверждать вправе только звук, канон и экран.
"""
from __future__ import annotations

import logging
from typing import Protocol

from conveyor.loop import Action, Finish, Place

log = logging.getLogger('asr')


class Policy(Protocol):
    kind: str

    async def next(self, place: Place) -> Action | Finish: ...

    def fallback(self, place: Place) -> Finish: ...


# --- правила ------------------------------------------------------------------------------------

class RulePolicy:
    """Правила конвейера как явные шаги. Читает конфиг стадии, ничего не помнит между местами."""

    kind = 'rule'

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    async def next(self, place: Place) -> Action | Finish:
        if place.kind == 'arbitrate':
            return self._arbitrate(place)
        if place.kind == 'final':
            return self._final(place)
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

    def _final(self, place: Place) -> Action | Finish:
        cfg = self.cfg
        if place.last('recall') is None:
            return Action('recall')
        if place.last('correct_turn') is None:
            return Action('correct_turn')
        if cfg.protect_known and cfg.final_ear:
            # Замену ИЗВЕСТНОГО слова сторож отверг — но финал-раунд иногда чинит прайминг (звук
            # за замену в 10 случаях из 104). Слушаем чистым ухом; звук ближе к замене — берём.
            for f in place.fixes:
                if f.get('why') != 'known_term':
                    continue
                was, now = f['was'], f['now']
                if place.last('listen', word=was) is None:
                    return Action('listen', {'word': was})
                said = place.last('sound_prefers', was=was, now=now)
                if said is None:
                    return Action('sound_prefers', {'was': was, 'now': now})
                if (said or {}).get('said') == 'now' and place.last('apply_fix', was=was, now=now) is None:
                    return Action('apply_fix', {'was': was, 'now': now})
        return Finish('apply', 'правила')


def make_policy(cfg, llm=None, kind: str = '') -> Policy:
    """Правила конвейера. LLM-оркестратор по месту снят 29.09: 1 745 вызовов модели на 88 минут без
    доказанного выигрыша (ADR-0031); решения по тексту модель принимает только в узле редактора."""
    return RulePolicy(cfg)
