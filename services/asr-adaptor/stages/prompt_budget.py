"""Бюджет Whisper initial_prompt (порт pass2_budget на morag `TokenCounter` ABC).

`WhisperTokenCounter(TokenCounter)` — точный счёт whisper-токенов (hard-лимит initial_prompt = 224;
держим запас ~200). `fit_prompt` набивает каноники до бюджета по ЦЕЛОМУ термину (термин не режем).
`build_prompt` — ветка podlodka: ТОЛЬКО латино-каноники (русские имена латиницей искажаются).
"""
from __future__ import annotations

import re
from functools import lru_cache

from morag.indexing.token_counter import TokenCounter

PREFIX = 'В разговоре про технологии упоминаются '
_LAT = re.compile('[a-zA-Z]')


@lru_cache(maxsize=2)
def _whisper_tok(model: str):
    from transformers import WhisperTokenizer
    return WhisperTokenizer.from_pretrained(model)


class WhisperTokenCounter(TokenCounter):
    """Счёт токенов Whisper-токенайзером (для бюджета initial_prompt)."""

    def __init__(self, model: str = 'openai/whisper-large-v3') -> None:
        self._tok = _whisper_tok(model)

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)['input_ids'])

    def truncate(self, text: str, limit: int) -> str:
        ids = self._tok(text, add_special_tokens=False)['input_ids']
        return text if len(ids) <= limit else self._tok.decode(ids[:limit])


def fit_prompt(terms, counter: WhisperTokenCounter, budget: int = 200, prefix: str = PREFIX) -> str:
    """Набить термины в бюджет (whisper-токены), по ЦЕЛОМУ термину. Пусто → ''."""
    terms = [t for t in terms if t]
    if not terms:
        return ''
    out = []
    for t in terms:
        if counter.count(prefix + ', '.join(out + [t]) + '.') > budget:
            break
        out.append(t)
    return (prefix + ', '.join(out) + '.') if out else ''


def build_prompt(canonicals, counter: WhisperTokenCounter, budget: int = 200,
                 always: tuple[str, ...] | list[str] = ()) -> str:
    """Промпт для podlodka: постоянные термины корпуса + латино-каноники куска.

    `always` — то, что не надо переоткрывать в каждом выпуске: имена ведущих, повторяющиеся
    продукты. Глоссарий строится заново на каждом выпуске и на устойчивом гарбле срывается —
    замерено на корпусе: «Колодзев» вместо «Колодезев» в 14 репликах, `MotorMost` вместо
    `Mattermost` в 24. Финал-раунд их тоже не чинит.

    Кириллица в подсказке РАБОТАЕТ, вопреки прежнему «только латиница» — проверено A/B на интро
    ep11: без подсказки и с латинской «Дмитрий Колодзев», с кириллическим именем «Дмитрий
    Колодезев». Смешанная подсказка (имена + латинские термины) тоже не ломает ни то, ни другое.
    Постоянные термины идут ПЕРВЫМИ: бюджет ≤200 whisper-токенов, и лучше потерять хвост
    случайных канонико́в куска, чем фамилию ведущего.
    """
    latin = [c for c in canonicals if _LAT.search(c)]
    terms = list(dict.fromkeys([t for t in always if t] + latin))
    return fit_prompt(terms, counter, budget)
