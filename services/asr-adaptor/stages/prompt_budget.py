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


def build_prompt(canonicals, counter: WhisperTokenCounter, budget: int = 200) -> str:
    """Промпт для podlodka: ТОЛЬКО латино-каноники (whisper их garble-ит → каноник помогает;
    русские имена кириллицей не праймим — латиница искажает Sbera/DERIPASKA)."""
    latin = [c for c in canonicals if _LAT.search(c)]
    return fit_prompt(latin, counter, budget)
