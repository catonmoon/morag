"""Глоссарий выпуска + per-chunk отбор (порт adventures/podlodka-asr/pass2_gloss.py на morag LLMClient).

LLM по тексту пасс-1 → [{heard, canonicals:[...]}]: heard — как записано в черновике, canonicals —
НАБОР гипотез (одна, если уверен; список, если нет — акустика выберет). Кривой каноник безвреден.
LLM-вызовы — `LLMClient.complete_json` (structured output, reasoning off).
"""
from __future__ import annotations

import re

try:
    from wordfreq import zipf_frequency as _zipf
    _HAS_WF = True
except ImportError:
    _HAS_WF = False

_SYS = (
    'Ты обобщаешь расшифровку любого разговора. СНАЧАЛА определи ТЕМЫ и доменную лексику фрагмента '
    '(IT, финансы, медицина, логистика — что угодно). ЗАТЕМ выпиши термины, записанные НЕКАНОНИЧНО, '
    'ЗАЗЕМЛЯЯ догадку в тему. Для каждого: "heard" — как ИМЕННО записано (с соседним словом для '
    'однозначности: «аж двести», не «двести») и "canonicals" — СПИСОК правдоподобных каноник-форм:\n'
    '— уверен → одна форма (часто латиницей; пример формата: heard «джапити», canonicals ["ChatGPT"]);\n'
    '— НЕ уверен (какая серия модели / какая компания) → ПЕРЕЧИСЛИ варианты, акустика выберет позже '
    '(heard «стенгаус» → ["Westinghouse","Alstom"]);\n'
    '— ПРОГОВОРЕННЫЕ НОМЕРА моделей: сохрани ПРОИЗНЕСЁННОЕ ЧИСЛО, варьируй только серию '
    '(heard «аж двести» → ["H200","A200"]; «а сто» → ["H100","A100","V100"]). Число НЕ меняй.\n'
    'НЕ выдумывай вне тем; обычные слова/числа без доменного смысла, переводы (китайцы→China), '
    'общеизвестные аббревиатуры (МГУ, ФНС, ИИ) — НЕ включай. Сортируй по важности. Верни СТРОГО '
    'JSON {"terms":[{"heard":..,"canonicals":[..]}]}, без иного текста.'
)

# Кап на glossary-вызов: реальный глоссарий батча (≤8000 симв.) — ~20-50 терминов ≈ <1.5k токенов.
# Бьёт по деген-петлям повторов (Grok reasoning-off иногда штампует один термин до упора): обрубает
# за секунды вместо минут. При обрезке батч может стать невалидным JSON → скип (редко; дедуп по другим
# батчам страхует). Без капа один залипший выпуск жёг ~5+ мин и тысячи токенов на повторах.
_GLOSSARY_MAX_TOKENS = 3000

_SCHEMA = {
    'type': 'object',
    'properties': {'terms': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'heard': {'type': 'string'},
                       'canonicals': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['heard', 'canonicals']}}},
    'required': ['terms'],
}


def _norm(s: str) -> str:
    return re.sub(r'[^0-9a-zа-яё]', '', s.lower())


def _keep_one(heard: str, canonical: str) -> bool:
    """Оставлять ли каноник: heard РЕДКОЕ/garbled (zipf низкий) или латиница. Guard на абзац-в-канонике."""
    if not canonical or len(canonical) > 40 or len(heard) > 40 or '\n' in canonical:
        return False
    if re.search('[a-zA-Z]', heard) or (
            re.search('[a-zA-Z0-9]', canonical) and not re.search('[а-яё]', _norm(canonical))):
        return True
    if not _HAS_WF:
        return True
    toks = [t for t in re.split(r'[^а-яё]+', heard.lower()) if len(t) > 2]
    return all(_zipf(t, 'ru') < 3.0 for t in toks) if toks else True


def _sentence_batches(text: str, max_chars: int = 8000):
    """Батчи ≤max_chars по границам ПРЕДЛОЖЕНИЙ (не рвём предложение/слово)."""
    sents = re.split(r'(?<=[.!?…])\s+', text.strip())
    cur, n = [], 0
    for s in sents:
        if n + len(s) > max_chars and cur:
            yield ' '.join(cur)
            cur, n = [], 0
        cur.append(s); n += len(s) + 1
    if cur:
        yield ' '.join(cur)


async def build_glossary(full_text: str, llm) -> list[dict]:
    """[{heard, canonicals:[...]}] по важности, дедуп по heard. llm — morag LLMClient."""
    seen, out = set(), []
    for batch in _sentence_batches(full_text):
        try:
            res = await llm.complete_json(
                [{'role': 'system', 'content': _SYS}, {'role': 'user', 'content': batch}],
                schema=_SCHEMA, schema_name='glossary', max_tokens=_GLOSSARY_MAX_TOKENS)
        except Exception:
            continue
        for r in (res or {}).get('terms', []):
            if not (isinstance(r, dict) and r.get('heard') and r.get('canonicals')):
                continue
            heard = str(r['heard']).strip()
            cans = r['canonicals'] if isinstance(r['canonicals'], list) else [r['canonicals']]
            cans = [str(c).strip() for c in cans if c and _keep_one(heard, str(c).strip())]
            key = heard.lower()
            if cans and key not in seen:
                seen.add(key)
                out.append({'heard': heard, 'canonicals': cans})
    return out


def relevant(chunk_text: str, glossary: list[dict]) -> list[str]:
    """НАБОРЫ каноников терминов, чья heard-форма встречается в чанке. Порядок = важность, дедуп."""
    ctoks = set(_norm(w) for w in chunk_text.split())
    cnorm = _norm(chunk_text)
    res, seen = [], set()
    for g in glossary:
        hn = _norm(g['heard'])
        if not hn:
            continue
        hit = hn in ctoks or (len(hn) >= 4 and hn in cnorm) or \
            any(_norm(c) in cnorm for c in g['canonicals'])
        if hit:
            for c in g['canonicals']:
                if c.lower() not in seen:
                    seen.add(c.lower()); res.append(c)
    return res
