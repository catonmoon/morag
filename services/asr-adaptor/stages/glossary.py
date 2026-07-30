"""Глоссарий выпуска + per-chunk отбор (порт adventures/podlodka-asr/pass2_gloss.py на morag LLMClient).

LLM по тексту пасс-1 → [{heard, canonicals:[...]}]: heard — как записано в черновике, canonicals —
НАБОР гипотез (одна, если уверен; список, если нет — акустика выберет). Кривой каноник безвреден.
LLM-вызовы — `LLMClient.complete_json` (structured output, reasoning off).
"""
from __future__ import annotations

import asyncio
import logging
import re

try:
    from wordfreq import zipf_frequency as _zipf
    _HAS_WF = True
except ImportError:  # без частотника гейт редкости не работает — а он несущий, см. _keep_one
    _HAS_WF = False
    logging.getLogger('asr').warning(
        'wordfreq не установлен: глоссарий примет ЛЮБОЙ термин, включая переводы обычных слов '
        '(«агенты»→agents). Поставьте wordfreq (он в requirements.txt).')

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
    'ЗАПРЕЩЕНЫ ПЕРЕВОДЫ. Каноник — это ПРАВИЛЬНОЕ НАПИСАНИЕ услышанного, а не английский эквивалент: '
    '«агенты»→agents, «холодный бэкап»→cold backup, «способности»→capabilities, «китайцы»→China — '
    'НЕЛЬЗЯ. Русское слово, записанное верно, в глоссарий не попадает вовсе — даже если у него есть '
    'английский аналог. Латинский каноник уместен, только когда сущность и в русском тексте пишется '
    'латиницей (джапити→ChatGPT, эн-видиа→NVIDIA).\n'
    'НЕ выдумывай вне тем; обычные слова/числа без доменного смысла, общеизвестные аббревиатуры '
    '(МГУ, ФНС, ИИ) — НЕ включай. Сортируй по важности. Верни СТРОГО '
    'JSON {"terms":[{"heard":..,"canonicals":[..]}]}, без иного текста.'
)

# Потолок — ПРЕДОХРАНИТЕЛЬ, не экономия: самый жирный честный глоссарий батча ~1500 токенов,
# запас пятикратный — до 8000 доходит только мусор деген-петли, и она обрубается за ~1.5 минуты.
# Совсем без капа (пробовали) деген льёт сотни КБ до таймаута клиента (180с), поверх которого SDK
# сам ретраит таймауты — один залипший батч жёг до 15 минут, глоссарий ep1 сидел 24 минуты.
# Прежний тесный кап (3000) был другой крайностью: обрезал ЧЕСТНЫЕ батчи → невалидный JSON →
# молчаливая потеря трети глоссария. Обрезанный мусорный батч добирает ретрай (_one_batch).
_GLOSSARY_MAX_TOKENS = 8000

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
    """Оставлять ли каноник: `heard` должно быть РЕДКИМ, то есть похожим на гарбл.

    Гейт по редкости — несущий: гарбл редок («стенгаус», «джапити», «василедец»), а обычное русское
    слово частотно. Раньше ЛЮБОЙ латинский каноник проходил мимо гейта, и через эту дыру шли
    переводы: «агенты»→agents (zipf 4.0), «умные очки»→smart glasses (3.9), «способностями»→
    capabilities (3.7). Замерено на корпусе: 2069 замен кириллицы на латиницу против 56 обратных.
    Промптом это не лечится — проверено на живом прогоне, стало даже чуть хуже (65→69).

    Гейт идёт по КИРИЛЛИЧЕСКИМ токенам `heard`, даже если рядом стоит латиница: «frontier
    способностями»→frontier capabilities проходило именно через смешанную форму. Если кириллицы
    в `heard` нет вовсе — ASR уже написал латиницей, и мы канонизируем написание, а не переводим.
    """
    if not canonical or len(canonical) > 40 or len(heard) > 40 or '\n' in canonical:
        return False
    if re.search(r'\d', canonical):
        # Обозначение модели: число проговаривают ОБЫЧНЫМИ словами, поэтому `heard` тут всегда
        # частотный («аж двести»→H200, «а сто»→A100, «десять-восемьдесят»→1080). Гейт редкости
        # такие кейсы вырезает подчистую — а это флагман Класса-2, ради которого схема и строилась.
        return True
    if not is_common_ru(heard):
        return True  # редкое слово = гарбл, ради него глоссарий и существует
    # `heard` частотный — сам по себе это ещё не приговор: промпт просит писать соседнее слово для
    # однозначности («институт Айри»), и обычный сосед не должен убивать запись. Приговор — когда
    # каноник ТОЖЕ обычное слово, только английское: это перевод, а не канонизация.
    return not is_plain_english(canonical)


def is_common_ru(text: str, threshold: float = 3.0) -> bool:
    """Есть ли в тексте ЧАСТОТНОЕ русское слово, то есть обычная речь, а не гарбл.

    Единственное определение «частотности» в конвейере: им гейтится глоссарий и им же финал-раунд
    отличает канонизацию от перевода. Без частотника — False: гейты размыкаются, а не срабатывают
    наугад (см. предупреждение при импорте).
    """
    if not _HAS_WF:
        return False
    toks = [t for t in re.split(r'[^а-яё]+', text.lower()) if len(t) > 2]
    return any(_zipf(t, 'ru') >= threshold for t in toks)


def is_plain_english(text: str, threshold: float = 3.5) -> bool:
    """Обычная английская фраза, а не имя: строчные слова, все частотные в английском.

    Главный признак — КАПИТАЛИЗАЦИЯ, а не частота: имя пишут с большой буквы («Tensor Train»,
    «Claude», «Sam Altman») либо капсом («AIRI», «NVIDIA»), а перевод строчный («cold backup»,
    «smart glasses», «frontier capabilities»). Одной частоты не хватает: Claude 3.8 в английском
    частотнее, чем hallucination 2.9, — по ней имя и перевод не разделить.

    Частота добивает остаток: строчный, но редкий токен — это термин, а не перевод (guardrails 2.1,
    inference 3.4). Известный промах: «галлюцинации»→hallucination (2.9) под порог не попадает.
    """
    if not _HAS_WF:
        return False
    toks = re.findall(r"[A-Za-z]+", text)
    if not toks or any(t[0].isupper() and not t.isupper() for t in toks):
        return False  # есть слово с Заглавной — это имя собственное
    lower = [t for t in toks if t.islower() and len(t) > 2]
    if not lower:
        return False  # одни аббревиатуры капсом (AIRI, NVIDIA) — тоже имя
    return all(_zipf(t, 'en') >= threshold for t in lower)


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


async def _one_batch(llm, batch: str, tag: str, tries: int = 2) -> list[dict]:
    """Один батч с одним ретраем: деген-петля даёт битый JSON (ловили 125КБ на 9253 строках),
    повтор обычно чистый — генерация недетерминирована."""
    for attempt in range(1, tries + 1):
        try:
            res = await llm.complete_json(
                [{'role': 'system', 'content': _SYS}, {'role': 'user', 'content': batch}],
                schema=_SCHEMA, schema_name='glossary', max_tokens=_GLOSSARY_MAX_TOKENS)
            return (res or {}).get('terms') or []
        except Exception as e:
            logging.getLogger('asr').warning(
                'glossary: батч %s, попытка %d — %s: %s', tag, attempt,
                type(e).__name__, str(e)[:120])
    return []


async def build_glossary(full_text: str, llm, passes: int = 2) -> list[dict]:
    """[{heard, canonicals:[...]}] по важности, дедуп по heard. llm — morag LLMClient.

    Каждый батч зовётся `passes` раз, результаты объединяются. Замерено на ep20 (один и тот же
    текст, temperature=0, seed=42): одиночные прогоны дают 62-96 терминов с пересечением всего 32 —
    провайдерская лотерея; объединение двух — 128. Recall аддитивен, кривой лишний каноник
    безвреден по построению (выбирает акустика), а узкое место схемы — 200 токенов подсказки
    Whisper, которые надо кормить лучшими канониками. Все вызовы идут ПАРАЛЛЕЛЬНО (батчи и проходы
    независимы); порядок слияния стабилен: батч за батчем, проход за проходом.
    """
    batches = list(_sentence_batches(full_text))
    calls = [(f'{i}/{len(batches)}#{p}', b)
             for p in range(1, max(1, passes) + 1) for i, b in enumerate(batches, 1)]
    results = await asyncio.gather(*(_one_batch(llm, b, tag) for tag, b in calls))

    seen, out = set(), []
    for terms in results:
        for r in terms:
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
