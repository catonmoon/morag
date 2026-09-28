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
# Самооценка (ADR-0030, 2и): модель помечает, ЗНАЕТ ли она каноник как реальное название, или это
# догадка по звучанию. В подсказку пасса-2 идут только известные. Строка добавляется к промпту и поле
# к схеме ТОЛЬКО по флагу — иначе выход стадии для чужого корпуса менялся бы молча.
_SYS_SELFLABEL = (
    ' Для КАЖДОЙ пары добавь "known": true, если каноник — реальное название, которое ты знаешь '
    '(продукт, технология, компания, стандарт, человек), и false, если это догадка по звучанию или '
    'слово, которого ты не знаешь.'
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
_SCHEMA_SELFLABEL = {
    'type': 'object',
    'properties': {'terms': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'heard': {'type': 'string'},
                       'canonicals': {'type': 'array', 'items': {'type': 'string'}},
                       'known': {'type': 'boolean'}},
        'required': ['heard', 'canonicals', 'known']}}},
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


async def _one_batch(llm, batch: str, tag: str, selflabel: bool = False) -> list[dict] | None:
    """Один батч. Ретраи (битый JSON деген-петли, спайки) — в RetryingLLM (config.py), не здесь:
    политика одна на все стадии. Исчерпал попытки → батч скипается с логом, дедуп по второму
    проходу страхует.

    ⚠️ `None` — это «сорвалось», а `[]` — «терминов не нашлось». Разница несущая: по ней
    `build_glossary` отличает живой эндпоинт от мёртвого (см. там)."""
    try:
        res = await llm.complete_json(
            [{'role': 'system', 'content': _SYS + (_SYS_SELFLABEL if selflabel else '')},
             {'role': 'user', 'content': batch}],
            schema=_SCHEMA_SELFLABEL if selflabel else _SCHEMA, schema_name='glossary',
            max_tokens=_GLOSSARY_MAX_TOKENS)
        return (res or {}).get('terms') or []
    except Exception as e:
        logging.getLogger('asr').warning(
            'glossary: батч %s пропущен — %s: %s', tag, type(e).__name__, str(e)[:120])
        return None


async def build_glossary(full_text: str, llm, passes: int = 2, selflabel: bool = False) -> list[dict]:
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
    results = await asyncio.gather(*(_one_batch(llm, b, tag, selflabel) for tag, b in calls))

    # Сорвались ВСЕ вызовы — это не «терминов не нашлось», а мёртвый эндпоинт, и молчать про это
    # нельзя: без глоссария пасс-2 идёт без подсказок, а финал-раунд без каноников. Замерено на
    # корпусе митапов — глоссарий чинит 74% доменных гарблов, то есть запись выйдет заметно хуже,
    # но по виду нормальной. Роняем джобу: `run_folder.sh` не положит `.json`, запись останется в
    # очереди и перегонится на следующем прогоне сама, без человека.
    # ⚠️ Порога «сорвалось больше половины» тут НЕТ намеренно. Отказ бьёт по всем вызовам разом —
    # они уходят параллельно и повторяются в ногу, — поэтому середина «умерло 20 из 26» пока лишь
    # теоретическая. Появится в логах (построчное предупреждение выше) — тогда и поставим порог
    # по замеру, а не наугад.
    failed = sum(1 for r in results if r is None)
    if results and failed == len(results):
        raise RuntimeError(f'глоссарий: все {failed} вызовов LLM сорвались — см. предупреждения выше')
    if failed:
        logging.getLogger('asr').warning(
            'glossary: сорвалось %d вызовов из %d — терминов меньше, чем есть в записи',
            failed, len(results))

    seen, out = set(), []
    for terms in (batch for batch in results if batch):
        for r in terms:
            if not (isinstance(r, dict) and r.get('heard') and r.get('canonicals')):
                continue
            heard = str(r['heard']).strip()
            cans = r['canonicals'] if isinstance(r['canonicals'], list) else [r['canonicals']]
            cans = [str(c).strip() for c in cans if c and _keep_one(heard, str(c).strip())]
            key = heard.lower()
            if cans and key not in seen:
                seen.add(key)
                entry = {'heard': heard, 'canonicals': cans}
                if selflabel and isinstance(r.get('known'), bool):
                    entry['known'] = r['known']
                out.append(entry)
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


# --- сверка глоссария с известными написаниями (ADR-0030, 2з) -------------------------------------

RECONCILE_SIM = 0.85      # похожесть по звучанию для подмены известным написанием (нестрого)
RECONCILE_HEARD = 0.75    # …и услышанное тоже обязано походить на него: иначе «creds» → «Redis»
RECONCILE_MIN = 5         # нестрогая сверка — только от 5 звуков; короче — лишь точное совпадение


def _english_word(tok: str, threshold: float = 3.0) -> bool:
    return _HAS_WF and _zipf(tok.lower(), 'en') >= threshold


def reconcile(glossary: list[dict], known: list[str] | tuple[str, ...] = (),
              log: list[dict] | None = None) -> list[dict]:
    """Каноники глоссария против ЗНАНИЯ снаружи: одно правило, по свидетелю, без LLM.

    Замерено 28.09 на 522 местах гарблов и десяти живых прогонах: свободный глоссарий «канонизирует»
    гарбл в него самого («Postgress» рядом с «Postgres»), а подсказка пасса-2 это воспроизводит.
    Выбрасывать латиницу целиком нельзя — «GitLab», «Big Data» находит только глоссарий.

    **Подмена известным написанием.** Каноник, звучащий как известное слово (`known`: постоянные
       термины → подсказки записи → написания снаружи, по убыванию доверия), но написанный иначе, —
       заменяется им. Точное совпадение звучания — всегда; нестрогое (≥ 0.85) — только от 5 звуков,
       когда и УСЛЫШАННОЕ похоже на известное (≥ 0.75) и каноник не обычное английское слово:
       иначе «creds» становился «Redis», а «Flow» — «MLflow». Каноник, который сам известен, не трогаем.
    ⓘ Правило «транслитерация без свидетеля — вон» здесь было и снято владельцем 28.09 («тупая
    идея»): признак «звучит как услышанное» — подпорка, а не знание. Журнал (`log`) — что подменили.
    """
    from .arbitrate import key, similar, sound
    if not glossary:
        return []
    known = [k for k in dict.fromkeys(str(k) for k in known if k)]
    kidx = [(k, sound(key(k))) for k in known]
    kkeys = {key(k) for k in known}
    out: list[dict] = []
    for entry in glossary:
        heard = str(entry.get('heard') or '')
        cans: list[str] = []
        for c in entry.get('canonicals') or ():
            c = str(c)
            if key(c) in kkeys:
                cans.append(c)
                continue
            sc = sound(key(c))
            best, best_r = None, 0.0
            for k, sk in kidx:
                if sk == sc:
                    best, best_r = k, 1.0
                    break
                if len(sk) >= RECONCILE_MIN and abs(len(sk) - len(sc)) <= 3 and not _english_word(c):
                    r = similar(c, k)
                    if r >= RECONCILE_SIM and r > best_r and similar(heard, k) >= RECONCILE_HEARD:
                        best, best_r = k, r
            if best is not None:
                if log is not None:
                    log.append({'heard': heard, 'was': c, 'now': best, 'why': 'известное написание'})
                cans.append(best)
                continue
            cans.append(c)
        cans = list(dict.fromkeys(cans))
        if cans:
            out.append({**entry, 'canonicals': cans})
    return out


_LATIN_WORD = re.compile(r'[A-Za-z]+')


def witnessed(glossary: list[dict], known: list[str] | tuple[str, ...] = (),
              vocabulary: set[str] | frozenset[str] | None = None, en_threshold: float = 3.0,
              log: list[dict] | None = None) -> list[dict]:
    """A. Латинский каноник глоссария остаётся, только если его знает хоть один свидетель:
    известные написания (`known`: профиль, подсказки записи, написания снаружи), словарь домена
    (`vocabulary` — латинские слова, живущие в его текстах) или английский словарь (частотник).
    Нет свидетеля — не знание: догадка модели в подсказку не идёт. Каноники кириллицей не
    трогаем — в подсказку они и так проходят только подтверждёнными. Правил про буквы и звучание
    здесь нет: критерий один — существование свидетеля (владелец, 28.09)."""
    from .arbitrate import key
    kkeys = {key(str(k)) for k in known if k}
    vocab = {v.lower() for v in (vocabulary or ())}
    out = []
    for entry in glossary or ():
        cans = []
        for c in entry.get('canonicals') or ():
            c = str(c)
            words = [w.lower() for w in _LATIN_WORD.findall(c)]
            if not words or key(c) in kkeys or all(w in vocab or (_HAS_WF and _zipf(w, 'en') >= en_threshold)
                                                    for w in words):
                cans.append(c)
            elif log is not None:
                log.append({'heard': entry.get('heard'), 'was': c, 'now': None, 'why': 'нет свидетеля'})
        if cans:
            out.append({**entry, 'canonicals': cans})
    return out


def selflabelled(glossary: list[dict], log: list[dict] | None = None) -> list[dict]:
    """B. Отбор по самооценке модели: пары, помеченные `known: false`, уходят; без пометки — остаются
    (глоссарий без поля — прежний, ничего не теряет)."""
    out = []
    for entry in glossary or ():
        if entry.get('known') is False:
            if log is not None:
                log.append({'heard': entry.get('heard'), 'was': ', '.join(map(str, entry.get('canonicals') or ())),
                            'now': None, 'why': 'модель: догадка'})
            continue
        out.append(entry)
    return out

