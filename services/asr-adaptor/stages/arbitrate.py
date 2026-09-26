"""Арбитраж: второе ухо другой модели, голосование под вето канона — ADR-0030.

Стадия после переслушивания и до финал-раунда. Каждый кусок пасса-2 слушается ещё раз ДРУГОЙ
моделью (и, если включено, той же моделью чистым ухом — без подсказки); расхождения разбираются
правилами, каждое из которых измерено отдельно на выверенном эталоне (ADR-0030):

| правило | что делает | замер |
|---|---|---|
| частота | слово второго уха обычнее слова куска в ≥ `ratio` раз, а слово куска редкое | чинит гарблы; абсолютный порог «слово ли это» врал |
| канон | второе ухо назвало ЗАВЕДОМО ВЕРНОЕ написание (канал подсказок), а кусок — нет | +2 / −0 |
| голосование | чистое ухо и вторая модель услышали одно, кусок — другое | 5 верно из 7 |
| вето | …но слово куска знает канон — большинство не указ | вернуло термин и улучшило WER |

⚠️⚠️ Слушаем ВЕСЬ кусок, не спорное слово: энкодер Whisper берёт ровно 30 с и короткий отрезок
дополняет тишиной, а узкое окно давало вдвое меньше попаданий (25 % против 50 %).
⚠️ Модель может ПРЕДЛАГАТЬ; подтверждать вправе только звук и канон. LLM здесь нет: судья языка
на LLM измерен и отвергнут (0 из 3, «исправлял» живую речь вопреки запрету).
⚠️ Порядок правил значим: гарбл и форма слова разбираются раньше «то же звучание» — иначе
«прегресс / регресс» и «поставил / поставила» считались бы одним словом и не разбирались вовсе.
⚠️ Края куска не трогаем: расхождение на границе — артефакт нарезки, а не речь.
⚠️ Замена только слово в слово (1:1, 2:2): терять или добавлять слова стадия не имеет права.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

RATIO = 100.0       # во сколько раз слово уха должно быть обычнее корпусного (колено замера)
COMMON = 5e-6       # частотность, ниже которой слово куска считаем редким
SIM = 0.55          # похожесть по звучанию, ниже которой это разговор о РАЗНЫХ словах
MAX_SPAN = 2        # длиннее — не «слово не расслышано», а развал, его не трогаем

_TR = {'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и',
       'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
       's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'}
_SPLIT = re.compile(r'^(\W*)(.*?)(\W*)$', re.U)


def key(token: str) -> str:
    """Ключ для выравнивания: без регистра, без знаков по краям, ё = е."""
    return _SPLIT.match(token).group(2).lower().replace('ё', 'е')


def sound(s: str) -> str:
    """Звуковая форма: латиница в кириллицу, двойные схлопнуты. «Postgres» ≈ «постгрес»."""
    s = ''.join(c.lower() for c in s if c.isalnum())
    s = s.replace('ph', 'ф').replace('sh', 'ш').replace('ch', 'ч').replace('th', 'т')
    s = ''.join(_TR.get(c, c) for c in s)
    s = s.replace('ё', 'е').replace('э', 'е').replace('ъ', '').replace('ь', '')
    return re.sub(r'(.)\1+', r'\1', s)


def similar(a: str, b: str) -> float:
    return SequenceMatcher(a=sound(a), b=sound(b)).ratio()


def inflection(a: str, b: str) -> bool:
    """Одно слово в разной форме («Kafka» / «Кафкой»): общая основа, разница в окончании."""
    sa, sb = sound(a), sound(b)
    pre = 0
    for x, y in zip(sa, sb):
        if x != y:
            break
        pre += 1
    return pre >= min(len(sa), len(sb)) - 3 and pre >= 4


def freq(word: str, lang: str) -> float | None:
    """Насколько слово обычно в языке; None — частотника нет (правило частоты тогда молчит)."""
    try:
        from wordfreq import word_frequency
    except ImportError:
        return None
    return word_frequency(key(word), lang)


def ordinary_swap(was: str, now: str, lang: str, ratio: float = RATIO, common: float = COMMON) -> bool:
    """Слово уха НАСТОЛЬКО обычнее слова куска, что это похоже на починку гарбла.

    ⚠️ Отношение, не абсолютный порог: частоты распределены по степенному закону, и назначенный
    порог «слово ли это» объявлял «не словом» каждое четвёртое настоящее слово речи.
    """
    fa, fb = freq(was, lang), freq(now, lang)
    if fa is None or fb is None:
        return False
    return fb > 0 and fb >= fa * ratio and fa < common


def canon_from(hints: dict | None, glossary: list | None) -> set[str]:
    """Заведомо верные написания — звуковыми формами. Источники: канал подсказок (ADR-0024:
    `terms`, `names`), подтверждённый глоссарий и `spellings` — написания ТОЛЬКО ДЛЯ СВЕРКИ.

    ⚠️ `spellings` в LLM-проход подсказок не идут: это может быть большой список (весь глоссарий
    организации, имена, названия систем — тысяча строк), и ему место в вето, а не в промпте. Откуда
    домен его взял — его дело; движку это «заведомо верно, написано людьми».
    Слово из имени в несколько слов входит и по отдельности."""
    out: set[str] = set()
    h = hints or {}
    terms = list(h.get('terms') or ()) + list(h.get('names') or ()) + list(h.get('spellings') or ())
    for g in glossary or ():
        terms += list((g or {}).get('canonicals') or ())
    for t in terms:
        if not isinstance(t, str):
            continue
        for part in t.split():
            k = sound(part)
            if len(k) >= 3:
                out.add(k)
    return out


CANON_SIM = 0.8     # сверка с каноном НЕСТРОГАЯ: канон знает именительный, а слышна другая форма


def in_canon(word: str, canon: set[str]) -> bool:
    """Знает ли канон это написание — по звучанию, с допуском на форму слова.

    ⚠️ Побуквенное равенство отвергло бы ровно то, ради чего правило заведено: канон знает имя во
    множественном числе, второе ухо услышало его в родительном падеже. Замерено: строгий вариант
    не взял ни одного места из восьми, нестрогий — два при нуле поломок.
    """
    k = sound(word)
    if not k:
        return False
    if k in canon:
        return True
    return len(k) >= 5 and any(SequenceMatcher(a=k, b=c).ratio() >= CANON_SIM for c in canon)


def _put(token: str, new_core: str) -> str:
    """Подставить новое слово, сохранив знаки по краям исходного токена."""
    lead, _, trail = _SPLIT.match(token).groups()
    return f'{lead}{_SPLIT.match(new_core).group(2)}{trail}'


def _map(a_keys: list[str], b_tokens: list[str], b_keys: list[str]) -> dict[int, str]:
    """Что второй список говорит на каждой позиции первого (1:1 участки)."""
    out: dict[int, str] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a_keys, b=b_keys, autojunk=False).get_opcodes():
        if tag in ('equal', 'replace') and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = b_tokens[j1 + k]
    return out


def arbitrate(raw: str, second: str, clean: str | None, canon: set[str], *, lang: str = 'ru',
              ratio: float = RATIO, common: float = COMMON,
              vote_words_only: bool = False) -> tuple[str, list[dict]]:
    """Разобрать расхождения куска со вторым ухом. Возвращает (новый текст, решения).

    Решение: `{i, was, now, by}`; `by` ∈ частота · канон · голосование, а отклонённое большинство
    пишется как `by: вето` с `taken: False` — журнал обязан показывать и то, что НЕ сделали.
    """
    toks, s_toks = raw.split(), (second or '').split()
    if not toks or not s_toks:
        return raw, []
    ka, kb = [key(t) for t in toks], [key(t) for t in s_toks]
    c_toks = (clean or '').split()
    clean_at = _map(ka, c_toks, [key(t) for t in c_toks]) if c_toks else {}

    out, decisions = list(toks), []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=ka, b=kb, autojunk=False).get_opcodes():
        if tag != 'replace' or (i2 - i1) != (j2 - j1) or (i2 - i1) > MAX_SPAN:
            continue
        if i1 == 0 or i2 >= len(toks):
            continue                                  # край куска — артефакт границы
        for i, j in zip(range(i1, i2), range(j1, j2)):
            was, now = toks[i], s_toks[j]
            kw, kn = ka[i], kb[j]
            if not kw or not kn or sound(kw) == sound(kn):
                continue                              # то же слово, иначе записанное — не спор
            # ⚠️ Для слова КУСКА сверка строгая, для кандидата — нестрогая. Иначе нестрогая сверка
            # признаёт «своим» и сам гарбл (он звучит похоже на канон по построению), и правило
            # не срабатывает никогда. Та же асимметрия, что в измеренном прототипе.
            in_was, in_now = sound(kw) in canon, in_canon(kn, canon)
            by = None
            if ordinary_swap(kw, kn, lang, ratio, common):
                by = 'частота'
            elif in_now and not in_was and similar(kw, kn) >= SIM and not inflection(kw, kn):
                by = 'канон'
            elif i in clean_at and sound(key(clean_at[i])) == sound(kn):
                # ⚠️ Страж «голосовать только за слова языка» ИЗМЕРЕН и по умолчанию выключен: на
                # живой записи он снял промах «обе модели одинаково ослышались на редком имени»
                # (неверно → неверно, WER не менялся), а на эталоне отнял верный голос за сленг,
                # которого частотник не знает («запушил»): 4.3 → 4.4 %. Опасный случай — вера
                # большинству ПРОТИВ верного термина — закрыт вето канона, а не этим стражем.
                if vote_words_only and not (freq(kn, lang) or 0) > 0:
                    continue                          # имя или не-слово: большинству не верим
                if in_was:
                    decisions.append({'i': i, 'was': was, 'now': now, 'by': 'вето', 'taken': False})
                    continue
                by = 'голосование'
            if by is None:
                continue
            out[i] = _put(was, now)
            decisions.append({'i': i, 'was': was, 'now': out[i], 'by': by, 'taken': True})
    return ' '.join(out), decisions


def apply(chunk: dict, second: str, clean: str | None, canon: set[str], *, lang: str = 'ru',
          ratio: float = RATIO) -> list[dict]:
    """Разобрать кусок и положить принятое в него: `raw`, тексты сегментов, слова декодера."""
    new_raw, decisions = arbitrate(chunk.get('raw') or '', second, clean, canon, lang=lang, ratio=ratio)
    taken = [d for d in decisions if d.get('taken')]
    if not taken:
        return decisions
    chunk['raw'] = new_raw
    for d in taken:
        kw = key(d['was'])
        core = _SPLIT.match(d['now']).group(2)                # ядро как есть: регистр кандидата
        rx = re.compile(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', re.I | re.U)
        for s in chunk.get('segments') or ():
            if s.get('text') and rx.search(s['text']):
                s['text'] = rx.sub(lambda _m: core, s['text'], count=1)
            for w in s.get('words') or ():
                if key(w.get('word') or '') == kw:
                    w['word'] = _put(w['word'], core)
    chunk['arbitrated'] = True
    return decisions
