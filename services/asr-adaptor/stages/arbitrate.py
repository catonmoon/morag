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
    пишется как `by: вето` с `taken: False` — журнал обязан показывать и то, что НЕ сделали; спор,
    который никто не решил, — как `by: спорно` (по нему конвейер зовёт третий голос по требованию).
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
                # Спор, который не решили ни правило, ни свидетель. В журнал — «спорно»: агент не
                # имеет права выбирать по вкусу, но и молчать о споре не должен; по этой пометке
                # конвейер решает, звать ли третий голос (чистое ухо по требованию).
                decisions.append({'i': i, 'was': was, 'now': now, 'by': 'спорно', 'taken': False})
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


# --- ворота читателя: слушать второй раз только там, где текст выглядит невменяемо ----------------

READER_SYS = ("Ты — читатель автоматической расшифровки русской речи с рабочей встречи. Оцени "
              "вменяемость слов: найди те, что выглядят ОШИБКОЙ РАСПОЗНАВАНИЯ — несуществующее слово, "
              "слово, не согласованное с соседями по роду, числу или падежу, искажённый термин или имя "
              "(сверяйся со списком известных написаний). Разговорную речь, просторечие и оговорки "
              "ошибкой не считай. Верни только подозрительные слова в том виде, как они стоят в тексте.")
READER_SCHEMA = {"type": "object", "properties": {"suspicious": {"type": "array", "items": {"type": "string"}}},
                 "required": ["suspicious"]}


async def reader_flags(llm, text: str, terms: list[str]) -> list[str]:
    """Какие слова куска читатель счёл невменяемыми. Пусто — кусок чист (или читатель промолчал).

    ⚠️⚠️ Читатель — ВОРОТА, а не судья: он решает, СЛУШАТЬ ли кусок ещё раз, а что принять — решают
    правила и свидетели. Замерено на выверенном эталоне: читатель видит 16 % настоящих ошибок при
    точности 50 %, но как ворота второго уха теряет лишь 0.1 пункта WER (4.4 против 4.3 «везде»)
    при ≈ 30 % экономии прохода. Механические триггеры (уверенность декодера, редкость слова) не
    сужают ничего: исправляемые ошибки модель делает уверенно, а редкое слово есть в каждом куске.
    ⚠️ Ответ — по схеме: свободный текст засорял пометки объяснениями вместо слов.
    """
    if llm is None or not text.strip():
        return []
    user = (f"Известные написания терминов и имён этой записи: {', '.join(terms[:80]) if terms else '—'}\n\n"
            f"Текст:\n{text}")
    try:
        got = await llm.complete_json([{'role': 'system', 'content': READER_SYS},
                                       {'role': 'user', 'content': user}], READER_SCHEMA, max_tokens=200)
    except Exception:                                        # noqa: BLE001 — ворота не роняют запись
        return []
    words = (got or {}).get('suspicious') if isinstance(got, dict) else None
    # ⚠️ Читатель иногда переписывает в «подозрительные» сам список известных написаний (а заодно
    # рвёт JSON на длинном ответе) — но это по определению ВЕРНЫЕ слова. Такие пометки — не пометки.
    known = {sound(t) for term in terms for t in term.split() if len(key(t)) >= 3}
    return [w for w in (words or ()) if isinstance(w, str) and key(w) and sound(w) not in known]


# --- страж подсказки: метрики Whisper видят, когда промпт сломал расшифровку --------------------

GUARD_LOGPROB_DROP = 0.3    # просадка avg_logprob относительно чистого уха, с которой результат — мусор
GUARD_COMPRESSION = 2.4     # порог петли (тот же, что у лесенки mlx-whisper)


def prompt_guard(clean: dict, prompted: dict) -> str:
    """Принять ли результат прослушивания С ПОДСКАЗКОЙ. Пусто — принять; иначе причина отказа.

    Измерено на 17 окнах эталона (ADR-0030, 2б): подсказка ломает расшифровку двумя РАЗНЫМИ
    способами, и ловятся они разными метриками. (1) Контекст в подсказке — модель «слепнет»:
    `avg_logprob` падает −0.25 → −0.44, корреляция просадки с ростом WER по окнам −0.77, худшие
    окна −1.2 при ΔWER +98 %. (2) Дообученная модель с подсказкой — ПЕТЛЯ: `compression_ratio`
    +20 при РАСТУЩЕМ logprob (+0.29) — модель уверенно зацикливается, и logprob этого не видит.
    ⚠️ Поэтому условия два, и заменить одно другим нельзя. ⚠️ Страж делает подсказку ДОПУСТИМОЙ, а
    не полезной: у первой модели глоссарий в подсказке и без поломок даёт ± 0.
    """
    comp = prompted.get('compression_ratio')
    if comp is not None and comp > GUARD_COMPRESSION:
        return f'петля: сжатие {comp:.1f}'
    a, b = clean.get('avg_logprob'), prompted.get('avg_logprob')
    if a is not None and b is not None and b < a - GUARD_LOGPROB_DROP:
        return f'ослепла: logprob {b:.2f} против {a:.2f}'
    return ''


# --- окно третьего голоса -----------------------------------------------------------------------

EAR_WINDOW = 30.0           # окно Whisper: короче — энкодер дополняет тишиной, длиннее — несколько окон
NEIGHBOURS_MAX = 90.0       # кусок с соседями — потолок, иначе это уже не окно, а проход


def dispute_time(chunk: dict, disputes: list[dict]) -> float:
    """Где звучит спорное место: по временам слов декодера, если они есть, иначе по доле токена."""
    toks = (chunk.get('raw') or '').split()
    idx = sorted(d['i'] for d in disputes) if disputes else [0]
    mid = idx[len(idx) // 2]
    want = key(toks[mid]) if 0 <= mid < len(toks) else ''
    for s in chunk.get('segments') or ():
        for w in s.get('words') or ():
            if want and key(w.get('word') or '') == want:
                return float(w['start'])
    a, b = float(chunk.get('start') or 0.0), float(chunk.get('end') or 0.0)
    return a + (b - a) * (mid + 0.5) / max(1, len(toks))


def ear_window(chunk: dict, disputes: list[dict], mode: str, audio_sec: float,
               neighbours: tuple[float, float] | None = None) -> tuple[float, float]:
    """Что слушать третьим голосом. ⚠️ Никогда не слово: окно по слову давало вдвое меньше попаданий.

    | режим | окно |
    |---|---|
    | `chunk` | кусок как есть (≤ 28 с; короткий кусок энкодер дополнит тишиной) |
    | `window30` | 30 с вокруг спорного места — поверх границ кусков, по временам слов декодера |
    | `neighbours` | кусок с соседями, не длиннее 90 с (несколько окон Whisper подряд) |

    Замерено на эталоне (ADR-0030, 2д): кусок как есть — лучший и самый дешёвый; окно шире
    ничего не добавляет, 15 с — уже обрывок, центрирование по спору не помогает.
    """
    a, b = float(chunk.get('start') or 0.0), float(chunk.get('end') or 0.0)
    if mode == 'window30':
        t = dispute_time(chunk, disputes)
        a, b = t - EAR_WINDOW / 2, t + EAR_WINDOW / 2
    elif mode == 'neighbours' and neighbours:
        na, nb = neighbours
        half = NEIGHBOURS_MAX / 2
        a, b = max(na, a - half), min(nb, b + half)
        if b - a > NEIGHBOURS_MAX:
            mid = (float(chunk['start']) + float(chunk['end'])) / 2
            a, b = mid - half, mid + half
    a = max(0.0, a)
    if audio_sec:
        b = min(audio_sec, b)
    if b - a < 1.0:
        a, b = max(0.0, a - 1.0), a + 2.0
    return round(a, 2), round(b, 2)

