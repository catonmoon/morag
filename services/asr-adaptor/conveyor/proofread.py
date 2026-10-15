"""Корректор после распознавания: детектор по тексту + решатель со звуком (`ASR_PROOFREAD`).

Два шага, порознь измеренных на «Капитанском мостике» (14.10.2026, 410 трудных мест из списка
Колодезева, 60 + 72 места по слуху владельца):

1. **Детектор** читает ГОТОВУЮ расшифровку батчами по репликам и помечает места, где текст,
   скорее всего, записан не так, как прозвучало (несуществующее слово, искажённое имя или термин,
   обрывок с угадываемой фразой). Это то, что делал gemma в списке Колодезева: 71% его пометок —
   настоящие ошибки. Наш детектор на deepseek-v4.1-flash находит 76% ошибок, оставшихся после
   конвейера, ценой ~180 пометок на выпуск (половина — шум). Шум снимает не второй детектор
   (пересечение с haiku ≈ haiku, 26%), а звук на следующем шаге.
2. **Решатель** на каждую пометку: окно ±4 с переслушивают уши без подсказки (чистое и второе,
   другой школы), модель видит текст с выделенным местом, догадку детектора, термины записи и уши,
   и предлагает замену 1–3 слов. Принимает КОД: замену должно слышать хотя бы `min_ears` ушей
   (`heard_supports`, как у редактора) либо это латиница из известных написаний; затем все барьеры
   `apply_fixes` финал-раунда. По слуху владельца: из 45 решённых принятых правок верны 42 (93%);
   при двух согласных ушах — 13 из 13; при одном — 26 из 29.

Чем это отличается от отвергнутого (ADR-0020/0030/0031: судья языка 0 из 3, редактор «из 8 правок
верна 1»): там модель САМА искала, где ошибка, по странице текста, и решала без звука на месте.
Здесь ей показывают место и дают два переслушивания ровно этого окна — как судье на стенде, который
по слуху владельца верен в 87%.

Принципы прежние: модель предлагает, подтверждают звук и канон; замена слово в слово (`was` ≤ 3
слов); текст вне правок не меняется; «сломано» всегда печатается рядом с «починено».

⚠️ Матерки и другие подозрительные слова (`ASR_PROOFREAD_SUSPECT`, корни через запятую, доменная
настройка профиля) помечаются механически и решаются так же: уши слышат другое слово — правка;
уши тоже слышат его — говорящий так и сказал, текст не трогаем.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from conveyor.editor import heard_supports
from conveyor.places import Slices

log = logging.getLogger('asr')

DEFAULT_CORPUS = 'записи разговора'
_CORPUS_SLOT = '@CORPUS@'
BATCH_CHARS = 7000
LOCATE_MIN = 0.75

DETECT_SYS = (
    f'Ты корректор автоматической расшифровки {_CORPUS_SLOT}. Текст получен распознаванием речи и местами '
    'записан НЕ ТАК, как прозвучало.\n'
    'Найди места, где распознавание, скорее всего, ошиблось: несуществующее или искажённое слово; имя, '
    'название или термин с искажённым написанием; обрывок, в котором угадывается другая фраза; число или '
    'аббревиатура, не вяжущиеся с контекстом; слово, грамматически не согласованное с соседями так, что '
    'угадывается другое слово. Грубое или бранное слово в деловом разговоре подозрительно само по себе: '
    'распознавание часто «слышит» его на неразборчивом месте — помечай.\n'
    'НЕ ошибки, их не помечай: слова-паразиты и повторы, разговорные формы и сленг, оговорки и '
    'незаконченные фразы живой речи, пунктуация, термины латиницей, слова из списка известных написаний, '
    'запись одного и того же термина латиницей или кириллицей, число словами вместо цифр.\n'
    'Для каждого места: "quote" — ТОЧНАЯ подстрока текста, 1–6 слов, ровно как написано; "guess" — что, '
    'по-твоему, прозвучало (коротко); "why" — одна короткая фраза. Помечай только места с разумным '
    'сомнением. Не больше 20 мест на фрагмент. Проверяй только раздел ТЕКСТ; раздел КОНТЕКСТ дан для '
    'понимания. Ответ — строго JSON: {"flags":[{"quote":"…","guess":"…","why":"…"}]}'
)
DETECT_SCHEMA = {
    'type': 'object',
    'properties': {'flags': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'quote': {'type': 'string'}, 'guess': {'type': 'string'}, 'why': {'type': 'string'}},
        'required': ['quote', 'guess', 'why']}}},
    'required': ['flags'],
}

RESOLVE_SYS = (
    f'Ты проверяешь одно место автоматической расшифровки {_CORPUS_SLOT}. Речь устная: оговорки, повторы, '
    'слова-паразиты, сленг и термины латиницей — НЕ ошибки.\n'
    'Дано: ТЕКСТ — фрагмент расшифровки с контекстом, подозрительное место выделено кавычками «…» '
    '(кавычки не часть текста, в "was" их не включай); ДОГАДКА — что, по мнению корректора, читавшего '
    'только текст, здесь прозвучало (может ошибаться); УШИ — что на этом окне звука услышали заново модели '
    'распознавания без подсказок (уши тоже ошибаются, но они слышали звук; совпадение двух ушей — сильное '
    'свидетельство); ИЗВЕСТНЫЕ НАПИСАНИЯ терминов и имён записи.\n'
    'Реши: 1. old_error — записано ли выделенное место НЕ так, как прозвучало: "yes" / "no" / "unclear". '
    'Оговорка самого говорящего, записанная как сказана, — "no". Число словами вместо цифр — не ошибка. '
    'Если уши слышат то же, что текст, а смысл подсказывает созвучное другое — решай по смыслу и контексту. '
    '2. heard — как звучит место на самом деле (коротко). '
    '3. fix — если old_error = "yes": {"was": точная подстрока ТЕКСТА, 1–3 слова, ровно как написано; '
    '"now": на что заменить, не длиннее 6 слов, в той грамматической форме, которой требует фраза}. Замена '
    'должна быть взята из ушей или быть известным написанием; своего не сочиняй; нет верного варианта — '
    'fix: null.\n'
    'Ответ — строго JSON: {"old_error": …, "heard": "…", "fix": {"was": "…", "now": "…"} | null, '
    '"why": "одна короткая фраза"}'
)
LISTEN_NOTE = ('ЗВУК: к сообщению приложено окно звука (±{pad} с вокруг места), ушей нет — слушаешь ты сам. Сначала '
               'выпиши в "heard" ДОСЛОВНО, что звучит в окне (расшифровка, не пересказ, слово в слово), и только '
               'потом решай. Замена в fix — то, что ты реально расслышал, или известное написание расслышанного.')
SUSPECT_NOTE = ('ПРИМЕЧАНИЕ: выделено слово из списка подозрительных для этой записи (например, брань в деловом '
                'разговоре). Распознавание часто «слышит» такое на неразборчивом месте. Уши слышат другое слово — '
                'это ошибка (fix на услышанное). Уши тоже слышат его — говорящий так и сказал (old_error no, fix null).')
RESOLVE_SCHEMA = {
    'type': 'object',
    'properties': {'old_error': {'type': 'string'}, 'heard': {'type': 'string'},
                   'fix': {'type': ['object', 'null'],
                           'properties': {'was': {'type': 'string'}, 'now': {'type': 'string'}},
                           'required': ['was', 'now']},
                   'why': {'type': 'string'}},
    'required': ['old_error', 'heard', 'fix', 'why'],
}


# --- текст: батчи, место, окно ------------------------------------------------------------------

def _norm(s: str) -> list[str]:
    return re.sub(r'[^\w ]', '', (s or '').lower().replace('ё', 'е')).split()


def _mmss(sec: float) -> str:
    return f'{int(sec // 60)}:{int(sec % 60):02d}'


def batches(turns: list[dict], max_chars: int = BATCH_CHARS) -> list[list[int]]:
    """Индексы реплик батчами ~max_chars по границам реплик."""
    out, cur, size = [], [], 0
    for i, t in enumerate(turns):
        n = len(t.get('final') or '')
        if cur and size + n > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n + 24
    if cur:
        out.append(cur)
    return out


def render(turns: list[dict], idx: list[int]) -> str:
    return '\n'.join(f"[{turns[i].get('speaker') or turns[i].get('cluster') or '?'} · {_mmss(float(turns[i]['start']))}] "
                     f"{turns[i].get('final') or ''}" for i in idx)


def locate(quote: str, turns: list[dict], idx: list[int]) -> tuple[int, int, int] | None:
    """(реплика, от токена, до токена) — где в репликах батча стоит цитата: окно слов той же длины
    (±1), лучше всего совпадающее по нормализованным токенам. Ниже LOCATE_MIN — цитата сочинена."""
    q = _norm(quote)
    if not q:
        return None
    best, best_score = None, 0.0
    for i in idx:
        toks = [' '.join(_norm(w)) for w in (turns[i].get('final') or '').split()]
        for size in range(max(1, len(q) - 1), len(q) + 2):
            for a in range(0, max(1, len(toks) - size + 1)):
                score = SequenceMatcher(None, toks[a:a + size], q, autojunk=False).ratio()
                if score > best_score:
                    best_score, best = score, (i, a, min(len(toks), a + size))
    return best if best is not None and best_score >= LOCATE_MIN else None


def time_of(turn: dict, a: int, b: int) -> tuple[float, float]:
    """Когда звучат токены [a, b) реплики: по временам слов декодера, иначе по доле токена."""
    toks = (turn.get('final') or '').split()
    t0, t1 = float(turn['start']), float(turn.get('end') or turn['start'])
    if not toks:
        return t0, t1
    words = [w for sg in turn.get('segments') or () for w in sg.get('words') or ()]
    if words and len(words) >= len(toks) // 2:
        keys = [(w.get('word') or '').strip().strip('.,!?;:«»"').lower() for w in words]

        def find(tok, start):
            k = tok.strip('.,!?;:«»"').lower()
            for j in range(start, len(keys)):
                if keys[j] == k:
                    return j
            return None
        ja = find(toks[a], 0)
        jb = find(toks[b - 1], ja or 0)
        if ja is not None and jb is not None:
            return float(words[ja]['start']), float(words[jb].get('end') or words[jb]['start'])
    frac_a, frac_b = a / len(toks), b / len(toks)
    return t0 + (t1 - t0) * frac_a, t0 + (t1 - t0) * frac_b


def marked(turn: dict, a: int, b: int) -> str:
    toks = (turn.get('final') or '').split()
    if a < len(toks):
        toks[a] = '«' + toks[a]
        toks[min(b, len(toks)) - 1] = toks[min(b, len(toks)) - 1] + '»'
    return ' '.join(toks)


def suspect_flags(turns: list[dict], roots: list[str]) -> list[dict]:
    """Механические пометки: слова с подозрительными корнями (`ASR_PROOFREAD_SUSPECT`)."""
    if not roots:
        return []
    rx = re.compile(r'(?<![^\W\d_])(' + '|'.join(map(re.escape, roots)) + r')[^\W\d_]*(?![^\W\d_])', re.I)
    out = []
    for i, t in enumerate(turns):
        toks = (t.get('final') or '').split()
        for k, tok in enumerate(toks):
            if rx.search(tok):
                out.append({'turn': i, 'a': k, 'b': k + 1, 'quote': tok, 'guess': '', 'why': 'подозрительное слово',
                            'suspect': True})
    return out


def latin_known(now: str, known: set[str]) -> bool:
    lat = re.findall(r"[A-Za-z][\w'.-]*", now)
    return bool(lat) and all(w.lower() in known for w in lat)


def new_latin(was: str, now: str) -> list[str]:
    """Латинские слова замены, которых не было в заменяемом: их одно слово слушателя не подтверждает."""
    old = {w.lower() for w in re.findall(r"[A-Za-z][\w'.-]*", was)}
    return [w for w in re.findall(r"[A-Za-z][\w'.-]*", now) if w.lower() not in old]


def known_near(A, text: str, gloss: list[dict], names: list[str], always: list[str], limit: int = 12) -> list[str]:
    """Известные написания, близкие по звучанию к словам окна (`A.similar` ≥ 0.6): записи глоссария
    (сверка по услышанной форме и по канону) и имена; постоянные термины — всегда. Слушателю длинный
    список не нужен — он слышит сам, а список уводит его к чужим терминам."""
    words = {A.key(w) for w in re.findall(r"[^\W\d_][\w'-]*", text) if len(A.key(w)) >= 4}
    entries = []
    for g in gloss:
        shown = ' / '.join(dict.fromkeys(g.get('canonicals') or ()))
        if shown:
            entries += [(g.get('heard') or shown, shown)] + [(c, shown) for c in g.get('canonicals') or ()]
    entries += [(n, n) for n in names]
    score: dict[str, float] = {}
    for probe, shown in entries:
        for pw in re.findall(r"[^\W\d_][\w'-]*", probe):
            if len(A.key(pw)) < 4:
                continue
            best = max((A.similar(pw, w) for w in words), default=0.0)
            if best >= 0.6 and best > score.get(shown, 0.0):
                score[shown] = best
    top = [k for k, _ in sorted(score.items(), key=lambda kv: -kv[1])[:limit]]
    return list(dict.fromkeys(list(always) + top))


def edit_kind(was: str, now: str, gloss_words: set[str]) -> str:
    """Род правки по самой замене: «написание» — латиница, заглавное слово или слово глоссария записи (имена,
    термины); «число» — цифры; иначе «смысл» — обычные слова. Маршрут корректора по роду — ADR-0032, 15.10."""
    if re.search(r'[A-Za-z]', was + now):
        return 'написание'
    toks = re.findall(r"[\w'-]+", now)
    if any(t[:1].isupper() for t in toks) or any(t.lower() in gloss_words for t in toks):
        return 'написание'
    return 'число' if re.search(r'\d', was + now) else 'смысл'


_LISTENERS: dict[str, object] = {}


def listener_llm(cfg, model: str):
    """Клиент слушателя — один на модель на процесс."""
    if model not in _LISTENERS:
        _LISTENERS[model] = cfg.build_llm(model)
    return _LISTENERS[model]


def audio_part(path: str) -> dict:
    return {'type': 'input_audio',
            'input_audio': {'data': base64.b64encode(Path(path).read_bytes()).decode('ascii'), 'format': 'wav'}}


# --- узел ----------------------------------------------------------------------------------------

async def run_proofread(st, d, ev) -> None:
    cfg = d.cfg
    A = d.arbitrate_stage
    ev.stage('proofread')
    _t = time.monotonic()
    turns = st.turns
    for t in turns:
        t['final'] = t.get('final') or t.get('raw') or ''
    corpus = cfg.corpus_desc or DEFAULT_CORPUS
    h = st.hints or {}
    known_list = list(dict.fromkeys(
        [x for x in list(cfg.always_terms) + list(h.get('terms') or ()) + list(h.get('names') or ())
         + list(h.get('spellings') or ()) if isinstance(x, str) and x]
        + [c for g in st.gloss for c in (g.get('canonicals') or ())]))
    known_latin = {w.lower() for term in known_list for w in re.findall(r"[A-Za-z][\w'.-]*", term)}
    known_str = ', '.join(known_list)[:1500]
    roots = [r.strip() for r in (cfg.proofread_suspect or '').split(',') if r.strip()]
    ears_wanted = [e.strip() for e in (cfg.proofread_ears or 'clean').split(',') if e.strip()]
    if 'second' in ears_wanted and not cfg.second_model:
        ears_wanted = [e for e in ears_wanted if e != 'second']

    # --- детектор ---
    bs = batches(turns)
    sem = asyncio.Semaphore(max(1, cfg.round_concurrency))

    async def detect(bi: int) -> list[dict]:
        idx = bs[bi]
        ctx = render(turns, bs[bi - 1][-2:]) if bi > 0 else ''
        user = (f'Известные написания этой записи: {known_str}\n\n'
                + (f'КОНТЕКСТ (не проверять):\n{ctx}\n\n' if ctx else '')
                + f'ТЕКСТ:\n{render(turns, idx)}')
        async with sem:
            try:
                got = await d.llm.complete_json(
                    [{'role': 'system', 'content': DETECT_SYS.replace(_CORPUS_SLOT, corpus)},
                     {'role': 'user', 'content': user}],
                    schema=DETECT_SCHEMA, schema_name='proofread_flags', max_tokens=2500)
            except Exception as error:      # noqa: BLE001 — батч без пометок, не падение записи
                log.warning('proofread: детектор батча %d сорвался: %s: %s', bi, type(error).__name__, error)
                st.meter['proofread_detect_errors'] = st.meter.get('proofread_detect_errors', 0) + 1
                return []
        out = []
        for f in (got or {}).get('flags') or ():
            if not isinstance(f, dict) or not str(f.get('quote') or '').strip():
                continue
            loc = locate(str(f['quote']), turns, idx)
            if loc is None:
                st.meter['proofread_unlocated'] = st.meter.get('proofread_unlocated', 0) + 1
                continue
            out.append({'turn': loc[0], 'a': loc[1], 'b': loc[2], 'quote': str(f['quote']),
                        'guess': str(f.get('guess') or ''), 'why': str(f.get('why') or '')})
        return out

    flags = [f for lst in await asyncio.gather(*(detect(i) for i in range(len(bs)))) for f in lst]
    flags += suspect_flags(turns, roots)
    # Бюджет: не больше `proofread_max_per_min` мест на минуту звука — детектор шумит (~2/мин на
    # Мостике), а каждое место стоит два прослушивания и вызов модели.
    cap = int(max(1.0, st.audio_sec / 60.0) * cfg.proofread_max_per_min)
    if len(flags) > cap:
        log.warning('proofread: пометок %d, бюджет %d — лишние пропущены', len(flags), cap)
        st.meter['proofread_over_budget'] = len(flags) - cap
        flags = flags[:cap]
    st.tm['n_proofread_flags'] = len(flags)
    ev.emit('proofread.flags', n=len(flags), batches=len(bs))

    # --- решатель ---
    pad = float(cfg.proofread_pad_s)
    accepted = {'n': 0}
    fixes_log: list[dict] = []

    async def hear(slices: Slices, a: float, b: float) -> dict[str, str]:
        path = await slices.get(max(0.0, a), min(float(st.audio_sec or b), b) if st.audio_sec else b)
        out = {}
        for ear in ears_wanted:
            extra = {'model': cfg.second_model} if ear == 'second' else {}
            async with d._res('whisper', cfg.whisper_slots):
                r = await asyncio.to_thread(lambda: d.audio_clients.asr(path, '', **extra))
            out[ear] = (r.get('text') or '').strip()
        return out

    gloss_words = {w.lower() for g in st.gloss for c in list(g.get('canonicals') or ()) + [g.get('heard') or '']
                   for w in re.findall(r"[\w'-]+", str(c)) if len(w) >= 3}
    listener = (getattr(cfg, 'proofread_listener', '') or '').strip()
    lpad = float(getattr(cfg, 'proofread_listener_pad_s', 6.0) or 6.0)
    names = [x for x in list(h.get('names') or ()) + list(st.known or ()) if isinstance(x, str) and x] \
        if isinstance(st.known, (list, tuple, set)) else [x for x in (h.get('names') or ()) if isinstance(x, str)]

    route = (getattr(cfg, 'proofread_route', '') or '').strip()
    both = bool(listener) and route == 'kind' and bool(ears_wanted)

    async def ask(llm, system: str, user, max_tokens: int, t0: float) -> dict | None:
        async with sem:
            try:
                got = await llm.complete_json(
                    [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                    schema=RESOLVE_SCHEMA, schema_name='proofread_fix', max_tokens=max_tokens)
            except Exception as error:      # noqa: BLE001
                log.warning('proofread: решатель на %.1f с сорвался: %s: %s', t0, type(error).__name__, error)
                st.meter['proofread_resolve_errors'] = st.meter.get('proofread_resolve_errors', 0) + 1
                return None
        if isinstance(got, list):       # qwen3.8-omni отвечает [{…}] вместо {…} (замерено 15.10)
            got = next((x for x in got if isinstance(x, dict)), {})
        return got if isinstance(got, dict) else {}

    def candidate(got: dict, by_of) -> dict:
        """Ответ решателя → кандидат {was, now, by} или отказ {why}. Свидетелей считает `by_of(was, now)`."""
        fix = got.get('fix') if isinstance(got.get('fix'), dict) else None
        base = {'old_error': got.get('old_error'), 'heard': got.get('heard')}
        if got.get('old_error') != 'yes' or not fix or not fix.get('was') or not fix.get('now'):
            return {**base, 'why': 'proofread:not_error' if got.get('old_error') == 'no' else 'proofread:no_fix'}
        was = str(fix['was']).strip().strip('«»"“”')
        now = str(fix['now']).strip().strip('«»"“”')
        by, why = by_of(was, now, got)
        return {**base, 'was': was, 'now': now, 'by': by, **({'why': why} if why else {})}

    def by_listener(was: str, now: str, got: dict):
        # Слух слушателя — свидетель; новую латиницу он один не подтверждает, только канон.
        by = ['listener'] if heard_supports(A, str(got.get('heard') or ''), was, now)[0] else []
        if by and new_latin(was, now) and not latin_known(now, known_latin):
            return by, 'proofread:listener_latin'
        if not by and not latin_known(now, known_latin):
            return by, 'proofread:no_witness'
        return by, ''

    def by_ears(ears: dict):
        def check(was: str, now: str, got: dict):
            by = [e for e, txt in ears.items() if heard_supports(A, txt, was, now)[0]]
            # Одно чистое ухо — не свидетель: это та же модель, что основной проход (см. config).
            weak = by == ['clean'] and len(ears_wanted) > 1 and not getattr(cfg, 'proofread_clean_alone', False)
            if (len(by) < cfg.proofread_min_ears or weak) and not latin_known(now, known_latin):
                return by, ('proofread:no_witness' if not weak else 'proofread:clean_alone')
            return by, ''
        return check

    def choose(e: dict | None, li: dict | None) -> dict | None:
        """Правило по роду ошибки (ADR-0032, 15.10): смысл — у ушей (GigaAM слышит покадрово, 61/1 по слуху владельца),
        иначе у слушателя; написание — у слушателя (17/2 против 24/4 у ушей), иначе у ушей лишь со вторым ухом
        (порчи ушей на написании давали чистое ухо и канон)."""
        ok_e = e if e and not e.get('why') else None
        ok_l = li if li and not li.get('why') else None
        if not ok_e and not ok_l:
            return None
        kind = edit_kind((ok_e or ok_l)['was'], (ok_e or ok_l)['now'], gloss_words)
        if kind == 'смысл':
            return ok_e or ok_l
        if ok_l:
            return ok_l
        return ok_e if 'second' in (ok_e.get('by') or ()) else None

    async def resolve(k: int, f: dict) -> None:
        i = f['turn']
        turn = turns[i]
        t0, t1 = time_of(turn, f['a'], f['b'])
        slices = Slices(d, st.tmp, st.wav, f'pf{k}')
        audio, ears = None, {}
        try:
            if listener:
                path = await slices.get(max(0.0, t0 - lpad), min(float(st.audio_sec), t1 + lpad) if st.audio_sec else t1 + lpad)
                audio = audio_part(path)
            if not listener or both:
                ears = await hear(slices, t0 - pad, t1 + pad)
        except Exception as error:      # noqa: BLE001 — без ушей место не решается, запись не падает
            log.warning('proofread: уши на %.1f с не вышли: %s: %s', t0, type(error).__name__, error)
            st.meter['proofread_ear_errors'] = st.meter.get('proofread_ear_errors', 0) + 1
            return
        finally:
            slices.cleanup()
        around = d._around(turns, i, 1) if hasattr(d, '_around') else ''
        text = marked(turn, f['a'], f['b'])
        head = (f'КОНТЕКСТ:\n{around}\n\nТЕКСТ:\n'
                f"[{turn.get('speaker') or turn.get('cluster') or '?'}] {text}\n\nДОГАДКА: {f.get('guess') or '—'}\n"
                + (SUSPECT_NOTE + '\n' if f.get('suspect') else ''))
        system = RESOLVE_SYS.replace(_CORPUS_SLOT, corpus)
        jobs = {}
        if listener:
            known_here = ', '.join(known_near(A, around + ' ' + text, st.gloss, names, list(cfg.always_terms)))
            jobs['listener'] = ask(listener_llm(cfg, listener), system + '\n' + LISTEN_NOTE.format(pad=f'{lpad:.0f}'),
                                   [{'type': 'text', 'text': f'ИЗВЕСТНЫЕ НАПИСАНИЯ: {known_here}\n\n' + head}, audio], 900, t0)
        if ears:
            jobs['ears'] = ask(d.llm, system, f'ИЗВЕСТНЫЕ НАПИСАНИЯ: {known_str[:1200]}\n\n' + head
                               + ''.join(f'УШИ[{e}] (окно ±{pad:.0f} с): «{txt}»\n' for e, txt in ears.items()), 600, t0)
        answers = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        if all(v is None for v in answers.values()):
            return
        c_l = candidate(answers['listener'], by_listener) if answers.get('listener') is not None else None
        c_e = candidate(answers['ears'], by_ears(ears)) if answers.get('ears') is not None else None
        pick = choose(c_e, c_l) if both else (c_l if listener else c_e)
        row = {'turn': i, 'start': round(t0, 2), 'quote': f['quote'], 'guess': f.get('guess', ''), 'ok': False, 'why': ''}
        if both:
            row['route'] = {'ears': {k2: c_e.get(k2) for k2 in ('was', 'now', 'by', 'why')} if c_e else None,
                            'listener': {k2: c_l.get(k2) for k2 in ('was', 'now', 'by', 'why')} if c_l else None}
        src = pick or c_l or c_e or {}
        row.update(old_error=src.get('old_error'), heard=src.get('heard'))
        if not pick or pick.get('why'):
            row.update({k2: src[k2] for k2 in ('was', 'now') if k2 in src})
            row['why'] = (src.get('why') or 'proofread:no_witness') if not both or not (c_e or c_l) or src.get('why') \
                else 'proofread:route_kind'
            fixes_log.append(row)
            return
        was, now, by = pick['was'], pick['now'], pick.get('by') or []
        row.update(was=was, now=now)
        verdicts: list[dict] = []
        max_was = int(getattr(cfg, 'proofread_max_was_words', 3) or 3)
        new_text, applied, _skipped = d.apply_fixes(turn['final'], [{'was': was, 'now': now}], known_list,
                                                   cfg.always_terms, log_to=verdicts, protect=st.known,
                                                   **({'max_was_words': max_was, 'max_now_words': max_was + 2} if max_was != 3 else {}))
        if not applied:
            row['why'] = 'proofread:' + ((verdicts[0].get('why') if verdicts else '') or 'rejected')
            fixes_log.append(row)
            return
        turn['final'] = new_text
        row.update(ok=True, why='proofread:' + ('+'.join(by) if by else 'canon'))
        fixes_log.append(row)
        accepted['n'] += 1
        ev.emit('turn.fix', turn=i, start=row['start'], was=was, now=now, ok=True, why=row['why'],
                witness='+'.join(by) if by else 'canon')

    # Места одной реплики — последовательно: две правки в одном тексте не должны гоняться.
    by_turn: dict[int, list[tuple[int, dict]]] = {}
    for k, f in enumerate(flags):
        by_turn.setdefault(f['turn'], []).append((k, f))

    async def per_turn(items):
        for k, f in items:
            await resolve(k, f)

    await asyncio.gather(*(per_turn(items) for items in by_turn.values()))
    fixes_log.sort(key=lambda r: (r['start'], r['turn']))
    st.round_log.extend(fixes_log)
    for i, t in enumerate(turns):
        st.raw_side.setdefault(f"{t['start']:.1f}", {'raw': t.get('raw'), 'final': t['final']})
        st.raw_side[f"{t['start']:.1f}"]['final'] = t['final']
    st.decisions.append({'place': 'proofread', 'flags': len(flags), 'accepted': accepted['n'],
                         'ears': ([f'listener:{listener}'] if listener else ears_wanted), 'min_ears': cfg.proofread_min_ears})
    st.tm['proofread_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_proofread_accepted'] = accepted['n']
    ev.stage_done('proofread', st.tm['proofread_s'], n=len(flags), accepted=accepted['n'])
