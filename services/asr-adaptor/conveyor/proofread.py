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
import json
import logging
import re
import time
from difflib import SequenceMatcher

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

    async def resolve(k: int, f: dict) -> None:
        i = f['turn']
        turn = turns[i]
        t0, t1 = time_of(turn, f['a'], f['b'])
        slices = Slices(d, st.tmp, st.wav, f'pf{k}')
        try:
            ears = await hear(slices, t0 - pad, t1 + pad)
        except Exception as error:      # noqa: BLE001 — без ушей место не решается, запись не падает
            log.warning('proofread: уши на %.1f с не вышли: %s: %s', t0, type(error).__name__, error)
            st.meter['proofread_ear_errors'] = st.meter.get('proofread_ear_errors', 0) + 1
            return
        finally:
            slices.cleanup()
        around = d._around(turns, i, 1) if hasattr(d, '_around') else ''
        text = marked(turn, f['a'], f['b'])
        user = (f'ИЗВЕСТНЫЕ НАПИСАНИЯ: {known_str[:1200]}\n\nКОНТЕКСТ:\n{around}\n\nТЕКСТ:\n'
                f"[{turn.get('speaker') or turn.get('cluster') or '?'}] {text}\n\nДОГАДКА: {f.get('guess') or '—'}\n"
                + (SUSPECT_NOTE + '\n' if f.get('suspect') else '')
                + ''.join(f'УШИ[{e}] (окно ±{pad:.0f} с): «{txt}»\n' for e, txt in ears.items()))
        async with sem:
            try:
                got = await d.llm.complete_json(
                    [{'role': 'system', 'content': RESOLVE_SYS.replace(_CORPUS_SLOT, corpus)},
                     {'role': 'user', 'content': user}],
                    schema=RESOLVE_SCHEMA, schema_name='proofread_fix', max_tokens=600)
            except Exception as error:      # noqa: BLE001
                log.warning('proofread: решатель на %.1f с сорвался: %s: %s', t0, type(error).__name__, error)
                st.meter['proofread_resolve_errors'] = st.meter.get('proofread_resolve_errors', 0) + 1
                return
        got = got or {}
        fix = got.get('fix') if isinstance(got.get('fix'), dict) else None
        row = {'turn': i, 'start': round(t0, 2), 'quote': f['quote'], 'guess': f.get('guess', ''),
               'old_error': got.get('old_error'), 'heard': got.get('heard'), 'ok': False, 'why': ''}
        if got.get('old_error') != 'yes' or not fix or not fix.get('was') or not fix.get('now'):
            row['why'] = 'proofread:not_error' if got.get('old_error') == 'no' else 'proofread:no_fix'
            fixes_log.append(row)
            return
        was = str(fix['was']).strip().strip('«»"“”')
        now = str(fix['now']).strip().strip('«»"“”')
        row.update(was=was, now=now)
        by = [e for e, txt in ears.items() if heard_supports(A, txt, was, now)[0]]
        # Одно чистое ухо — не свидетель: это та же модель, что основной проход (см. config).
        weak = by == ['clean'] and len(ears_wanted) > 1 and not getattr(cfg, 'proofread_clean_alone', False)
        if (len(by) < cfg.proofread_min_ears or weak) and not latin_known(now, known_latin):
            row['why'] = 'proofread:no_witness' if not weak else 'proofread:clean_alone'
            fixes_log.append(row)
            return
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
                         'ears': ears_wanted, 'min_ears': cfg.proofread_min_ears})
    st.tm['proofread_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_proofread_accepted'] = accepted['n']
    ev.stage_done('proofread', st.tm['proofread_s'], n=len(flags), accepted=accepted['n'])
