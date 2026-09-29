"""Редактор расшифровки: агент читает СТРАНИЦУ черновика (несколько минут речи с голосами и
временем), сам решает, где сомнительно, зовёт инструменты и предлагает правки со свидетелем.

Замысел владельца (29.09): редактор — как человек, который переслушивает запись и правит её.
Системный промпт описывает роль и домен, у редактора инструменты (переслушать окно, спросить
другую модель, найти написание в каноне), список расширяется. Отличие от пошагового оркестратора
(ADR-0031), который стоил 1 745 вызовов модели на 88 минут: там код вёл модель по каждому куску
в 28 с; здесь модель читает страницу целиком и за один ответ заказывает несколько инструментов.

⚠️⚠️ Модель ПРЕДЛАГАЕТ, принимает КОД — только со свидетелем (ADR-0030, три замера подряд):
- `sound` — окно вокруг места переслушано (`listen`), и услышанное поддерживает новые слова;
- `canon` — новое написание есть среди известных написаний записи и звучит как старое.
Поверх — вето финал-раунда (`apply_fixes`): перевод, выдуманное имя, изъятие слов, смена числа,
замена известного слова записи. Вставка потерянной речи — только со свидетелем-звуком.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from difflib import SequenceMatcher

from graph.places import Slices
from graph.tools import Registry, Tool, ToolError

log = logging.getLogger('asr')

PAGE_S = 240.0             # страница — около четырёх минут речи; режется по сегменту, не посреди
CALLS_PER_MIN = 1.0        # бюджет вызовов модели: не больше одного на минуту записи (+1 на страницу)
LISTEN_S_PER_MIN = 30.0    # бюджет прослушивания: полминуты звука на минуту страницы
WINDOW_MIN, WINDOW_MAX = 20.0, 30.0   # окно слушать не короче куска: по слову — вдвое меньше попаданий
SUPPORT = 0.75             # похожесть по звучанию: новое слово «услышано», если в окне есть такое
MAX_INSERT = 20            # вставка потерянной речи — не длиннее стольких слов

EDITOR_SYS = (
    'Ты — редактор автоматической расшифровки русской речи с записи @CORPUS@. Тебе дана страница '
    'черновика: реплики с голосом и временем начала. Твоя работа — как у человека, который '
    'переслушивает запись и правит её: найти слова, распознанные неверно, и исправить их, доказав '
    'правку.\n'
    'Что обычно не так: искажённые имена, названия систем и терминов; технические названия, '
    'записанные кириллицей по звуку, хотя их пишут латиницей (JSON, Git, Kafka); латиница там, где '
    'по-русски пишут кириллицей (тимлид); выпавшие куски фраз — речь обрывается на полуслове; '
    'бессмысленное слово на месте обычного.\n'
    'Порядок работы — три ответа. 1) Прочитай страницу целиком и выбери до четырёх мест, где слово '
    'выглядит ошибкой; одним ответом закажи для всех сразу listen и lookup. Слушать «на всякий '
    'случай» нельзя: звук дорог, на страницу его хватит на несколько окон. 2) По услышанному '
    'предложи правки, тоже одним ответом. 3) finish.\n'
    'Инструменты: listen(t0, t1) — окно 20–30 с вокруг места (ear=clean — та же модель без '
    'подсказки; ear=second — модель другой школы, зови её, когда чистое ухо слышит то же, что в '
    'тексте, а слово всё равно бессмысленно); lookup(word) — известные написания этой записи.\n'
    'Правка — propose(was, now, at, witness). was — САМЫЙ КОРОТКИЙ кусок с ошибкой, одно–три слова '
    'дословно из текста страницы; now — тот же кусок исправленный, не длиннее шести слов; at — время '
    'начала строки, где он стоит. witness = sound, если в переслушанном окне слышно именно now, или '
    'canon, если написание нашлось в lookup. Написание звук не доказывает («джейсон» и JSON звучат '
    'одинаково): для латиницы и имён свидетель — canon. Выпавшую речь вставляй, только если ухо её '
    'ясно слышит: was — слово перед дырой, now — оно же и услышанные слова.\n'
    'Чего не делать: не переписывай фразу целиком и не дописывай слов, которых нет в услышанном '
    'окне; не трогай стиль и живую речь, не исправляй грамматику говорящего, не переводи, не '
    'придумывай имён. Ухо слышит то же, что в тексте, а lookup молчит — оставь как есть.'
)


# --- страницы ------------------------------------------------------------------------------------

def make_pages(turns: list[dict], page_s: float = PAGE_S) -> list[list[tuple[int, dict]]]:
    """Страницы из сегментов реплик: (номер реплики, сегмент). Режем только между сегментами."""
    pages: list[list[tuple[int, dict]]] = []
    cur: list[tuple[int, dict]] = []
    start = None
    for ti, t in enumerate(turns):
        segs = [s for s in t.get('segments') or () if (s.get('text') or '').strip()]
        if not segs and (t.get('raw') or '').strip():
            segs = [{'start': t['start'], 'end': t.get('end', t['start']), 'text': t['raw']}]
        for s in segs:
            if start is None:
                start = float(s['start'])
            if cur and float(s['start']) - start >= page_s:
                pages.append(cur)
                cur, start = [], float(s['start'])
            cur.append((ti, s))
    if cur:
        pages.append(cur)
    return pages


def _mmss(sec: float) -> str:
    return f'{int(sec // 60)}:{sec % 60:04.1f}'


def page_text(page: list[tuple[int, dict]], turns: list[dict]) -> str:
    """Строки страницы: «[голос · мм:сс.с | at=секунды] текст сегмента». Текст — ТЕКУЩИЙ текст реплики
    не показываем по сегментам (правки меняют реплику целиком); сегмент — ориентир места и времени."""
    lines, last = [], None
    for ti, s in page:
        who = turns[ti].get('speaker') or turns[ti].get('cluster') or '?'
        head = f"[{who} · {_mmss(float(s['start']))} | at={float(s['start']):.1f}]"
        if ti != last:
            lines.append('')
        lines.append(f"{head} {(s.get('text') or '').strip()}")
        last = ti
    return '\n'.join(lines).strip()


# --- свидетели -----------------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r'[\w+#.-]+', text or '', re.U)


def heard_supports(A, heard: str, was: str, now: str) -> tuple[bool, str]:
    """Поддерживает ли услышанное новые слова правки. Возвращает (да/нет, причина)."""
    hs = [A.sound(A.key(w)) for w in _words(heard) if A.key(w)]
    if not hs:
        return False, 'в окне тишина'
    was_keys = {A.sound(A.key(w)) for w in _words(was)}
    fresh = [w for w in _words(now) if A.key(w) and A.sound(A.key(w)) not in was_keys]
    for w in fresh:
        k = A.sound(A.key(w))
        if max((SequenceMatcher(a=k, b=h).ratio() for h in hs), default=0.0) < SUPPORT:
            return False, f'в окне не слышно «{w}»'
    return True, ''


def turn_at(turns: list[dict], at: float) -> int | None:
    best, dist = None, 1e9
    for i, t in enumerate(turns):
        a, b = float(t['start']), float(t.get('end') or t['start'])
        d = 0.0 if a - 0.5 <= at <= b + 0.5 else min(abs(at - a), abs(at - b))
        if d < dist:
            best, dist = i, d
    return best if dist <= 5.0 else None


# --- инструменты страницы ------------------------------------------------------------------------

class PageState:
    def __init__(self, pid: int, page, audio_sec: float) -> None:
        self.pid = pid
        self.page = page
        self.a = float(page[0][1]['start'])
        self.b = max(float(s.get('end') or s['start']) for _, s in page)
        self.minutes = max(0.5, (self.b - self.a) / 60)
        self.audio_sec = audio_sec
        self.heard: list[tuple[float, float, str, str]] = []     # (t0, t1, ухо, текст)
        self.listened = 0.0
        self.fixes: list[dict] = []


def editor_tools(ps: PageState, st, d, slices: Slices, known: list[str], canon: set,
                 *, journal: list, meter: dict) -> Registry:
    cfg = d.cfg
    A = d.arbitrate_stage
    turns = st.turns
    # Не меньше двух окон на страницу: короткая страница иначе не может переслушать ни одного места.
    listen_budget = max(2 * WINDOW_MAX, LISTEN_S_PER_MIN * ps.minutes)

    async def listen(t0: float, t1: float, ear: str = 'clean') -> dict:
        mid = (float(t0) + float(t1)) / 2
        half = min(WINDOW_MAX, max(WINDOW_MIN, float(t1) - float(t0))) / 2
        a, b = max(0.0, mid - half), min(ps.audio_sec or mid + half, mid + half)
        if ps.listened + (b - a) > listen_budget:
            raise ToolError('бюджет прослушивания страницы исчерпан', recoverable=False,
                            hint='предлагай правки по уже услышанному или заканчивай')
        if ear == 'second' and not cfg.second_model:
            raise ToolError('второго уха нет в этой установке', hint='ear=clean')
        path = await slices.get(a, b)
        async with d._res('whisper', cfg.whisper_slots):
            r = await asyncio.to_thread(lambda: d.audio_clients.asr(
                path, '', **({'model': cfg.second_model} if ear == 'second' else {})))
        text = r.get('text') or ''
        ps.listened += b - a
        ps.heard.append((a, b, ear, text))
        return {'t0': round(a, 1), 't1': round(b, 1), 'ear': ear, 'text': text}

    async def lookup(word: str) -> dict:
        scored = sorted(((A.similar(word, k), k) for k in known), reverse=True)
        hits = [k for s, k in scored if s >= 0.5][:5]
        return {'word': word, 'known_spellings': hits}

    async def propose(was: str, now: str, at: float, witness: str, why: str = '') -> dict:
        row = {'was': was, 'now': now, 'at': round(float(at), 1), 'witness': witness}
        ti = turn_at(turns, float(at))
        if ti is None:
            raise ToolError(f'реплики на {at} с нет', hint='at — время начала строки со страницы')
        t = turns[ti]
        pattern = re.compile(rf'(?<!\w){re.escape(was.strip())}(?!\w)')
        if not was.strip() or not pattern.search(t['final']):
            raise ToolError(f'«{was}» нет в реплике с {t["start"]:.1f} с дословно',
                            hint='was копируй из текста страницы символ в символ')
        verdict = ''
        if witness == 'sound':
            wins = [h for h in ps.heard if h[0] - 1.0 <= float(at) <= h[1] + 1.0]
            if not wins:
                verdict = 'нет свидетеля: окно вокруг места не переслушано'
            else:
                ok = [heard_supports(A, h[3], was, now) for h in wins]
                if not any(o for o, _ in ok):
                    verdict = 'звук не подтверждает: ' + ok[-1][1]
        elif witness == 'canon':
            key_words = [w for w in _words(now) if A.key(w) and A.sound(A.key(w)) not in
                         {A.sound(A.key(x)) for x in _words(was)}]
            if not key_words or not all(A.in_canon(w, canon) for w in key_words):
                verdict = 'канон не знает нового написания'
            elif A.similar(was, now) < A.SIM:
                verdict = 'написание из канона не звучит как исходное'
        else:
            verdict = 'неизвестный свидетель'
        if verdict:
            row.update(ok=False, why=verdict, turn=ti)
            ps.fixes.append(row)
            return {'accepted': False, 'why': verdict}
        before = t['final']
        was_w, now_w = _words(was), _words(now)
        insertion = (witness == 'sound' and len(now_w) > len(was_w)
                     and all(w in now_w for w in was_w) and len(now_w) - len(was_w) <= MAX_INSERT)
        if insertion:
            # Потерянная речь: исходные слова целы, новые подтверждены звуком — вставка мимо
            # ограничения длины финал-раунда (там «не больше шести слов» стережёт выдумку, здесь её
            # стережёт звук).
            t['final'] = pattern.sub(now.replace('\\', r'\\'), before, count=1)
            verdicts = [{'was': was, 'now': now, 'ok': True, 'why': ''}]
        else:
            verdicts: list[dict] = []
            t['final'], _, _ = d.apply_fixes(before, [{'was': was, 'now': now}], list(known),
                                             list(cfg.always_terms), log_to=verdicts,
                                             **({'protect': st.known} if st.known else {}))
        v = verdicts[-1] if verdicts else {'ok': False, 'why': 'не применилась'}
        row.update(ok=bool(v.get('ok')), why=v.get('why') or ('вставка' if insertion else ''), turn=ti)
        ps.fixes.append(row)
        return {'accepted': row['ok'], **({'why': row['why']} if not row['ok'] else {})}

    async def finish(note: str = '') -> dict:
        return {'done': True}

    tools = [
        Tool('listen', 'Переслушать окно записи 20–30 с (короче не бывает: по слову модель ошибается '
             'вдвое чаще). ear=clean — та же модель без подсказки; ear=second — модель другой школы.',
             listen, schema={'properties': {'t0': {'type': 'number'}, 't1': {'type': 'number'},
                                            'ear': {'type': 'string', 'enum': ['clean', 'second']}},
                             'required': ['t0', 't1']},
             cost=lambda a, r: {'audio_s': float(r['t1'] - r['t0'])}),
        Tool('lookup', 'Известные написания этой записи, похожие по звучанию на слово (канон, '
             'подсказки, глоссарий).', lookup,
             schema={'properties': {'word': {'type': 'string'}}, 'required': ['word']}),
        Tool('propose', 'Предложить правку: was — фраза из текста дословно, now — как должно быть, at — '
             'время начала строки, witness — sound (окно переслушано, слова слышны) или canon (написание '
             'нашлось в lookup). Код примет правку только со свидетелем.', propose,
             schema={'properties': {'was': {'type': 'string'}, 'now': {'type': 'string'},
                                    'at': {'type': 'number'},
                                    'witness': {'type': 'string', 'enum': ['sound', 'canon']},
                                    'why': {'type': 'string'}},
                     'required': ['was', 'now', 'at', 'witness']}),
        Tool('finish', 'Закончить страницу.', finish,
             schema={'properties': {'note': {'type': 'string'}}}),
    ]
    return Registry(tools, journal=journal, meter=meter, place=f'page:{ps.pid}')


# --- цикл страницы -------------------------------------------------------------------------------

def system_prompt(corpus_desc: str) -> str:
    import graph.editor as me  # noqa: PLC0415 — константу могли переопределить из файла промптов
    return me.EDITOR_SYS.replace('@CORPUS@', corpus_desc or 'рабочей встречи')


async def edit_page(ps: PageState, reg: Registry, llm, *, turns: list[dict], system: str, about: str,
                    known: list[str], prev_tail: str, meter: dict, max_tokens: int = 1200) -> None:
    budget = int(math.ceil(ps.minutes * CALLS_PER_MIN)) + 1
    user = (f"О записи: {about[:800] or '—'}\n"
            f"Известные написания записи: {', '.join(known[:60]) if known else '—'}\n"
            + (f"Конец предыдущей страницы (только для понимания):\n{prev_tail}\n\n" if prev_tail else '')
            + f"СТРАНИЦА {_mmss(ps.a)}–{_mmss(ps.b)} — здесь ищем ошибки:\n{page_text(ps.page, turns)}\n\n"
            f"Бюджет: {budget} ответов, прослушивание — до {int(max(2 * WINDOW_MAX, LISTEN_S_PER_MIN * ps.minutes))} с звука.")
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    schemas = reg.schemas()
    for _ in range(budget):
        meter['editor_calls'] = meter.get('editor_calls', 0) + 1
        try:
            resp = await llm.complete_with_tools(msgs, schemas, max_tokens=max_tokens)
        except Exception as e:                                       # noqa: BLE001 — страница как есть
            log.warning('редактор, страница %d: %s: %s', ps.pid, type(e).__name__, str(e)[:120])
            meter['editor_errors'] = meter.get('editor_errors', 0) + 1
            return
        msg = ((resp or {}).get('choices') or [{}])[0].get('message') or {}
        calls = msg.get('tool_calls') or []
        if not calls:
            return
        good, bad = [], []
        for tc in calls:
            fn = tc.get('function') or {}
            try:
                args = json.loads(fn.get('arguments') or '{}') or {}
                if not isinstance(args, dict):
                    raise ValueError('ожидался объект')
                good.append((tc, fn.get('name') or '', args))
            except ValueError as e:
                bad.append(f"{fn.get('name') or '?'}: {e}")
        # ⚠️ Битые вызовы в историю не кладём: шлюз разбирает аргументы прошлых вызовов и отвечает
        # 400 на всю переписку (живой прогон оркестратора 28.09).
        msgs.append({'role': 'assistant', 'content': msg.get('content') or '',
                     'tool_calls': [tc for tc, _, _ in good]} if good else
                    {'role': 'assistant', 'content': msg.get('content') or ''})
        done = False
        for tc, name, args in good:
            try:
                res = await reg.call(name, **args)
            except ToolError as e:
                res = e.as_result()
            if name == 'finish':
                done = True
            msgs.append({'role': 'tool', 'tool_call_id': tc.get('id') or f'call_{len(msgs)}',
                         'content': json.dumps(res, ensure_ascii=False, default=str)[:4000]})
        if bad:
            meter['editor_bad_args'] = meter.get('editor_bad_args', 0) + len(bad)
            msgs.append({'role': 'user', 'content': 'Не приняты вызовы с битыми аргументами: '
                         + '; '.join(bad) + '. Повтори по схеме.'})
        if done:
            return
    meter['editor_budget_exhausted'] = meter.get('editor_budget_exhausted', 0) + 1


# --- узел ----------------------------------------------------------------------------------------

async def run_editor(st, d, ev) -> None:
    cfg = d.cfg
    A = d.arbitrate_stage
    ev.stage('editor')
    _t = time.monotonic()
    turns = st.turns
    for t in turns:
        t['final'] = t.get('final') or t['raw']
    if not st.dsum:
        st.dsum = await d.doc_summary(st.full_text, d.llm)
    h = st.hints or {}
    known = list(dict.fromkeys(
        [x for x in list(cfg.always_terms) + list(h.get('terms') or ()) + list(h.get('names') or ())
         + list(h.get('spellings') or ()) if isinstance(x, str) and x]
        + [c for g in st.gloss for c in (g.get('canonicals') or ())]))
    canon = A.canon_from(st.hints, st.gloss) | {A.sound(p) for x in cfg.always_terms for p in x.split()}
    pages = make_pages(turns)
    system = system_prompt(cfg.corpus_desc)
    sem = asyncio.Semaphore(max(1, cfg.round_concurrency))
    states = [PageState(i, p, st.audio_sec) for i, p in enumerate(pages)]

    async def one(ps: PageState) -> None:
        prev = pages[ps.pid - 1][-2:] if ps.pid else []
        prev_tail = page_text(prev, turns) if prev else ''
        slices = Slices(d, st.tmp, st.wav, f'ed{ps.pid}')
        reg = editor_tools(ps, st, d, slices, known, canon, journal=st.journal, meter=st.meter)
        try:
            async with sem:
                await edit_page(ps, reg, d.llm, turns=turns, system=system, about=st.dsum or st.title,
                                known=known, prev_tail=prev_tail, meter=st.meter)
        finally:
            slices.cleanup()
        for f in ps.fixes:
            ev.emit('turn.fix', turn=f.get('turn'), start=f['at'], was=f['was'], now=f['now'],
                    ok=f['ok'], why=f.get('why', ''))
        st.decisions.append({'place': f'page:{ps.pid}', 'start': round(ps.a, 1), 'end': round(ps.b, 1),
                             'proposed': len(ps.fixes), 'accepted': sum(1 for f in ps.fixes if f['ok']),
                             'listened_s': round(ps.listened, 1)})

    await asyncio.gather(*(one(ps) for ps in states))
    st.raw_side = {f"{t['start']:.1f}": {'raw': t['raw'], 'final': t['final']}
                   for t in turns if t['final'] != t['raw']}
    st.round_log = [{'start': f['at'], **{k: v for k, v in f.items() if k != 'at'}}
                    for ps in states for f in ps.fixes]
    st.tm['editor_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_editor_pages'] = len(pages)
    st.tm['n_editor_accepted'] = sum(1 for f in st.round_log if f['ok'])
    ev.stage_done('editor', st.tm['editor_s'], n=len(pages), accepted=st.tm['n_editor_accepted'],
                  proposed=len(st.round_log))
