"""Оркестрация end-to-end (порт adventures/podlodka-asr/pass2_full.main на morag-core, async).

audio → diarize → пасс-1(whole-file) → глоссарий[LLM] → чанк → пасс-2(per-chunk+промпт) → реплики →
финал-раунд[LLM] → Speaker_N(реестр) → выравнивание слов → обогащённый транскрипт
(.md + turns + words + coverage + raw-сайдкар + timing).
Аудио (diarize/asr/campp) — блокирующий HTTP к Маку через asyncio.to_thread; LLM — await morag LLMClient.
Один in-flight job (Mac GPU — горло) — обеспечивается в jobs.py/app.py.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import audio_clients
from config import CFG
from stages import align, coverage, registry
from stages import relisten as relisten_stage
from stages import arbitrate as arbitrate_stage
from stages.chunking import MIN_S as chunking_min_s
from stages.chunking import chunk as chunk_fn
from stages.chunking import gap_chunks
from stages.final_round import apply_fixes, correct, doc_summary, has_entity_signal, recall_entities
from fingerprint import one_line, stack_fingerprint
from stages.glossary import build_glossary, relevant
from stages.hints import build_hints, hinted as hinted_canonicals, merge as merge_hints
from stages.namer import name_speakers
from stages.prompt_budget import WhisperTokenCounter, build_prompt

log = logging.getLogger('asr')

PAD_S = 0.3  # паддинг куска при повторе пустого чанка (см. _decode)

# --- параллельные выпуски: ресурсы и порядок ---------------------------------------------------
# У каждого аудио-бэкенда один инстанс модели, поэтому ресурсы гейтятся по отдельности
# (ASR_*_SLOTS, дефолт 1): при ASR_MAX_JOBS>1 выигрывает КОНВЕЙЕРИЗАЦИЯ стадий — GPU-стадии
# выпуска B идут, пока у A работает LLM-раунд. Семафоры лениво: на импорте event loop не тот.
_RES_SEMS: dict[str, asyncio.Semaphore] = {}


def _res(name: str, slots: int) -> asyncio.Semaphore:
    if name not in _RES_SEMS:
        _RES_SEMS[name] = asyncio.Semaphore(max(1, slots))
    return _RES_SEMS[name]


# Реестр голосов требует ДЕТЕРМИНИЗМА порядка (нумерация Speaker_N по порядку появления): при
# параллельных выпусках стадия реестра идёт строго в порядке ПОСТУПЛЕНИЯ — цепочка билетов.
_TICKETS: dict[int, asyncio.Event] = {}
_NEXT_TICKET = 0


def _take_ticket() -> int:
    global _NEXT_TICKET
    t = _NEXT_TICKET
    _NEXT_TICKET += 1
    _TICKETS[t] = asyncio.Event()
    return t


async def _wait_turn(t: int) -> None:
    prev = _TICKETS.get(t - 1)
    if prev is not None:
        await prev.wait()


def _release_turn(t: int) -> None:
    """Идемпотентно: владелец изымает СВОЙ билет и сигналит. Чужие билеты не трогаем.

    Первая версия прибирала хвост чужим pop(t-2) и предполагала финиш примерно по порядку —
    медленный выпуск, переживший релиз соседа через два билета, падал в finally на KeyError,
    и исключение из finally перекрывало ГОТОВЫЙ результат (ep3: полчаса работы в мусор).
    Ожидающие держат ссылку на событие через .get() до pop — сигнал их достигает; кто пришёл
    после pop, видит None и проходит: снятый билет по определению отработан."""
    ev = _TICKETS.pop(t, None)
    if ev is not None:
        ev.set()


def _ffmpeg(args):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *args], check=True)


def _to_wav(src, dst):
    _ffmpeg(['-i', src, '-ar', '16000', '-ac', '1', dst])


def _slice(wav, a, b, dst):
    _ffmpeg(['-ss', f'{a:.2f}', '-to', f'{b:.2f}', '-i', wav, '-ar', '16000', '-ac', '1', dst])


def _neighbour_text(chunks, i: int) -> str:
    """Текст соседнего чанка: у добранного из дыры своего нет, а каноники для промпта отбирать
    надо по чему-то — разговор рядом ближе всего по теме."""
    for k in (i - 1, i + 1):
        if 0 <= k < len(chunks) and chunks[k].get('text'):
            return chunks[k]['text']
    return ''


def _around(turns, i: int, n: int = 2) -> str:
    """Разговор вокруг реплики — ЦЕЛЫМИ РЕПЛИКАМИ, по n с каждой стороны.

    Раньше правке давали пересказ ЭТОГО ЖЕ фрагмента, сделанный той же моделью: она пересказывала
    себе то, что и так видит. Настоящий контекст — соседние реплики: на них видно, что WeChat здесь
    второй игрок рядом с Alipay, а не описка. Заодно ушёл лишний вызов LLM на каждую реплику.

    Единица — реплика, а не символы и не токены. Реплика приходит из диаризации: это непрерывная
    речь одного человека, законченная мысль. Окно в символах резало бы фразы посередине и тащило
    шум, а окно в токенах вообще ничего не значит для смысла.
    """
    lo, hi = max(0, i - n), min(len(turns), i + n + 1)
    parts = [f"[{t.get('speaker') or t.get('cluster') or '?'}] {t['raw']}"
             for k, t in enumerate(turns[lo:hi], start=lo) if k != i]
    return '\n'.join(parts)


# --- лента событий: сведение сырых данных стадии к тому, что рисуется -------------------------
# ⚠️ Форма выбрана из расчёта на объём, а не на удобство чтения: у часовой встречи спанов
# диаризации под тысячу, и словари с именами полей раздули бы одно событие втрое.

SPAN_GAP = 0.2       # склейка соседних отрезков одного голоса: короче — это дыхание, не пауза
MAX_SPANS = 4000
DRAFT_WIN = 30.0     # окно черновика — то самое, которым whisper и слушает


def _emit_spans(emit, spans) -> None:
    """Лента диаризации: `[начало, конец, индекс голоса]` плюс список меток.

    Склеиваем соседние отрезки одного голоса: пауза короче 0.2 с — это вдох, а не смена
    говорящего, и на картинке шириной в экран такие щели всё равно не видны, зато событий втрое
    меньше.
    """
    if not spans:
        return
    names: list[str] = []
    out: list[list] = []
    for sp in spans:
        who = sp.get('speaker') or ''
        if who not in names:
            names.append(who)
        idx = names.index(who)
        if out and out[-1][2] == idx and sp['start'] - out[-1][1] <= SPAN_GAP:
            out[-1][1] = round(float(sp['end']), 2)
            continue
        out.append([round(float(sp['start']), 2), round(float(sp['end']), 2), idx])
        if len(out) >= MAX_SPANS:
            break
    emit('diar.spans', speakers=names, spans=out)


def _project(cents: dict) -> dict:
    """192 измерения → две, чтобы близость голосов можно было НАРИСОВАТЬ.

    ⚠️⚠️ Считаем ЗДЕСЬ, в движке, и наружу отдаём только пару чисел. Центроид — биометрический
    признак живого человека: он остаётся в реестре под 0600, а лента событий сохраняется в файл и
    может уехать куда угодно. Проекции для картинки достаточно, восстановить по ней голос нельзя.

    Главные компоненты — по этой записи, а не по корпусу: масштаб всё равно условный, а зависеть
    от состояния реестра картинке незачем.
    """
    labels = list(cents)
    if not labels:
        return {}
    import numpy as _np   # noqa: PLC0415 — нужен только здесь

    m = _np.asarray([cents[k] for k in labels], dtype=_np.float32)
    if len(labels) == 1:
        return {labels[0]: [0.0, 0.0]}
    m = m - m.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = _np.linalg.svd(m, full_matrices=False)
        xy = m @ vt[:2].T
    except Exception:  # noqa: BLE001 — картинка не повод ронять стадию
        return {k: [0.0, 0.0] for k in labels}
    span = float(_np.abs(xy).max()) or 1.0
    return {k: [round(float(v[0] / span), 3), round(float(v[1] / span), 3)] for k, v in zip(labels, xy)}


def _emit_draft(emit, segs) -> None:
    """Черновик окнами по 30 секунд — теми же, которыми его слушала модель."""
    if not segs:
        return
    win: dict[int, list[str]] = {}
    for sg in segs:
        text = (sg.get('text') or '').strip()
        if text:
            win.setdefault(int(float(sg.get('start') or 0.0) // DRAFT_WIN), []).append(text)
    for k in sorted(win):
        emit('draft.window', **{'from': round(k * DRAFT_WIN, 1), 'to': round((k + 1) * DRAFT_WIN, 1),
                                'text': ' '.join(win[k])[:600], 'bulk': True})


TURN_WINDOW = 700     # сколько знаков реплики показываем: экран окна, а не вся речь


def _window(text: str, fixes) -> tuple[str, bool]:
    """Кусок реплики, в котором видны правки: (текст, целиком ли).

    Показывать реплику ЦЕЛИКОМ нельзя — она бывает на четыре минуты речи, это стена букв и
    килобайты в каждом событии. Но и резать с начала нельзя: замена окажется за краем, и человек
    увидит текст, в котором ничего не меняется. Поэтому окно двигается к ПЕРВОЙ замене и режется
    по границам слов.
    """
    if len(text) <= TURN_WINDOW:
        return text, True
    first = min((text.find(f['was']) for f in fixes or () if f.get('was') and f['was'] in text),
                default=-1)
    start = 0 if first < 0 else max(0, first - TURN_WINDOW // 3)
    piece = text[start:start + TURN_WINDOW]
    if start:                                   # не начинаем с середины слова
        cut = piece.find(' ')
        piece = piece[cut + 1:] if 0 <= cut < 40 else piece
    tail = piece.rfind(' ')
    if tail > TURN_WINDOW - 40:
        piece = piece[:tail]
    return piece, False


def _ear_prefers(heard: str | None, was: str, now: str, margin: float = 0.15) -> str:
    """За кого звук: 'now' · 'was' · 'tie' · 'silent'. Сравнение ПО ЗВУЧАНИЮ с лучшим словом окна."""
    from difflib import SequenceMatcher
    from stages.arbitrate import key as _key, sound as _sound
    if not heard:
        return 'silent'
    hs = [_sound(_key(w)) for w in heard.split() if _key(w)]
    sw, sn = _sound(was), _sound(now)
    bw = max((SequenceMatcher(a=sw, b=h).ratio() for h in hs), default=0.0)
    bn = max((SequenceMatcher(a=sn, b=h).ratio() for h in hs), default=0.0)
    if bn - bw >= margin:
        return 'now'
    return 'was' if bw - bn >= margin else 'tie'


async def _final_round(turns, dsum: str, gloss, llm, concurrency: int, step,
                       always=(), sweep_delay: float = 30.0, emit=None,
                       protect=(), ear=None) -> tuple[int, int]:
    """Правка сущностей по репликам — параллельно, с повтором и добивочным проходом.

    Реплики независимы; последовательный проход держал стадию 8-12 мин из 15-18 на выпуск.
    Отказоустойчивость трёхслойная, потому что «реплика навсегда осталась сырой из-за лага сети» —
    недопустимый исход:
      1) два захода на месте (деген даёт битый JSON, второй заход обычно чистый);
      2) ДОБИВОЧНЫЙ проход через sweep_delay — транзиентный спайк (сеть, загрузка провайдера)
         за это время проходит, а немедленный повтор бьёт в него же;
      3) не долечилось — реплика помечается `correction_failed` в артефакте, её доправит офлайн
         `client/repair_turns.py` (правка текстовая, аудио не нужно). НЕ «до успеха»:
         систематическая ошибка (402 кредиты, 403 прокси — ловили обе) зависла бы навсегда.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(i: int, t: dict) -> tuple[bool, bool]:
        nonlocal done
        raw = t['raw']
        if len(raw.split()) < 3 or not has_entity_signal(raw, gloss):
            t['final'] = raw
            return False, False
        # Ретраи вызова (3 × 10с) — в RetryingLLM; здесь ловим только полное исчерпание.
        ok = False
        try:
            async with sem:
                recalled = await recall_entities(dsum, raw, llm)
                # ⚠️ Вердикты собираются ВСЕГДА, не только при включённой ленте: это единственное
                # место, где видно, что модель предложила и почему код принял или отверг, — а
                # аудит замен корпуса пришлось восстанавливать диффом raw → text, потому что
                # вердикты нигде не сохранялись (кольцо в памяти). Едут в артефакт: x_enriched.fixes.
                fixes: list[dict] = []
                t['final'] = await correct(raw, dsum, _around(turns, i, CFG.context_turns),
                                           relevant(raw, gloss), llm, always, recalled,
                                           corpus_desc=CFG.corpus_desc, fixes_out=fixes,
                                           **({'protect': protect} if protect else {}))
                # Арбитраж ЗВУКОМ: замену известного слова отверг сторож — но финал-раунд иногда
                # ЧИНИТ прайминг (звук за замену в 10 случаях из 104). Кусок слушается чистым
                # ухом; звук ближе к замене — принимаем, ближе к прежнему или ничья — прежнее.
                if ear is not None:
                    for f in fixes:
                        if f.get('why') != 'known_term':
                            continue
                        heard = await ear(t, f['was'])
                        said = _ear_prefers(heard, f['was'], f['now'])
                        if said == 'now':
                            t['final'], n_ok, _ = apply_fixes(t['final'], [{'was': f['was'], 'now': f['now']}],
                                                              relevant(raw, gloss), always)
                            f.update(ok=bool(n_ok), why='known_term→ear:now' if n_ok else 'known_term→ear:now,not_found')
                        else:
                            f['why'] = f'known_term (ear: {said})'
                if emit and fixes:
                    # ⚠️ Текст ПЕРЕД правками и в одном блоке с ними: события одной реплики
                    # уходят подряд, без await между ними, поэтому в ленте они не перемешаются с
                    # соседними репликами — а окно показывает правки прямо в тексте.
                    piece, whole = _window(raw, fixes)
                    emit('turn.text', turn=i, start=round(t['start'], 1), text=piece, whole=whole)
                if emit:
                    for f in fixes:
                        emit('turn.fix', turn=i, start=round(t['start'], 1), **f)
                t['fixes'] = fixes
            ok = True
        except Exception as e:
            t['final'] = raw
            log.warning('final-round failed at %.1fs (%s: %s)',
                        t['start'], type(e).__name__, str(e)[:120])
        done += 1
        if emit:
            # ⚠️ Реплики считаются ПАРАЛЛЕЛЬНО (шесть разом) плюс добивочный проход — значит
            # события приходят не в порядке записи. Клиенту нужен тайм-код, чтобы человек не
            # решил, будто запись обрабатывается задом наперёд.
            emit('turn.done', turn=i, start=round(t['start'], 1), done=done, n=len(turns),
                 changed=bool(ok and t.get('final') != raw), failed=not ok)
        if done % 20 == 0:
            step(f'final-round {done}')
        return True, not ok

    res = await asyncio.gather(*(one(i, t) for i, t in enumerate(turns)))
    n_round = sum(1 for c, _ in res if c)
    failed_idx = [i for i, (_, f) in enumerate(res) if f]

    if failed_idx:
        log.warning('final-round: %d реплик сорвались — добивочный проход через %.0fс',
                    len(failed_idx), sweep_delay)
        if sweep_delay:
            await asyncio.sleep(sweep_delay)
        res2 = await asyncio.gather(*(one(i, turns[i]) for i in failed_idx))
        failed_idx = [i for i, (_, f) in zip(failed_idx, res2) if f]

    for i in failed_idx:
        turns[i]['correction_failed'] = True
        log.warning('final-round: реплика %.1fs осталась сырой — помечена correction_failed',
                    turns[i]['start'])
    return n_round, len(failed_idx)


async def _decode(wav: str, sl: str, c: dict, prompt: str, audio_sec: float) -> None:
    """Кусок → текст + сегменты в АБСОЛЮТНОМ времени; пусто → один повтор с вариацией.

    Простой повтор бессмыслен: декодирование детерминировано (temperature=0) и вернёт ровно то же.
    Поэтому повтор идёт БЕЗ initial_prompt и с паддингом — снимаем разом и подавление промптом, и
    обрезку речи ровно на границе куска. Паддинг маленький (0.3 с): он может прихватить край
    соседнего слова, и это дешевле, чем потерять кусок целиком.
    """
    await asyncio.to_thread(_slice, wav, c['start'], c['end'], sl)
    # ⚠️ Флаг передаётся ТОЛЬКО когда включён: вызов клиента без него — прежний, и заглушки в
    # тестах с фиксированной сигнатурой не ломаются.
    ask = {'words': True} if CFG.word_times else {}
    async with _res('whisper', CFG.whisper_slots):
        r = await asyncio.to_thread(lambda: audio_clients.asr(sl, prompt, **ask))
    off = c['start']
    if not r['text'] and CFG.retry_empty:
        a, b = max(0.0, c['start'] - PAD_S), min(audio_sec, c['end'] + PAD_S)
        await asyncio.to_thread(_slice, wav, a, b, sl)
        async with _res('whisper', CFG.whisper_slots):
            again = await asyncio.to_thread(lambda: audio_clients.asr(sl, '', **ask))
        if again['text']:
            r, off, c['retried'] = again, a, True
    Path(sl).unlink(missing_ok=True)

    c['raw'] = r['text']
    # ⚠️ Метрики сегмента (`avg_logprob`, `no_speech_prob`, `compression_ratio`) НЕ выбрасываем:
    # это единственное место, где модель говорит, насколько она уверена. Без них после прогона
    # нельзя понять, где она сомневалась, — а именно там и надо переслушивать.
    c['segments'] = [{'start': round(off + float(s.get('start') or 0.0), 2),
                      'end': round(off + float(s.get('end') or 0.0), 2),
                      'text': (s.get('text') or '').strip(),
                      **{k: s[k] for k in ('avg_logprob', 'no_speech_prob', 'compression_ratio')
                         if s.get(k) is not None},
                      # Времена слов от декодера — в АБСОЛЮТНОМ времени, тем же сдвигом, что и
                      # сегмент; иначе слово из куска на 40-й минуте лежало бы «на 3-й секунде».
                      # ⚠️ Это НЕ `turns[].words` выравнивания (MMS_FA): те точные и упорядоченные,
                      # эти — приблизительные и доступны сразу. Читатели времён реплик сюда не смотрят.
                      **({'words': [{'word': str(w.get('word') or '').strip(),
                                     'start': round(off + float(w.get('start') or 0.0), 2),
                                     'end': round(off + float(w.get('end') or 0.0), 2),
                                     **({'probability': round(float(w['probability']), 3)}
                                        if w.get('probability') is not None else {})}
                                    for w in s['words']]} if s.get('words') else {})}
                     for s in r['segments']]
    if not c['raw']:
        log.warning('pass2 returned nothing for %.1f-%.1fs (speaker %s)',
                    c['start'], c['end'], c.get('speaker'))


async def _relisten_chunk(wav: str, sl: str, c: dict, audio_sec: float) -> dict:
    """Переслушать кусок ЧИСТЫМ УХОМ: то же окно с запасом, но без подсказки.

    Подсказка пасса-2 несёт канонические написания, и на невнятном звуке модель охотно «слышит»
    подсказанное — поэтому здесь её нет вовсе: нам нужно знать, что в звуке НА САМОМ ДЕЛЕ.
    Порча осталась и в чистом ухе — пробуем мягкий отступ по температуре (замерено: полный отступ
    вреден, он сочиняет).
    """
    pad = relisten_stage.PAD_S
    a, b = max(0.0, c['start'] - pad), min(audio_sec, c['end'] + pad)
    await asyncio.to_thread(_slice, wav, a, b, sl)
    async with _res('whisper', CFG.whisper_slots):
        r = await asyncio.to_thread(audio_clients.asr, sl, '')
    if relisten_stage.looped(r['text']):
        async with _res('whisper', CFG.whisper_slots):
            mild = await asyncio.to_thread(audio_clients.asr, sl, '', relisten_stage.MILD)
        if not relisten_stage.looped(mild['text']):
            r = mild
    Path(sl).unlink(missing_ok=True)
    return {'text': r['text'], 'segments': r['segments'], 'offset': a}


async def run_pipeline(audio_path: str, llm, *, episode: str = '', title: str = '',
                       url: str = '', hints: dict | None = None, progress=None) -> dict:
    """`hints` — знание об ЭТОЙ записи, известное ДО расшифровки: `{terms: [...], names: [...],
    about: str, spellings: [...]}` (`spellings` — написания только для сверки в арбитраже, в
    LLM-проход не идут; ADR-0030). Форма нарочно generic: движок принимает «заведомо верные написания», а откуда
    домен их взял (презентация, метки, каталог) — его дело. Пусто → конвейер прежний.
    """
    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix='asr_'))
    tm: dict = {}
    ticket = _take_ticket()  # порядок реестра = порядок поступления выпусков

    def step(m):
        if progress:
            progress(m)

    def emit(kind, /, **f):
        """Событие стадии в ленту (см. jobs.py). Молча ничего не делает, если ленты не просили.

        ⚠️ Обёрнуто в try/except намеренно: показ работы — украшение, и оно не имеет права
        уронить расшифровку. Час работы GPU дороже любой картинки.

        ⚠️⚠️ Первый аргумент — ПОЗИЦИОННЫЙ (`/`), и это не педантизм: у событий свои поля, и поле
        с именем `kind` (форма находки, род кадра) сталкивалось бы с именем параметра. Прогон
        падал на `got multiple values for argument 'kind'` уже после того, как вся тяжёлая работа
        сделана. Тот же класс ошибки ловился и в клиенте загрузки — закрываем его формой подписи.
        """
        if not progress:
            return
        try:
            progress({'t': kind, 'at': round(time.monotonic() - t0, 2), **f})
        except Exception:  # noqa: BLE001
            log.debug('событие %s не отправилось', kind, exc_info=True)

    def stage(name, **f):
        """Начало стадии: прежняя строка прогресса И событие — одним вызовом."""
        step(name)
        emit('stage.start', stage=name, **f)

    def stage_done(name, sec, **f):
        emit('stage.end', stage=name, sec=sec, **f)

    try:
        env = stack_fingerprint()
        log.info('стек: %s', one_line(env))
        wav = str(tmp / 'in.wav')
        await asyncio.to_thread(_to_wav, audio_path, wav)

        # Длительность нужна ДО диаризации: событие `job.meta` открывает ленту и задаёт шкалу,
        # на которой клиент рисует всё остальное. Чтение дешёвое, порядок значения не имеет.
        audio_sec = coverage.wav_duration(wav)
        emit('job.meta', audio_sec=round(audio_sec, 2), env=one_line(env))

        stage('diarize')
        _t = time.monotonic()
        async with _res('diarize', CFG.diarize_slots):
            spans = await asyncio.to_thread(audio_clients.diarize, wav)
        tm['diarize_s'] = round(time.monotonic() - _t, 1)
        stage_done('diarize', tm['diarize_s'], n=len(spans))
        _emit_spans(emit, spans)

        stage('pass1')
        _t = time.monotonic()
        async with _res('whisper', CFG.whisper_slots):
            p1 = await asyncio.to_thread(audio_clients.asr, wav, '')
        segs = p1['segments']
        words = {'segments': segs}
        full_text = ' '.join((s.get('text') or '').strip() for s in segs)
        tm['pass1_s'] = round(time.monotonic() - _t, 1)
        stage_done('pass1', tm['pass1_s'], n=len(segs))
        # ⚠️ Черновик приходит ЦЕЛИКОМ: whisper слушает файл одним вызовом. Окна по 30 с внутри
        # него есть, но наружу их отдаёт только печать модели — поэтому отдаём пачкой и говорим
        # об этом честно (`bulk`), а не изображаем поток задним числом.
        _emit_draft(emit, segs)

        p1_holes = coverage.holes([(s['start'], s['end']) for s in segs], 0.0, audio_sec,
                                  CFG.hole_min_s)
        if p1_holes:
            log.info('pass1 skipped %.1fs in %d place(s) — часть закроет пасс-2 внутри чанков',
                     sum(b - a for a, b in p1_holes), len(p1_holes))

        stage('glossary')
        _t = time.monotonic()
        # Свободный проход (гипотезы по черновику) и seed-проход (сверка известных написаний с тем
        # же черновиком) независимы — идут ПАРАЛЛЕЛЬНО, лишнего времени стадия не стоит.
        h = hints or {}
        gloss, seed = await asyncio.gather(
            build_glossary(full_text, llm),
            build_hints(full_text, llm, terms=h.get('terms') or (), names=h.get('names') or (),
                        about=h.get('about') or title))
        gloss = merge_hints(seed, gloss)
        hint_set = hinted_canonicals(seed)
        tm['glossary_s'] = round(time.monotonic() - _t, 1)
        tm['n_glossary'] = len(gloss)  # размер глоссария — чем кормим подсказку пасса-2 (бюджет ≤200 ток.)
        stage_done('glossary', tm['glossary_s'], n=len(gloss), n_hinted=len(seed),
                   terms=[g['canonicals'][0] for g in gloss[:40] if g.get('canonicals')])
        if seed:
            tm['n_hints'] = len(seed)
            log.info('hints: подтверждено %d известных написаний из %d предложенных',
                     len(seed), len(h.get('terms') or ()) + len(h.get('names') or ()))

        chunks = chunk_fn(words, spans)
        # ⚠️ Пасс-1 умеет отдавать времена ЗА КОНЦОМ записи — известная беда whisper на длинных
        # файлах. Чанк с началом за пределом звука режется в пустоту, пасс-2 отдаёт пусто, а
        # повтор с паддингом зовёт уже `-ss 4006.52 -to 4000.00` (конец клампится по длине, начало
        # нет) и роняет ВСЮ запись. Поймано на переносе: 67-минутная запись умерла на 260-м чанке
        # из 261, после девяти минут работы.
        chunks = [c for c in chunks if c['start'] < audio_sec]
        for c in chunks:
            c['end'] = min(c['end'], audio_sec)
        chunks = [c for c in chunks if c['end'] - c['start'] >= chunking_min_s]
        # Звук, которого пасс-2 НЕ услышит, — это дыры между ЧАНКАМИ, а не между сегментами
        # пасса-1: чанк переслушивается целиком, [start, end], поэтому дыра пасса-1 внутри чанка
        # уже покрыта. Добор по сегментам дублировал бы текст — замерено на ep2-10: фраза
        # «этот сам термин недоопределён…» пришла дважды, во второй раз хуже.
        # Теряется же вот что: `_split_turn` режет чанк по НАИБОЛЬШЕЙ паузе, то есть крупная дыра
        # пасса-1 сама становится границей чанка и проваливается между ними.
        unheard = coverage.holes([(c['start'], c['end']) for c in chunks], 0.0, audio_sec,
                                 CFG.hole_min_s)
        for a, b in unheard:
            log.warning('pass2 will never hear %.1fs at %.1f-%.1fs', b - a, a, b)
        if CFG.recover_gaps and unheard:
            extra = gap_chunks(unheard, spans)
            chunks = sorted(chunks + extra, key=lambda c: c['start'])
            log.warning('recovering %d chunk(s) from %.1fs of unheard audio',
                        len(extra), sum(b - a for a, b in unheard))
        counter = WhisperTokenCounter(CFG.whisper_tokenizer)

        stage('pass2')
        _t = time.monotonic()
        n_broken = n_hinted_chunks = 0
        for i, c in enumerate(chunks):
            src = c['text'] or _neighbour_text(chunks, i)
            canon = relevant(src, gloss)
            if any(c.casefold() in hint_set for c in canon):
                n_hinted_chunks += 1
            prompt = build_prompt(canon, counter, CFG.prompt_budget, CFG.always_terms, hint_set,
                                  # ⚠️ Форму и выключатель глоссария передаём ТОЛЬКО когда они
                                  # заданы: заглушки в тестах и чужие обёртки знают прежнюю сигнатуру.
                                  **({'prefix': CFG.prompt_prefix} if CFG.prompt_prefix else {}),
                                  **({'free_latin': False} if CFG.prompt_glossary == 'none' else {}),
                                  **({'conflict_free': True} if CFG.prompt_glossary == 'clean' else {}))
            # ⚠️ ЕДИНСТВЕННОЕ место, где `prompt` и `canon` существуют: дальше цикл их затирает.
            # Контекст, который уходит в whisper, показать больше неоткуда.
            emit('chunk.start', i=i + 1, n=len(chunks), **{'from': round(c['start'], 2),
                 'to': round(c['end'], 2)}, spk=c.get('speaker') or '',
                 draft=(c.get('text') or '')[:200], terms=canon[:12], prompt=prompt[:200])
            _ct = time.monotonic()
            try:
                await _decode(wav, str(tmp / f'c{i}.wav'), c, prompt, audio_sec)
                # ⚠️ Кусок отдаём ЦЕЛИКОМ (до 900 знаков — это заведомо больше, чем помещается
                # в 28 секунд речи): из этих кусков окно собирает сплошной текст записи, по
                # которому человек листает. Обрезка по 300 знаков рвала бы его на полуслове.
                emit('chunk.done', i=i + 1, sec=round(time.monotonic() - _ct, 2),
                     raw=(c.get('raw') or '')[:900], retried=bool(c.get('retried')),
                     n_seg=len(c.get('segments') or ()))
            except Exception as error:
                # ОДИН плохой чанк не должен стоить всей записи. До этой обработки любое падение
                # ffmpeg или whisper на одном куске уносило час работы вместе с диаризацией и
                # пассом-1 (ловили дважды за день). Пустой чанк честно виден в coverage как
                # неуслышанный звук — это лучше, чем потерянная запись.
                log.warning('pass2: чанк %d (%.1f-%.1f с) не расшифрован: %s: %s',
                            i, c['start'], c['end'], type(error).__name__, error)
                c['raw'], c['segments'] = '', []
                n_broken += 1
                emit('chunk.done', i=i + 1, sec=round(time.monotonic() - _ct, 2),
                     error=type(error).__name__)
            if (i + 1) % 20 == 0:
                step(f'pass2 {i + 1}/{len(chunks)}')
        tm['pass2_s'] = round(time.monotonic() - _t, 1)
        tm['n_chunks'] = len(chunks)
        tm['n_retried'] = sum(1 for c in chunks if c.get('retried'))
        stage_done('pass2', tm['pass2_s'], n=len(chunks), n_broken=n_broken, n_retried=tm['n_retried'])
        if seed:
            # Сколько кусков реально получили подтверждённый каноник — это и есть работа канала.
            tm['n_chunks_hinted'] = n_hinted_chunks
        if n_broken:
            tm['n_broken_chunks'] = n_broken
            log.warning('pass2: %d чанк(ов) из %d не расшифрованы — запись собрана без них',
                        n_broken, len(chunks))

        # --- переслушивание: место, где модель сорвалась, слушаем ещё раз чистым ухом ----------
        relisten_log: list[dict] = []
        if CFG.relisten:
            stage('relisten')
            _t = time.monotonic()
            for i, c in enumerate(chunks):
                kind = relisten_stage.suspect(c)
                if not kind:
                    continue
                was = c.get('raw') or ''
                try:
                    got = await _relisten_chunk(wav, str(tmp / f'r{i}.wav'), c, audio_sec)
                except Exception as error:      # noqa: BLE001 — стадия не имеет права ронять запись
                    log.warning('переслушивание %.1f-%.1f с не вышло: %s: %s',
                                c['start'], c['end'], type(error).__name__, error)
                    continue
                said = relisten_stage.verdict(was, got['text'])
                relisten_stage.apply(c, got['text'], got['segments'], got['offset'], said)
                relisten_log.append({'start': round(c['start'], 2), 'end': round(c['end'], 2),
                                     'kind': kind, 'verdict': said,
                                     'was': was[:300], 'now': got['text'][:300]})
                emit('relisten.span', i=len(relisten_log), **{'from': round(c['start'], 2),
                     'to': round(c['end'], 2)}, kind=kind, verdict=said,
                     was=was[:200], now=got['text'][:200])
            tm['relisten_s'] = round(time.monotonic() - _t, 1)
            tm['n_relistened'] = len(relisten_log)
            counts = {v: sum(1 for x in relisten_log if x['verdict'] == v)
                      for v in ('речь', 'тишина', 'петля осталась', 'без изменений')}
            stage_done('relisten', tm['relisten_s'], n=len(relisten_log), **counts)
            if relisten_log:
                log.info('переслушано мест: %d (%s)', len(relisten_log),
                         ', '.join(f'{k}: {v}' for k, v in counts.items() if v))

        # --- арбитраж: второе ухо другой модели, голосование под вето канона (ADR-0030) ---------
        arbitrate_log: list[dict] = []
        if CFG.second_model:
            stage('arbitrate')
            _t = time.monotonic()
            canon = arbitrate_stage.canon_from(hints, gloss)
            gate_terms = [t for t in list((hints or {}).get('terms') or ()) + list((hints or {}).get('names') or ())
                          if isinstance(t, str)] + [x for g in (gloss or ()) for x in (g.get('canonicals') or ())]
            gated = {'слушали': 0, 'пропущено': 0}

            async def _second(i: int, c: dict) -> None:
                if not c.get('raw'):
                    return
                if CFG.arbitrate_gate == 'reader':
                    # Ворота читателя: второй раз слушаем только то, что выглядит невменяемо.
                    flags = await arbitrate_stage.reader_flags(llm, c['raw'], gate_terms)
                    if not flags:
                        gated['пропущено'] += 1
                        return
                    c['reader_flags'] = flags
                    gated['слушали'] += 1
                sl = str(tmp / f'a{i}.wav')
                try:
                    await asyncio.to_thread(_slice, wav, c['start'], c['end'], sl)
                    async with _res('whisper', CFG.whisper_slots):
                        second = await asyncio.to_thread(
                            lambda: audio_clients.asr(sl, '', model=CFG.second_model))
                        clean = None
                        if CFG.clean_ear == 'always':
                            clean = await asyncio.to_thread(lambda: audio_clients.asr(sl, ''))
                        elif CFG.clean_ear == 'demand':
                            # Третий голос по требованию: только если после второго уха остались
                            # споры, не решённые ни правилом, ни свидетелем.
                            _, first = arbitrate_stage.arbitrate(c['raw'], second['text'], None, canon,
                                                                 ratio=CFG.arbitrate_ratio)
                            disputes = [d for d in first if d['by'] == 'спорно']
                            if disputes:
                                # Окном, а не словом (ADR-0030): кусок · 30 с вокруг спора · с соседями.
                                nb = (float(chunks[i - 1]['start']) if i > 0 else float(c['start']),
                                      float(chunks[i + 1]['end']) if i + 1 < len(chunks) else float(c['end']))
                                a, b = arbitrate_stage.ear_window(c, disputes, CFG.clean_ear_window, audio_sec, nb)
                                sl3 = str(tmp / f'e{i}.wav')
                                try:
                                    await asyncio.to_thread(_slice, wav, a, b, sl3)
                                    clean = await asyncio.to_thread(lambda: audio_clients.asr(sl3, ''))
                                finally:
                                    Path(sl3).unlink(missing_ok=True)
                                gated['третий голос'] = gated.get('третий голос', 0) + 1
                                gated['секунд третьего голоса'] = round(gated.get('секунд третьего голоса', 0) + (b - a), 1)
                except Exception as error:      # noqa: BLE001 — стадия не имеет права ронять запись
                    log.warning('арбитраж %.1f-%.1f с не вышел: %s: %s',
                                c['start'], c['end'], type(error).__name__, error)
                    return
                finally:
                    Path(sl).unlink(missing_ok=True)
                for d in arbitrate_stage.apply(c, second['text'], (clean or {}).get('text'), canon,
                                               ratio=CFG.arbitrate_ratio):
                    row = {'start': round(c['start'], 2), 'end': round(c['end'], 2), **d}
                    arbitrate_log.append(row)
                    emit('arbitrate.swap', **{'from': row['start'], 'to': row['end']},
                         was=d['was'], now=d['now'], by=d['by'], taken=d.get('taken', False))

            await asyncio.gather(*(_second(i, c) for i, c in enumerate(chunks)))
            arbitrate_log.sort(key=lambda r: (r['start'], r['i']))
            tm['arbitrate_s'] = round(time.monotonic() - _t, 1)
            counts = {k: sum(1 for x in arbitrate_log if x['by'] == k)
                      for k in ('частота', 'канон', 'голосование', 'вето', 'спорно')}
            if CFG.arbitrate_gate == 'reader' or CFG.clean_ear == 'demand':
                counts.update(gated)
                tm['arbitrate_gate'] = dict(gated)
            stage_done('arbitrate', tm['arbitrate_s'], n=len(arbitrate_log), **counts)
            if arbitrate_log:
                log.info('арбитраж: %d решений (%s)', len(arbitrate_log),
                         ', '.join(f'{k}: {v}' for k, v in counts.items() if v))

        # группировка подряд идущих чанков одного кластера в реплики
        turns = []
        for c in chunks:
            if turns and turns[-1]['cluster'] == c['speaker']:
                turns[-1]['chunks'].append(c)
            else:
                turns.append({'cluster': c['speaker'], 'start': c['start'], 'chunks': [c]})
        for t in turns:
            t['segments'] = [s for c in t['chunks'] for s in c['segments']]
            t['end'] = round(max([c['end'] for c in t['chunks']]
                                 + [s['end'] for s in t['segments']]), 2)

        # Сверка после склейки. Сравнивать СУММУ распознанного с длительностью реплики шумно:
        # в четырёхминутной реплике полно обычных пауз. Сигнал даёт непрерывная дыра — по ней же
        # считались потери на корпусе (find_gaps у потребителя).
        for t, (a, b) in zip(turns, coverage.turn_windows(turns, audio_sec)):
            for h0, h1 in coverage.holes(coverage.turn_segments(t), a, b, CFG.coverage_warn_s):
                log.warning('speech lost: %.1fs at %.1fs (turn %.1f-%.1f)', h1 - h0, h0, a, b)

        stage('final-round')
        _t = time.monotonic()
        dsum = await doc_summary(full_text, llm)
        for t in turns:
            t['raw'] = ' '.join(c['raw'] for c in t['chunks'] if c['raw']).strip()
        # Подтверждённые каноники идут в финал-раунд как ЗАЩИТА: замена, ломающая уже верный
        # термин, отбрасывается. Защищаем только подтверждённые, а не весь список снаружи —
        # `_term_survives` перебирает `always` на каждую замену.
        protect = list(CFG.always_terms) + [c for h in seed for c in h['canonicals']]
        # ⚠️ `protect` ниже — ПРЕЖНЯЯ переменная (постоянные термины для `_term_survives`), её не
        # трогаем: мой первый вариант затенил её пустым кортежем и при выключенной защите отключал
        # старое вето. Известные слова записи — отдельное имя.
        known = ()
        if CFG.protect_known:
            known = tuple(dict.fromkeys(
                [x for x in list((hints or {}).get('terms') or ()) + list((hints or {}).get('names') or ())
                 if isinstance(x, str)] + [x for x in (CFG.always_terms or ()) if x]))

        async def _ear(turn: dict, was: str) -> str | None:
            """Чистое ухо на 30 с вокруг спорного слова реплики — окно, не слово (ADR-0030)."""
            segs = turn.get('segments') or []
            if not segs:
                return None
            at = None
            for sg in segs:                              # точные времена от декодера, если есть
                for w in sg.get('words') or ():
                    if (w.get('word') or '').strip().strip('.,!?;:«»"') .lower() == was.strip('.,!?;:«»"').lower():
                        at = float(w['start']); break
                if at is not None:
                    break
            if at is None:                               # иначе — по доле слова в реплике
                words = (turn.get('raw') or '').split()
                pos = next((k for k, w in enumerate(words) if w.strip('.,!?;:«»"').lower() == was.strip('.,!?;:«»"').lower()), None)
                if pos is None:
                    return None
                a0, b0 = segs[0]['start'], segs[-1]['end']
                at = a0 + (b0 - a0) * pos / max(1, len(words))
            sl = str(tmp / f'ear{int(at * 100)}.wav')
            try:
                await asyncio.to_thread(_slice, wav, max(0.0, at - 15), min(audio_sec, at + 15), sl)
                async with _res('whisper', CFG.whisper_slots):
                    got = await asyncio.to_thread(lambda: audio_clients.asr(sl, ''))
                return got.get('text') or ''
            except Exception as error:      # noqa: BLE001 — ухо не роняет запись
                log.warning('арбитраж звуком не вышел (%s): %s', was, error)
                return None
            finally:
                Path(sl).unlink(missing_ok=True)

        n_round, n_failed = await _final_round(turns, dsum, gloss, llm, CFG.round_concurrency,
                                              step, protect, emit=emit,
                                             **({'protect': known} if known else {}),
                                             **({'ear': _ear} if (CFG.protect_known and CFG.final_ear) else {}))
        raw_side = {f"{t['start']:.1f}": {'raw': t['raw'], 'final': t['final']}
                    for t in turns if t['final'] != t['raw']}
        tm['round_s'] = round(time.monotonic() - _t, 1)
        tm['n_round_turns'] = n_round
        stage_done('final-round', tm['round_s'], n=len(turns), changed=n_round, failed=n_failed)
        # Плоский журнал вердиктов финал-раунда — в артефакт; из реплик убираем, чтобы не дублировать.
        round_log = [{'turn': i, 'start': round(t['start'], 2), **f}
                     for i, t in enumerate(turns) for f in (t.pop('fixes', None) or ())]
        if n_failed:
            tm['n_round_failed'] = n_failed
            log.warning('final-round: %d реплик остались сырыми из-за ошибок LLM', n_failed)

        stage('speakers')
        await _wait_turn(ticket)  # реестр — строго в порядке поступления выпусков
        async with _res('campp', CFG.campp_slots):
            cents, air = await asyncio.to_thread(audio_clients.campp, wav, spans)
        decided: list[dict] = [] if progress else None
        mapping = await asyncio.to_thread(
            registry.assign, cents, air, episode or 'adhoc', CFG.registry_path,
            CFG.match_threshold, CFG.max_centroids, decided)
        _release_turn(ticket)
        if decided:
            xy = _project(cents)
            for d in decided:
                emit('spk.vec', xy=xy.get(d['cluster'], [0.0, 0.0]), **d)
        # реплики кластеров без матча (короткий шум) → доминирующий Speaker по air-time
        air_by_lbl = defaultdict(float)
        for t in turns:
            lbl = mapping.get(t['cluster'])
            if lbl:
                air_by_lbl[lbl] += sum(c['end'] - c['start'] for c in t['chunks'])
        dominant = max(air_by_lbl, key=air_by_lbl.get) if air_by_lbl else 'Speaker_0'
        for t in turns:
            t['speaker'] = mapping.get(t['cluster'], dominant)

        # авто-наминг: Speaker_N → реальное имя (интро = истина, реестр = fallback, коррекция ложных
        # voice-матчей — см. stages/namer.py). speaker_id хранит исходный Speaker_N (трассируемость).
        name_map: dict = {}
        name_conflicts: list = []
        if CFG.enable_naming:
            stage('naming')
            _t = time.monotonic()
            name_map, name_conflicts = await name_speakers(
                turns, registry.names(CFG.registry_path), llm, corpus_desc=CFG.corpus_desc)
            tm['naming_s'] = round(time.monotonic() - _t, 1)
            stage_done('naming', tm['naming_s'], n=len(name_map))
        emit('spk.map', mapping=mapping, names=name_map, conflicts=name_conflicts)
        for t in turns:
            t['speaker_id'] = t['speaker']
            t['speaker'] = name_map.get(t['speaker'], t['speaker'])

        # emit
        n = episode.replace('ep', '') if episode else ''
        fm = f"title: {title or ('Капитанский мостик №' + n if n else 'Транскрипт')}\nurl: {url}"
        lines, plain, out_turns = ['---', fm, '---', ''], [], []
        for t in turns:
            lines.append(f"[{t['speaker']}] <!-- t:{t['start']:.1f} --> {t['final']}")
            lines.append('')
            plain.append(t['final'])
            out_turns.append({'speaker': t['speaker'], 'speaker_id': t['speaker_id'],
                              'start': round(t['start'], 1), 'end': t['end'],
                              'text': t['final'], 'raw': t['raw'], 'segments': t['segments'],
                              **({'correction_failed': True} if t.get('correction_failed') else {})})

        # Пословные тайм-коды: звук и реплики уже здесь, поэтому и время слова считается здесь.
        # Стадия не критичная (торч ставится отдельно, см. requirements-align.txt) — падает мягко.
        words_doc = None
        if CFG.enable_align:
            stage('align')
            _t = time.monotonic()
            try:
                async with _res('align', CFG.align_slots):
                    words_doc = await asyncio.to_thread(
                        align.align_turns, wav, out_turns, audio_sec,
                        episode=episode, device=CFG.align_device)
            except Exception as e:
                log.warning('word alignment skipped: %s: %s', type(e).__name__, e)
            tm['align_s'] = round(time.monotonic() - _t, 1)
            stage_done('align', tm['align_s'], n=len((words_doc or {}).get('turns') or ()))

        cov = coverage.summarize(audio_sec, segs, chunks, turns, CFG.hole_min_s)
        log.info('coverage: audio %.0fs, unheard by pass2 %.1fs, recovered %.1fs in %d chunk(s), '
                 'retried %d, still lost %.1fs',
                 cov['audio_sec'], cov['unheard_sec'], cov['recovered_sec'],
                 cov['recovered_chunks'], cov['retried_chunks'], cov['lost_sec'])

        tm['total_s'] = round(time.monotonic() - t0, 1)
        return {'markdown': '\n'.join(lines), 'text': ' '.join(plain), 'turns': out_turns,
                'raw_sidecar': raw_side, 'timing': tm, 'speaker_map': mapping,
                'speaker_names': name_map, 'name_conflicts': name_conflicts,
                'coverage': cov, 'words': words_doc,
                # Чем и на чём сделана расшифровка. В артефакте, а не в отдельной команде: через
                # год «почему на той машине вышло иначе» отвечается из самого файла (fingerprint.py).
                'env': env,
                # глоссарий и сводка — в артефакт: офлайн-долечивание реплик без пересчёта
                'glossary': gloss, 'doc_summary': dsum,
                # ⚠️ Журнал переслушивания — В АРТЕФАКТЕ, а не только в живой ленте: через месяц
                # «почему тут дыра» и «что здесь стояло раньше» отвечаются из самого файла.
                **({'relisten': relisten_log} if relisten_log else {}),
                **({'arbitration': arbitrate_log} if arbitrate_log else {}),
                **({'fixes': round_log} if round_log else {})}
    finally:
        _release_turn(ticket)  # идемпотентно: упавший выпуск не вешает очередь реестра
        shutil.rmtree(tmp, ignore_errors=True)
