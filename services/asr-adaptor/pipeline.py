"""Помощники конвейера: ресурсы GPU, звук, пасс-2 куска, финал-раунд, лента — и стадии, собранные в
одном модуле. Сам конвейер — узлы `conveyor/` (`conveyor.run.run_conveyor`); прежний линейный
`run_pipeline` заменён ими 29.09 (равенство держал золотой тест, его выход заморожен в
`tests/asr_adaptor/golden/`).

⚠️ Узлы берут всё отсюда АТРИБУТОМ МОДУЛЯ в момент вызова (`conveyor.deps.Deps`): тесты патчат
`pipeline.<имя>`. Имя, которое узлы берут, перечислено в `conveyor.deps.NAMES` — импорт здесь,
которого не видно в коде этого файла, не лишний.
Аудио (diarize/asr/campp) — блокирующий HTTP через asyncio.to_thread; LLM — await morag LLMClient.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import audio_clients
from config import CFG
from stages import align, coverage, registry  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages import relisten as relisten_stage  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages import arbitrate as arbitrate_stage  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages import seams  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages import resplit as resplit_stage  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages import recover as recover_stage  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.chunking import MIN_S as chunking_min_s  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.chunking import chunk as chunk_fn  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.chunking import gap_chunks  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.final_round import apply_fixes, correct, doc_summary, has_entity_signal, recall_entities  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from fingerprint import one_line, stack_fingerprint  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.glossary import build_glossary, reconcile, relevant, selflabelled, witnessed  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.hints import build_hints, hinted as hinted_canonicals, merge as merge_hints  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.namer import name_speakers  # noqa: F401 — берут узлы (conveyor.deps.NAMES)
from stages.prompt_budget import WhisperTokenCounter, build_prompt  # noqa: F401 — берут узлы (conveyor.deps.NAMES)

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


def _inside(r: dict, off: float, c: dict) -> dict:
    """Ответ модели на окно с запасом → только то, что звучит внутри куска (см. stages/seams.py).

    Сегменты возвращаются уже в АБСОЛЮТНОМ времени — вызывающий кладёт их со сдвигом 0.
    """
    segs = seams.keep_inside(r.get('segments'), off, float(c['start']), float(c['end']))
    return {'text': seams.text_of(segs), 'segments': segs}


async def _decode(wav: str, sl: str, c: dict, prompt: str, audio_sec: float) -> None:
    """Кусок → текст + сегменты в АБСОЛЮТНОМ времени; пусто → один повтор с вариацией.

    Простой повтор бессмыслен: декодирование детерминировано (temperature=0) и вернёт ровно то же.
    Поэтому повтор идёт БЕЗ initial_prompt и с паддингом — снимаем разом и подавление промптом, и
    обрезку речи ровно на границе куска. Паддинг маленький (0.3 с): он может прихватить край
    соседнего слова, и это дешевле, чем потерять кусок целиком.
    """
    # ⚠️ Флаг передаётся ТОЛЬКО когда включён: вызов клиента без него — прежний, и заглушки в
    # тестах с фиксированной сигнатурой не ломаются.
    ask = {'words': True} if CFG.word_times else {}
    # Шов (stages/seams.py): кусок добора дыр режется ровно посреди непокрытого звука, то есть
    # порой по слову, — слушаем его с запасом и оставляем слова, чья середина внутри куска.
    wide = CFG.seam and c.get('recovered')
    s0, s1 = ((max(0.0, c['start'] - PAD_S), min(audio_sec, c['end'] + PAD_S)) if wide
              else (c['start'], c['end']))
    await asyncio.to_thread(_slice, wav, s0, s1, sl)
    ask1 = {'words': True} if wide else ask
    async with _res('whisper', CFG.whisper_slots):
        r = await asyncio.to_thread(lambda: audio_clients.asr(sl, prompt, **ask1))
    off = s0
    if wide:
        r = _inside(r, off, c)
        off = 0.0
    if not r['text'] and CFG.retry_empty:
        a, b = max(0.0, c['start'] - PAD_S), min(audio_sec, c['end'] + PAD_S)
        await asyncio.to_thread(_slice, wav, a, b, sl)
        ask2 = {'words': True} if CFG.seam else ask
        async with _res('whisper', CFG.whisper_slots):
            again = await asyncio.to_thread(lambda: audio_clients.asr(sl, '', **ask2))
        if CFG.seam:
            again, a = _inside(again, a, c), 0.0
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
    # Шов: запас в полторы секунды звучит речью соседа — с `ASR_SEAM` берём только своё.
    ask = {'words': True} if CFG.seam else {}
    async with _res('whisper', CFG.whisper_slots):
        r = await asyncio.to_thread(lambda: audio_clients.asr(sl, '', **ask))
    if CFG.seam:
        r = _inside(r, a, c)
    if relisten_stage.looped(r['text']):
        async with _res('whisper', CFG.whisper_slots):
            mild = await asyncio.to_thread(
                lambda: audio_clients.asr(sl, '', relisten_stage.MILD, **ask))
        if CFG.seam:
            mild = _inside(mild, a, c)
        if not relisten_stage.looped(mild['text']):
            r = mild
    Path(sl).unlink(missing_ok=True)
    return {'text': r['text'], 'segments': r['segments'], 'offset': 0.0 if CFG.seam else a}
