"""Узлы графа: одна стадия — одна функция `async def node(st, d, ev)`.

Каждый узел читает поля `State`, пишет свои и зовёт помощники конвейера через `Deps`. Порядок и
условия — `NODES` внизу. Тексты логов, события ленты и поля таймингов те же, что у
`pipeline.run_pipeline`: золотой тест держит оба конвейера равными на одних заглушках.

⚠️ Порт сделан НАМЕРЕННО дословно, включая комментарии-«почему» из конвейера там, где они
объясняют неочевидное решение: узел без объяснения через месяц перепишут «как проще».
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path

from graph.deps import Deps
from graph.events import Emitter
from graph.state import State

log = logging.getLogger('asr')


def _known_spellings(cfg, h: dict) -> list[str]:
    """Известные написания записи: постоянные термины корпуса + канал подсказок."""
    return list(cfg.always_terms) + [str(t) for k_ in ('terms', 'names', 'spellings')
                                     for t in (h.get(k_) or ()) if t]


# --- prepare ------------------------------------------------------------------------------------

async def prepare(st: State, d: Deps, ev: Emitter) -> None:
    st.env = d.stack_fingerprint()
    log.info('стек: %s', d.one_line(st.env))
    st.wav = str(Path(st.tmp) / 'in.wav')
    await asyncio.to_thread(d._to_wav, st.audio_path, st.wav)
    # Длительность нужна ДО диаризации: событие `job.meta` открывает ленту и задаёт шкалу,
    # на которой клиент рисует всё остальное. Чтение дешёвое, порядок значения не имеет.
    st.audio_sec = d.coverage.wav_duration(st.wav)
    ev.emit('job.meta', audio_sec=round(st.audio_sec, 2), env=d.one_line(st.env))


# --- diarize / pass1 ----------------------------------------------------------------------------

async def diarize(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    ev.stage('diarize')
    _t = time.monotonic()
    async with d._res('diarize', cfg.diarize_slots):
        st.spans = await asyncio.to_thread(d.audio_clients.diarize, st.wav)
    st.tm['diarize_s'] = round(time.monotonic() - _t, 1)
    ev.stage_done('diarize', st.tm['diarize_s'], n=len(st.spans))
    d._emit_spans(ev.emit, st.spans)


async def pass1(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    ev.stage('pass1')
    _t = time.monotonic()
    async with d._res('whisper', cfg.whisper_slots):
        p1 = await asyncio.to_thread(d.audio_clients.asr, st.wav, '')
    st.segs = p1['segments']
    st.full_text = ' '.join((s.get('text') or '').strip() for s in st.segs)
    st.tm['pass1_s'] = round(time.monotonic() - _t, 1)
    ev.stage_done('pass1', st.tm['pass1_s'], n=len(st.segs))
    # ⚠️ Черновик приходит ЦЕЛИКОМ: whisper слушает файл одним вызовом. Окна по 30 с внутри
    # него есть, но наружу их отдаёт только печать модели — поэтому отдаём пачкой (`bulk`).
    d._emit_draft(ev.emit, st.segs)

    st.p1_holes = d.coverage.holes([(s['start'], s['end']) for s in st.segs], 0.0, st.audio_sec,
                                   cfg.hole_min_s)
    if st.p1_holes:
        log.info('pass1 skipped %.1fs in %d place(s) — часть закроет пасс-2 внутри чанков',
                 sum(b - a for a, b in st.p1_holes), len(st.p1_holes))


# --- glossary -----------------------------------------------------------------------------------

async def glossary(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    ev.stage('glossary')
    _t = time.monotonic()
    # Свободный проход (гипотезы по черновику) и seed-проход (сверка известных написаний с тем
    # же черновиком) независимы — идут ПАРАЛЛЕЛЬНО, лишнего времени стадия не стоит.
    h = st.hints or {}
    gloss, seed = await asyncio.gather(
        d.build_glossary(st.full_text, d.llm, **({'selflabel': True} if cfg.glossary_selflabel else {})),
        d.build_hints(st.full_text, d.llm, terms=h.get('terms') or (), names=h.get('names') or (),
                      about=h.get('about') or st.title))
    gloss = d.merge_hints(seed, gloss)
    if cfg.glossary_reconcile:
        rec_log: list[dict] = []
        gloss = d.reconcile(gloss, known=_known_spellings(cfg, h), log=rec_log)
        st.tm['n_reconciled'] = len(rec_log)
        if rec_log:
            log.info('глоссарий: сверка с известными написаниями — подменено %d', len(rec_log))
    if cfg.glossary_selflabel or cfg.glossary_witness:
        drop_log: list[dict] = []
        if cfg.glossary_selflabel:
            gloss = d.selflabelled(gloss, log=drop_log)
        if cfg.glossary_witness:
            gloss = d.witnessed(gloss, known=_known_spellings(cfg, h),
                                vocabulary=set(str(v) for v in (h.get('vocabulary') or ())),
                                en_threshold=cfg.glossary_witness_en, log=drop_log)
        st.tm['n_glossary_dropped'] = len(drop_log)
        if drop_log:
            log.info('глоссарий: без свидетеля / по самооценке выброшено %d каноников', len(drop_log))
    st.gloss, st.seed = gloss, seed
    st.hint_set = d.hinted_canonicals(seed)
    st.tm['glossary_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_glossary'] = len(gloss)  # размер глоссария — чем кормим подсказку пасса-2
    ev.stage_done('glossary', st.tm['glossary_s'], n=len(gloss), n_hinted=len(seed),
                  terms=[g['canonicals'][0] for g in gloss[:40] if g.get('canonicals')])
    if seed:
        st.tm['n_hints'] = len(seed)
        log.info('hints: подтверждено %d известных написаний из %d предложенных',
                 len(seed), len(h.get('terms') or ()) + len(h.get('names') or ()))


# --- chunking -----------------------------------------------------------------------------------

async def chunking(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    chunks = d.chunk_fn({'segments': st.segs}, st.spans)
    # ⚠️ Пасс-1 умеет отдавать времена ЗА КОНЦОМ записи — известная беда whisper на длинных
    # файлах. Чанк с началом за пределом звука режется в пустоту, повтор с паддингом зовёт
    # `-ss 4006.52 -to 4000.00` и роняет ВСЮ запись (67-минутная запись умерла на 260-м чанке).
    chunks = [c for c in chunks if c['start'] < st.audio_sec]
    for c in chunks:
        c['end'] = min(c['end'], st.audio_sec)
    chunks = [c for c in chunks if c['end'] - c['start'] >= d.chunking_min_s]
    # Звук, которого пасс-2 НЕ услышит, — это дыры между ЧАНКАМИ, а не между сегментами пасса-1:
    # чанк переслушивается целиком, и дыра пасса-1 внутри чанка уже покрыта. Добор по сегментам
    # дублировал бы текст (замерено на ep2-10).
    unheard = d.coverage.holes([(c['start'], c['end']) for c in chunks], 0.0, st.audio_sec,
                               cfg.hole_min_s)
    for a, b in unheard:
        log.warning('pass2 will never hear %.1fs at %.1f-%.1fs', b - a, a, b)
    if cfg.recover_gaps and unheard:
        extra = d.gap_chunks(unheard, st.spans)
        chunks = sorted(chunks + extra, key=lambda c: c['start'])
        log.warning('recovering %d chunk(s) from %.1fs of unheard audio',
                    len(extra), sum(b - a for a, b in unheard))
    st.chunks = chunks
    st.counter = d.WhisperTokenCounter(cfg.whisper_tokenizer)


# --- pass2 --------------------------------------------------------------------------------------

async def pass2(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    chunks = st.chunks
    ev.stage('pass2')
    _t = time.monotonic()
    n_broken = n_hinted_chunks = 0
    # Режим «подменять»: карта «звучание → известное написание», постоянные термины старше
    # подсказок, подсказки старше spellings (снимок написаний снаружи — только для подмены).
    spell_map: dict[str, str] = {}
    if cfg.prompt_glossary == 'substitute':
        for k in _known_spellings(cfg, st.hints or {}):
            spell_map.setdefault(d.arbitrate_stage.sound(d.arbitrate_stage.key(k)), k)
    for i, c in enumerate(chunks):
        src = c['text'] or d._neighbour_text(chunks, i)
        canon = d.relevant(src, st.gloss)
        if any(x.casefold() in st.hint_set for x in canon):
            n_hinted_chunks += 1
        prompt = d.build_prompt(canon, st.counter, cfg.prompt_budget, cfg.always_terms, st.hint_set,
                                # ⚠️ Форму и выключатель глоссария передаём ТОЛЬКО когда они
                                # заданы: заглушки в тестах и чужие обёртки знают прежнюю сигнатуру.
                                **({'prefix': cfg.prompt_prefix} if cfg.prompt_prefix else {}),
                                **({'free_latin': False} if cfg.prompt_glossary == 'none' else {}),
                                **({'conflict_free': True} if cfg.prompt_glossary == 'clean' else {}),
                                **({'substitute': spell_map} if cfg.prompt_glossary == 'substitute' else {}))
        ev.emit('chunk.start', i=i + 1, n=len(chunks), **{'from': round(c['start'], 2),
                'to': round(c['end'], 2)}, spk=c.get('speaker') or '',
                draft=(c.get('text') or '')[:200], terms=canon[:12], prompt=prompt[:200])
        _ct = time.monotonic()
        try:
            await d._decode(st.wav, str(Path(st.tmp) / f'c{i}.wav'), c, prompt, st.audio_sec)
            ev.emit('chunk.done', i=i + 1, sec=round(time.monotonic() - _ct, 2),
                    raw=(c.get('raw') or '')[:900], retried=bool(c.get('retried')),
                    n_seg=len(c.get('segments') or ()))
        except Exception as error:      # noqa: BLE001 — один плохой чанк не стоит всей записи
            log.warning('pass2: чанк %d (%.1f-%.1f с) не расшифрован: %s: %s',
                        i, c['start'], c['end'], type(error).__name__, error)
            c['raw'], c['segments'] = '', []
            n_broken += 1
            ev.emit('chunk.done', i=i + 1, sec=round(time.monotonic() - _ct, 2),
                    error=type(error).__name__)
        if (i + 1) % 20 == 0:
            ev.step(f'pass2 {i + 1}/{len(chunks)}')
    st.tm['pass2_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_chunks'] = len(chunks)
    st.tm['n_retried'] = sum(1 for c in chunks if c.get('retried'))
    ev.stage_done('pass2', st.tm['pass2_s'], n=len(chunks), n_broken=n_broken, n_retried=st.tm['n_retried'])
    if st.seed:
        st.tm['n_chunks_hinted'] = n_hinted_chunks
    if n_broken:
        st.tm['n_broken_chunks'] = n_broken
        log.warning('pass2: %d чанк(ов) из %d не расшифрованы — запись собрана без них',
                    n_broken, len(chunks))


# --- relisten -----------------------------------------------------------------------------------

async def relisten(st: State, d: Deps, ev: Emitter) -> None:
    rs = d.relisten_stage
    ev.stage('relisten')
    _t = time.monotonic()
    relisten_log: list[dict] = []
    for i, c in enumerate(st.chunks):
        kind = rs.suspect(c)
        if not kind:
            continue
        was = c.get('raw') or ''
        try:
            got = await d._relisten_chunk(st.wav, str(Path(st.tmp) / f'r{i}.wav'), c, st.audio_sec)
        except Exception as error:      # noqa: BLE001 — стадия не имеет права ронять запись
            log.warning('переслушивание %.1f-%.1f с не вышло: %s: %s',
                        c['start'], c['end'], type(error).__name__, error)
            continue
        said = rs.verdict(was, got['text'])
        rs.apply(c, got['text'], got['segments'], got['offset'], said)
        relisten_log.append({'start': round(c['start'], 2), 'end': round(c['end'], 2),
                             'kind': kind, 'verdict': said,
                             'was': was[:300], 'now': got['text'][:300]})
        ev.emit('relisten.span', i=len(relisten_log), **{'from': round(c['start'], 2),
                'to': round(c['end'], 2)}, kind=kind, verdict=said,
                was=was[:200], now=got['text'][:200])
    st.relisten_log = relisten_log
    st.tm['relisten_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_relistened'] = len(relisten_log)
    counts = {v: sum(1 for x in relisten_log if x['verdict'] == v)
              for v in ('речь', 'тишина', 'петля осталась', 'без изменений')}
    ev.stage_done('relisten', st.tm['relisten_s'], n=len(relisten_log), **counts)
    if relisten_log:
        log.info('переслушано мест: %d (%s)', len(relisten_log),
                 ', '.join(f'{k}: {v}' for k, v in counts.items() if v))


# --- arbitrate ----------------------------------------------------------------------------------
# Место = кусок пасса-2. Инструменты (`graph/places.py`) + цикл по месту (`graph/loop.py`) +
# политика (`graph/policy.py`): правила повторяют `_second` линейного конвейера шаг в шаг, LLM-
# оркестратор выбирает те же инструменты сам. Учёт ворот и третьего голоса — тот же, что в
# конвейере (`timing.arbitrate_gate`), по истории места.

async def arbitrate(st: State, d: Deps, ev: Emitter) -> None:
    from graph.loop import Place, decide_place
    from graph.places import Slices, arbitrate_tools
    from graph.policy import make_policy

    cfg = d.cfg
    A = d.arbitrate_stage
    chunks, hints, gloss = st.chunks, st.hints, st.gloss
    ev.stage('arbitrate')
    _t = time.monotonic()
    canon = A.canon_from(hints, gloss)
    gate_terms = [t for t in list((hints or {}).get('terms') or ()) + list((hints or {}).get('names') or ())
                  if isinstance(t, str)] + [x for g in (gloss or ()) for x in (g.get('canonicals') or ())]
    gated = {'слушали': 0, 'пропущено': 0}
    failed = {'n': 0}      # куски, где стадия не вышла (бэкенд упал): в артефакт, не только в лог
    arbitrate_log: list[dict] = []
    policy = make_policy(cfg, d.llm)

    def _account(place: Place) -> None:
        """Ворота и третий голос — по истории места, теми же счётчиками, что у конвейера."""
        gate = place.last('reader_flags')
        if gate is not None and place.count('reader_flags') and 'error' not in gate:
            if gate.get('suspicious'):
                place.chunk['reader_flags'] = gate['suspicious']
                gated['слушали'] += 1
            else:
                gated['пропущено'] += 1
        if cfg.clean_ear == 'demand':
            for tool, args, res in place.history:
                if tool == 'listen' and args.get('ear') == 'clean' and isinstance(res, dict) and 'error' not in res:
                    gated['третий голос'] = gated.get('третий голос', 0) + 1
                    gated['секунд третьего голоса'] = round(gated.get('секунд третьего голоса', 0)
                                                            + (res['b'] - res['a']), 1)

    async def _place(i: int, c: dict) -> None:
        if not c.get('raw'):
            return
        nb = (float(chunks[i - 1]['start']) if i > 0 else float(c['start']),
              float(chunks[i + 1]['end']) if i + 1 < len(chunks) else float(c['end']))
        place = Place(f'chunk:{i}', 'arbitrate', chunk=c, canon=canon, gate_terms=gate_terms,
                      audio_sec=st.audio_sec, neighbours=nb)
        slices = Slices(d, st.tmp, st.wav, f'a{i}')
        reg = arbitrate_tools(place, d, slices, journal=st.journal, meter=st.meter)
        try:
            fin = await decide_place(place, policy, reg, cfg.graph_place_steps)
            _account(place)
            decisions = []
            if fin.decision == 'apply' and place.heard.get('second') is not None:
                decisions = (await reg.call('apply_swaps'))['decisions']
            for x in decisions:
                row = {'start': round(c['start'], 2), 'end': round(c['end'], 2), **x}
                arbitrate_log.append(row)
                ev.emit('arbitrate.swap', **{'from': row['start'], 'to': row['end']},
                        was=x['was'], now=x['now'], by=x['by'], taken=x.get('taken', False))
            st.decisions.append({'place': place.name, 'start': round(c['start'], 2), 'end': round(c['end'], 2),
                                 'decision': fin.decision, 'why': fin.why, 'steps': len(place.history),
                                 'taken': sum(1 for x in decisions if x.get('taken')),
                                 **({'exhausted': True} if place.exhausted else {})})
        except Exception as error:      # noqa: BLE001 — стадия не имеет права ронять запись
            _account(place)
            failed['n'] += 1
            log.warning('арбитраж %.1f-%.1f с не вышел: %s: %s',
                        c['start'], c['end'], type(error).__name__, error)
        finally:
            slices.cleanup()

    await asyncio.gather(*(_place(i, c) for i, c in enumerate(chunks)))
    arbitrate_log.sort(key=lambda r: (r['start'], r['i']))
    st.arbitrate_log = arbitrate_log
    st.tm['arbitrate_s'] = round(time.monotonic() - _t, 1)
    counts = {k: sum(1 for x in arbitrate_log if x['by'] == k)
              for k in ('частота', 'канон', 'голосование', 'вето', 'спорно')}
    if cfg.arbitrate_gate == 'reader' or cfg.clean_ear == 'demand':
        counts.update(gated)
        st.tm['arbitrate_gate'] = dict(gated)
    if failed['n']:
        # ⚠️ Число — в артефакт и одной строкой ошибки: расшифровка есть, но на этих кусках она
        # без арбитража (бэкенд упал посреди длинной записи, а прогон выглядел удавшимся).
        st.tm['arbitrate_failed'] = failed['n']
        counts['не вышло'] = failed['n']
        log.error('стадия арбитража не вышла на %d кусках из %d — расшифровка на них без арбитража'
                  ' (бэкенд недоступен?)', failed['n'], len(chunks))
    ev.stage_done('arbitrate', st.tm['arbitrate_s'], n=len(arbitrate_log), **counts)
    if arbitrate_log:
        log.info('арбитраж: %d решений (%s)', len(arbitrate_log),
                 ', '.join(f'{k}: {v}' for k, v in counts.items() if v))


# --- turns --------------------------------------------------------------------------------------

async def turns(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    # группировка подряд идущих чанков одного кластера в реплики
    out: list[dict] = []
    for c in st.chunks:
        if out and out[-1]['cluster'] == c['speaker']:
            out[-1]['chunks'].append(c)
        else:
            out.append({'cluster': c['speaker'], 'start': c['start'], 'chunks': [c]})
    for t in out:
        t['segments'] = [s for c in t['chunks'] for s in c['segments']]
        t['end'] = round(max([c['end'] for c in t['chunks']] + [s['end'] for s in t['segments']]), 2)
    # Сверка после склейки: сигнал даёт непрерывная дыра, а не сумма распознанного.
    for t, (a, b) in zip(out, d.coverage.turn_windows(out, st.audio_sec)):
        for h0, h1 in d.coverage.holes(d.coverage.turn_segments(t), a, b, cfg.coverage_warn_s):
            log.warning('speech lost: %.1fs at %.1fs (turn %.1f-%.1f)', h1 - h0, h0, a, b)
    st.turns = out


# --- final-round --------------------------------------------------------------------------------
# Шаг 1: через `pipeline._final_round` с портом уха. Шаг 3 разложит реплику на цикл по месту.

async def final_round(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    turns_ = st.turns
    ev.stage('final-round')
    _t = time.monotonic()
    st.dsum = await d.doc_summary(st.full_text, d.llm)
    for t in turns_:
        t['raw'] = ' '.join(c['raw'] for c in t['chunks'] if c['raw']).strip()
    # Подтверждённые каноники — ЗАЩИТА: замена, ломающая уже верный термин, отбрасывается.
    st.protect = list(cfg.always_terms) + [c for h in st.seed for c in h['canonicals']]
    # ⚠️ `protect` — постоянные термины для `_term_survives`; известные слова записи — отдельное
    # имя `known` (первый вариант затенил `protect` пустым кортежем и отключил старое вето).
    st.known = ()
    if cfg.protect_known:
        st.known = tuple(dict.fromkeys(
            [x for x in list((st.hints or {}).get('terms') or ()) + list((st.hints or {}).get('names') or ())
             if isinstance(x, str)] + [x for x in (cfg.always_terms or ()) if x]))

    async def _ear(turn: dict, was: str) -> str | None:
        """Чистое ухо на 30 с вокруг спорного слова реплики — окно, не слово (ADR-0030)."""
        segs = turn.get('segments') or []
        if not segs:
            return None
        at = None
        for sg in segs:                              # точные времена от декодера, если есть
            for w in sg.get('words') or ():
                if (w.get('word') or '').strip().strip('.,!?;:«»"').lower() == was.strip('.,!?;:«»"').lower():
                    at = float(w['start'])
                    break
            if at is not None:
                break
        if at is None:                               # иначе — по доле слова в реплике
            words = (turn.get('raw') or '').split()
            pos = next((k for k, w in enumerate(words)
                        if w.strip('.,!?;:«»"').lower() == was.strip('.,!?;:«»"').lower()), None)
            if pos is None:
                return None
            a0, b0 = segs[0]['start'], segs[-1]['end']
            at = a0 + (b0 - a0) * pos / max(1, len(words))
        sl = str(Path(st.tmp) / f'ear{int(at * 100)}.wav')
        try:
            await asyncio.to_thread(d._slice, st.wav, max(0.0, at - 15), min(st.audio_sec, at + 15), sl)
            async with d._res('whisper', cfg.whisper_slots):
                got = await asyncio.to_thread(lambda: d.audio_clients.asr(sl, ''))
            return got.get('text') or ''
        except Exception as error:      # noqa: BLE001 — ухо не роняет запись
            log.warning('арбитраж звуком не вышел (%s): %s', was, error)
            return None
        finally:
            Path(sl).unlink(missing_ok=True)

    n_round, n_failed = await d._final_round(
        turns_, st.dsum, st.gloss, d.llm, cfg.round_concurrency, ev.step, st.protect, emit=ev.emit,
        **({'protect': st.known} if st.known else {}),
        **({'ear': _ear} if (cfg.protect_known and cfg.final_ear) else {}))
    st.raw_side = {f"{t['start']:.1f}": {'raw': t['raw'], 'final': t['final']}
                   for t in turns_ if t['final'] != t['raw']}
    st.tm['round_s'] = round(time.monotonic() - _t, 1)
    st.tm['n_round_turns'] = n_round
    ev.stage_done('final-round', st.tm['round_s'], n=len(turns_), changed=n_round, failed=n_failed)
    # Плоский журнал вердиктов — в артефакт; из реплик убираем, чтобы не дублировать.
    st.round_log = [{'turn': i, 'start': round(t['start'], 2), **f}
                    for i, t in enumerate(turns_) for f in (t.pop('fixes', None) or ())]
    if n_failed:
        st.tm['n_round_failed'] = n_failed
        log.warning('final-round: %d реплик остались сырыми из-за ошибок LLM', n_failed)


# --- speakers -----------------------------------------------------------------------------------

async def speakers(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    ev.stage('speakers')
    await d._wait_turn(st.ticket)  # реестр — строго в порядке поступления выпусков
    async with d._res('campp', cfg.campp_slots):
        st.cents, st.air = await asyncio.to_thread(d.audio_clients.campp, st.wav, st.spans)
    decided: list[dict] | None = [] if ev.enabled else None
    st.mapping = await asyncio.to_thread(
        d.registry.assign, st.cents, st.air, st.episode or 'adhoc', cfg.registry_path,
        cfg.match_threshold, cfg.max_centroids, decided)
    d._release_turn(st.ticket)
    if decided:
        xy = d._project(st.cents)
        for x in decided:
            ev.emit('spk.vec', xy=xy.get(x['cluster'], [0.0, 0.0]), **x)
    # реплики кластеров без матча (короткий шум) → доминирующий Speaker по air-time
    air_by_lbl: dict[str, float] = defaultdict(float)
    for t in st.turns:
        lbl = st.mapping.get(t['cluster'])
        if lbl:
            air_by_lbl[lbl] += sum(c['end'] - c['start'] for c in t['chunks'])
    dominant = max(air_by_lbl, key=air_by_lbl.get) if air_by_lbl else 'Speaker_0'
    for t in st.turns:
        t['speaker'] = st.mapping.get(t['cluster'], dominant)


async def naming(st: State, d: Deps, ev: Emitter) -> None:
    """Speaker_N → имя (интро = истина, реестр = fallback). speaker_id хранит исходный номер."""
    cfg = d.cfg
    ev.stage('naming')
    _t = time.monotonic()
    st.name_map, st.name_conflicts = await d.name_speakers(
        st.turns, d.registry.names(cfg.registry_path), d.llm, corpus_desc=cfg.corpus_desc)
    st.tm['naming_s'] = round(time.monotonic() - _t, 1)
    ev.stage_done('naming', st.tm['naming_s'], n=len(st.name_map))


# --- assemble / align / coverage ----------------------------------------------------------------

async def assemble(st: State, d: Deps, ev: Emitter) -> None:
    ev.emit('spk.map', mapping=st.mapping, names=st.name_map, conflicts=st.name_conflicts)
    for t in st.turns:
        t['speaker_id'] = t['speaker']
        t['speaker'] = st.name_map.get(t['speaker'], t['speaker'])
    n = st.episode.replace('ep', '') if st.episode else ''
    fm = f"title: {st.title or ('Капитанский мостик №' + n if n else 'Транскрипт')}\nurl: {st.url}"
    lines, plain, out_turns = ['---', fm, '---', ''], [], []
    for t in st.turns:
        lines.append(f"[{t['speaker']}] <!-- t:{t['start']:.1f} --> {t['final']}")
        lines.append('')
        plain.append(t['final'])
        out_turns.append({'speaker': t['speaker'], 'speaker_id': t['speaker_id'],
                          'start': round(t['start'], 1), 'end': t['end'],
                          'text': t['final'], 'raw': t['raw'], 'segments': t['segments'],
                          **({'correction_failed': True} if t.get('correction_failed') else {})})
    st.markdown, st.text, st.out_turns = '\n'.join(lines), ' '.join(plain), out_turns


async def align(st: State, d: Deps, ev: Emitter) -> None:
    """Пословные тайм-коды. Стадия не критичная (торч ставится отдельно) — падает мягко."""
    cfg = d.cfg
    ev.stage('align')
    _t = time.monotonic()
    try:
        async with d._res('align', cfg.align_slots):
            st.words_doc = await asyncio.to_thread(
                d.align.align_turns, st.wav, st.out_turns, st.audio_sec,
                episode=st.episode, device=cfg.align_device)
    except Exception as e:      # noqa: BLE001
        log.warning('word alignment skipped: %s: %s', type(e).__name__, e)
    st.tm['align_s'] = round(time.monotonic() - _t, 1)
    ev.stage_done('align', st.tm['align_s'], n=len((st.words_doc or {}).get('turns') or ()))


async def coverage(st: State, d: Deps, ev: Emitter) -> None:
    cfg = d.cfg
    st.cov = d.coverage.summarize(st.audio_sec, st.segs, st.chunks, st.turns, cfg.hole_min_s)
    cov = st.cov
    log.info('coverage: audio %.0fs, unheard by pass2 %.1fs, recovered %.1fs in %d chunk(s), '
             'retried %d, still lost %.1fs',
             cov['audio_sec'], cov['unheard_sec'], cov['recovered_sec'],
             cov['recovered_chunks'], cov['retried_chunks'], cov['lost_sec'])


# Порядок узлов и условие каждого (по конфигу). Узел без условия идёт всегда.
NODES: list[tuple[str, object, object]] = [
    ('prepare', prepare, None),
    ('diarize', diarize, None),
    ('pass1', pass1, None),
    ('glossary', glossary, None),
    ('chunking', chunking, None),
    ('pass2', pass2, None),
    ('relisten', relisten, lambda cfg: bool(cfg.relisten)),
    ('arbitrate', arbitrate, lambda cfg: bool(cfg.second_model)),
    ('turns', turns, None),
    ('final_round', final_round, None),
    ('speakers', speakers, None),
    ('naming', naming, lambda cfg: bool(cfg.enable_naming)),
    ('assemble', assemble, None),
    ('align', align, lambda cfg: bool(cfg.enable_align)),
    ('coverage', coverage, None),
]
