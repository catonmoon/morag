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
from stages.chunking import chunk as chunk_fn
from stages.chunking import gap_chunks
from stages.final_round import correct, doc_summary, has_entity_signal, recall_entities
from stages.glossary import build_glossary, relevant
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


async def _final_round(turns, dsum: str, gloss, llm, concurrency: int, step,
                       always=(), sweep_delay: float = 30.0) -> tuple[int, int]:
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
                t['final'] = await correct(raw, dsum, _around(turns, i, CFG.context_turns),
                                           relevant(raw, gloss), llm, always, recalled)
            ok = True
        except Exception as e:
            t['final'] = raw
            log.warning('final-round failed at %.1fs (%s: %s)',
                        t['start'], type(e).__name__, str(e)[:120])
        done += 1
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
    async with _res('whisper', CFG.whisper_slots):
        r = await asyncio.to_thread(audio_clients.asr, sl, prompt)
    off = c['start']
    if not r['text'] and CFG.retry_empty:
        a, b = max(0.0, c['start'] - PAD_S), min(audio_sec, c['end'] + PAD_S)
        await asyncio.to_thread(_slice, wav, a, b, sl)
        async with _res('whisper', CFG.whisper_slots):
            again = await asyncio.to_thread(audio_clients.asr, sl, '')
        if again['text']:
            r, off, c['retried'] = again, a, True
    Path(sl).unlink(missing_ok=True)

    c['raw'] = r['text']
    c['segments'] = [{'start': round(off + float(s.get('start') or 0.0), 2),
                      'end': round(off + float(s.get('end') or 0.0), 2),
                      'text': (s.get('text') or '').strip()} for s in r['segments']]
    if not c['raw']:
        log.warning('pass2 returned nothing for %.1f-%.1fs (speaker %s)',
                    c['start'], c['end'], c.get('speaker'))


async def run_pipeline(audio_path: str, llm, *, episode: str = '', title: str = '',
                       url: str = '', progress=None) -> dict:
    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix='asr_'))
    tm: dict = {}
    ticket = _take_ticket()  # порядок реестра = порядок поступления выпусков

    def step(m):
        if progress:
            progress(m)

    try:
        wav = str(tmp / 'in.wav')
        await asyncio.to_thread(_to_wav, audio_path, wav)

        step('diarize')
        _t = time.monotonic()
        async with _res('diarize', CFG.diarize_slots):
            spans = await asyncio.to_thread(audio_clients.diarize, wav)
        tm['diarize_s'] = round(time.monotonic() - _t, 1)

        audio_sec = coverage.wav_duration(wav)

        step('pass1')
        _t = time.monotonic()
        async with _res('whisper', CFG.whisper_slots):
            p1 = await asyncio.to_thread(audio_clients.asr, wav, '')
        segs = p1['segments']
        words = {'segments': segs}
        full_text = ' '.join((s.get('text') or '').strip() for s in segs)
        tm['pass1_s'] = round(time.monotonic() - _t, 1)

        p1_holes = coverage.holes([(s['start'], s['end']) for s in segs], 0.0, audio_sec,
                                  CFG.hole_min_s)
        if p1_holes:
            log.info('pass1 skipped %.1fs in %d place(s) — часть закроет пасс-2 внутри чанков',
                     sum(b - a for a, b in p1_holes), len(p1_holes))

        step('glossary')
        _t = time.monotonic()
        gloss = await build_glossary(full_text, llm)
        tm['glossary_s'] = round(time.monotonic() - _t, 1)
        tm['n_glossary'] = len(gloss)  # размер глоссария — чем кормим подсказку пасса-2 (бюджет ≤200 ток.)

        chunks = chunk_fn(words, spans)
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

        step('pass2')
        _t = time.monotonic()
        for i, c in enumerate(chunks):
            src = c['text'] or _neighbour_text(chunks, i)
            prompt = build_prompt(relevant(src, gloss), counter, CFG.prompt_budget, CFG.always_terms)
            await _decode(wav, str(tmp / f'c{i}.wav'), c, prompt, audio_sec)
            if (i + 1) % 20 == 0:
                step(f'pass2 {i + 1}/{len(chunks)}')
        tm['pass2_s'] = round(time.monotonic() - _t, 1)
        tm['n_chunks'] = len(chunks)

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

        step('final-round')
        _t = time.monotonic()
        dsum = await doc_summary(full_text, llm)
        for t in turns:
            t['raw'] = ' '.join(c['raw'] for c in t['chunks'] if c['raw']).strip()
        n_round, n_failed = await _final_round(turns, dsum, gloss, llm, CFG.round_concurrency,
                                              step, CFG.always_terms)
        raw_side = {f"{t['start']:.1f}": {'raw': t['raw'], 'final': t['final']}
                    for t in turns if t['final'] != t['raw']}
        tm['round_s'] = round(time.monotonic() - _t, 1)
        tm['n_round_turns'] = n_round
        if n_failed:
            tm['n_round_failed'] = n_failed
            log.warning('final-round: %d реплик остались сырыми из-за ошибок LLM', n_failed)

        step('speakers')
        await _wait_turn(ticket)  # реестр — строго в порядке поступления выпусков
        async with _res('campp', CFG.campp_slots):
            cents, air = await asyncio.to_thread(audio_clients.campp, wav, spans)
        mapping = await asyncio.to_thread(
            registry.assign, cents, air, episode or 'adhoc', CFG.registry_path,
            CFG.match_threshold, CFG.max_centroids)
        _release_turn(ticket)
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
            step('naming')
            _t = time.monotonic()
            name_map, name_conflicts = await name_speakers(
                turns, registry.names(CFG.registry_path), llm)
            tm['naming_s'] = round(time.monotonic() - _t, 1)
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
            step('align')
            _t = time.monotonic()
            try:
                async with _res('align', CFG.align_slots):
                    words_doc = await asyncio.to_thread(
                        align.align_turns, wav, out_turns, audio_sec,
                        episode=episode, device=CFG.align_device)
            except Exception as e:
                log.warning('word alignment skipped: %s: %s', type(e).__name__, e)
            tm['align_s'] = round(time.monotonic() - _t, 1)

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
                # глоссарий и сводка — в артефакт: офлайн-долечивание реплик без пересчёта
                'glossary': gloss, 'doc_summary': dsum}
    finally:
        _release_turn(ticket)  # идемпотентно: упавший выпуск не вешает очередь реестра
        shutil.rmtree(tmp, ignore_errors=True)
