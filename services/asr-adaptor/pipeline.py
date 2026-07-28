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
from stages.final_round import chunk_summary, correct, doc_summary, has_entity_signal
from stages.glossary import build_glossary, relevant
from stages.namer import name_speakers
from stages.prompt_budget import WhisperTokenCounter, build_prompt

log = logging.getLogger('asr')

PAD_S = 0.3  # паддинг куска при повторе пустого чанка (см. _decode)


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


async def _decode(wav: str, sl: str, c: dict, prompt: str, audio_sec: float) -> None:
    """Кусок → текст + сегменты в АБСОЛЮТНОМ времени; пусто → один повтор с вариацией.

    Простой повтор бессмыслен: декодирование детерминировано (temperature=0) и вернёт ровно то же.
    Поэтому повтор идёт БЕЗ initial_prompt и с паддингом — снимаем разом и подавление промптом, и
    обрезку речи ровно на границе куска. Паддинг маленький (0.3 с): он может прихватить край
    соседнего слова, и это дешевле, чем потерять кусок целиком.
    """
    await asyncio.to_thread(_slice, wav, c['start'], c['end'], sl)
    r = await asyncio.to_thread(audio_clients.asr, sl, prompt)
    off = c['start']
    if not r['text'] and CFG.retry_empty:
        a, b = max(0.0, c['start'] - PAD_S), min(audio_sec, c['end'] + PAD_S)
        await asyncio.to_thread(_slice, wav, a, b, sl)
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

    def step(m):
        if progress:
            progress(m)

    try:
        wav = str(tmp / 'in.wav')
        await asyncio.to_thread(_to_wav, audio_path, wav)

        step('diarize')
        _t = time.monotonic()
        spans = await asyncio.to_thread(audio_clients.diarize, wav)
        tm['diarize_s'] = round(time.monotonic() - _t, 1)

        audio_sec = coverage.wav_duration(wav)

        step('pass1')
        _t = time.monotonic()
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
            prompt = build_prompt(relevant(src, gloss), counter, CFG.prompt_budget)
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
        raw_side, n_round = {}, 0
        for t in turns:
            raw = ' '.join(c['raw'] for c in t['chunks'] if c['raw']).strip()
            t['raw'] = raw
            if len(raw.split()) >= 3 and has_entity_signal(raw, gloss):
                n_round += 1
                csum = await chunk_summary(dsum, raw, llm)
                t['final'] = await correct(raw, dsum, csum, relevant(raw, gloss), llm)
            else:
                t['final'] = raw
            if t['final'] != raw:
                raw_side[f"{t['start']:.1f}"] = {'raw': raw, 'final': t['final']}
        tm['round_s'] = round(time.monotonic() - _t, 1)
        tm['n_round_turns'] = n_round

        step('speakers')
        cents, air = await asyncio.to_thread(audio_clients.campp, wav, spans)
        mapping = await asyncio.to_thread(
            registry.assign, cents, air, episode or 'adhoc', CFG.registry_path,
            CFG.match_threshold, CFG.max_centroids)
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
                              'text': t['final'], 'raw': t['raw'], 'segments': t['segments']})

        # Пословные тайм-коды: звук и реплики уже здесь, поэтому и время слова считается здесь.
        # Стадия не критичная (торч ставится отдельно, см. requirements-align.txt) — падает мягко.
        words_doc = None
        if CFG.enable_align:
            step('align')
            _t = time.monotonic()
            try:
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
                'coverage': cov, 'words': words_doc}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
