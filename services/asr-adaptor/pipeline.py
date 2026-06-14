"""Оркестрация end-to-end (порт adventures/podlodka-asr/pass2_full.main на morag-core, async).

audio → diarize → пасс-1(whole-file) → глоссарий[LLM] → чанк → пасс-2(per-chunk+промпт) → реплики →
финал-раунд[LLM] → Speaker_N(реестр) → обогащённый транскрипт (.md + turns + raw-сайдкар + timing).
Аудио (diarize/asr/campp) — блокирующий HTTP к Маку через asyncio.to_thread; LLM — await morag LLMClient.
Один in-flight job (Mac GPU — горло) — обеспечивается в jobs.py/app.py.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import audio_clients
from config import CFG
from stages import registry
from stages.chunking import chunk as chunk_fn
from stages.final_round import chunk_summary, correct, doc_summary, has_entity_signal
from stages.glossary import build_glossary, relevant
from stages.namer import name_speakers
from stages.prompt_budget import WhisperTokenCounter, build_prompt


def _ffmpeg(args):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *args], check=True)


def _to_wav(src, dst):
    _ffmpeg(['-i', src, '-ar', '16000', '-ac', '1', dst])


def _slice(wav, a, b, dst):
    _ffmpeg(['-ss', f'{a:.2f}', '-to', f'{b:.2f}', '-i', wav, '-ar', '16000', '-ac', '1', dst])


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

        step('pass1')
        _t = time.monotonic()
        p1 = await asyncio.to_thread(audio_clients.asr, wav, '', True)
        segs = p1.get('segments') or []
        words = {'segments': segs}
        full_text = ' '.join((s.get('text') or '').strip() for s in segs)
        tm['pass1_s'] = round(time.monotonic() - _t, 1)

        step('glossary')
        _t = time.monotonic()
        gloss = await build_glossary(full_text, llm)
        tm['glossary_s'] = round(time.monotonic() - _t, 1)

        chunks = chunk_fn(words, spans)
        counter = WhisperTokenCounter(CFG.whisper_tokenizer)

        step('pass2')
        _t = time.monotonic()
        for i, c in enumerate(chunks):
            prompt = build_prompt(relevant(c['text'], gloss), counter, CFG.prompt_budget)
            sl = str(tmp / f'c{i}.wav')
            await asyncio.to_thread(_slice, wav, c['start'], c['end'], sl)
            c['raw'] = await asyncio.to_thread(audio_clients.asr, sl, prompt, False)
            Path(sl).unlink(missing_ok=True)
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

        step('final-round')
        _t = time.monotonic()
        dsum = await doc_summary(full_text, llm)
        raw_side, n_round = {}, 0
        for t in turns:
            raw = ' '.join(c['raw'] for c in t['chunks']).strip()
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
                              'start': round(t['start'], 1), 'text': t['final'], 'raw': t['raw']})
        tm['total_s'] = round(time.monotonic() - t0, 1)
        return {'markdown': '\n'.join(lines), 'text': ' '.join(plain), 'turns': out_turns,
                'raw_sidecar': raw_side, 'timing': tm, 'speaker_map': mapping,
                'speaker_names': name_map, 'name_conflicts': name_conflicts}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
