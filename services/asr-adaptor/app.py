"""asr-adaptor — OpenAI-совместимый `POST /v1/audio/transcriptions` адаптер (morag-сервис).

Аудио → обогащённый транскрипт (Speaker_N + тайминги + канонизация сущностей). Внутри — весь пайплайн
(diarize → пасс-1 → глоссарий → пасс-2 → финал-раунд → Speaker_N), аудио на Маке по HTTP, LLM (облако) =
Grok-4.3 на OpenRouter (reasoning off). См. CLAUDE.md.

Ответ: стандартный verbose_json (`text`/`segments`) + кастомный `x_enriched` (markdown, turns, raw, timing).
mode=async (дефолт) → 202 {job_id}, поллинг GET /v1/jobs/{id}. mode=sync — для коротких/smoke.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import audio_clients
import jobs
from config import CFG
from pipeline import run_pipeline

app = FastAPI(title='asr-adaptor', version='1.0')
_LLM = CFG.build_llm()


def _enriched(r: dict) -> dict:
    return {
        'task': 'transcribe', 'language': 'ru', 'text': r['text'],
        'segments': [{'id': i, 'start': t['start'], 'end': t['start'], 'text': t['text']}
                     for i, t in enumerate(r['turns'])],
        'x_enriched': {'format': 'morag-md-v1', 'markdown': r['markdown'], 'turns': r['turns'],
                       'raw_sidecar': r['raw_sidecar'], 'timing': r['timing'],
                       'speaker_map': r['speaker_map'],
                       'speaker_names': r.get('speaker_names', {}),
                       'name_conflicts': r.get('name_conflicts', [])},
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'downstream': audio_clients.health(), 'llm': CFG.llm_model}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': 'asr-adaptor', 'object': 'model'}]}


@app.post('/v1/audio/transcriptions')
async def transcribe(file: UploadFile = File(...), model: str = Form('asr-adaptor'),
                     response_format: str = Form('verbose_json'), mode: str = Form(''),
                     episode: str = Form(''), title: str = Form(''), url: str = Form('')):
    suffix = Path(file.filename or 'audio').suffix or '.mp3'
    tmp = tempfile.mktemp(suffix=suffix)
    Path(tmp).write_bytes(await file.read())

    async def job(progress):
        try:
            return _enriched(await run_pipeline(
                tmp, _LLM, episode=episode, title=title, url=url, progress=progress))
        finally:
            Path(tmp).unlink(missing_ok=True)

    if (mode or CFG.mode) == 'sync':
        return await job(lambda _: None)
    return {'job_id': jobs.submit(job), 'status': 'queued'}


@app.get('/v1/jobs/{job_id}')
def job_status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, 'job not found')
    out = {'job_id': job_id, 'status': j['status'], 'progress': j.get('progress', '')}
    if j['status'] == 'done':
        out['result'] = j['result']
    elif j['status'] == 'error':
        out['error'] = j.get('error')
    return out
