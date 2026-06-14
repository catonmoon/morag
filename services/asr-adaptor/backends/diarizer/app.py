"""Pyannote-audio diarization service.

Стартует один раз, держит модель в RAM, обслуживает POST /diarize.
Auth: Bearer-токен из env DIARIZER_API_KEY.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pyannote.audio import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
log = logging.getLogger('diarizer')

MODEL_ID = os.environ.get('DIARIZER_MODEL', 'pyannote/speaker-diarization-3.1')
API_KEY = os.environ.get('DIARIZER_API_KEY')
DEVICE_OVERRIDE = os.environ.get('DIARIZER_DEVICE', 'auto').lower()  # auto|cpu|mps|cuda
if not API_KEY:
    raise RuntimeError('DIARIZER_API_KEY env var is required')

_pipeline: Pipeline | None = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _device
    if DEVICE_OVERRIDE == 'cpu':
        _device = torch.device('cpu')
    elif DEVICE_OVERRIDE == 'mps':
        _device = torch.device('mps')
    elif DEVICE_OVERRIDE == 'cuda':
        _device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        _device = torch.device('mps')
    elif torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    log.info('Loading %s on %s ...', MODEL_ID, _device)
    t0 = time.time()
    _pipeline = Pipeline.from_pretrained(MODEL_ID)
    _pipeline.to(_device)
    log.info('Model loaded in %.1fs', time.time() - t0)

    # Прогрев: первый MPS-прогон компилирует Metal-шейдеры (разово, минуты).
    # Гоняем 10с шума при старте, чтобы первый реальный запрос был уже быстрым.
    if _device.type == 'mps':
        import numpy as np
        log.info('Warming up MPS shaders (10s noise) ...')
        t0 = time.time()
        sr = 16000
        noise = (np.random.randn(sr * 10).astype('float32') * 0.05)
        wav = torch.from_numpy(noise).unsqueeze(0)  # (channel, time)
        _pipeline({'waveform': wav, 'sample_rate': sr})
        log.info('Warmup done in %.1fs', time.time() - t0)
    yield


app = FastAPI(title='pyannote diarizer', lifespan=lifespan)


def _check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Missing Bearer token')
    token = authorization.removeprefix('Bearer ').strip()
    if token != API_KEY:
        raise HTTPException(403, 'Invalid token')


@app.get('/health')
def health() -> dict:
    return {
        'ok': _pipeline is not None,
        'device': str(_device),
        'model': MODEL_ID,
    }


@app.post('/diarize')
async def diarize(
    audio: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    authorization: str | None = Header(None),
) -> dict:
    _check_auth(authorization)
    if _pipeline is None:
        raise HTTPException(503, 'Model not loaded')

    # Сохраняем во временный файл — pyannote читает с диска через torchcodec
    suffix = Path(audio.filename or 'audio').suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    size_mb = len(content) / 1024 / 1024
    log.info(
        'diarize: file=%s size=%.1fMB num=%s min=%s max=%s',
        audio.filename, size_mb, num_speakers, min_speakers, max_speakers,
    )

    kwargs = {}
    if num_speakers is not None:
        kwargs['num_speakers'] = num_speakers
    if min_speakers is not None:
        kwargs['min_speakers'] = min_speakers
    if max_speakers is not None:
        kwargs['max_speakers'] = max_speakers

    try:
        t0 = time.time()
        diarization = _pipeline(tmp_path, **kwargs)
        elapsed = time.time() - t0
    finally:
        os.unlink(tmp_path)

    spans = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        spans.append({
            'start': round(turn.start, 3),
            'end': round(turn.end, 3),
            'speaker': speaker,
        })
    spans.sort(key=lambda s: s['start'])

    audio_dur = max((s['end'] for s in spans), default=0.0)
    log.info(
        'diarize done: %d spans, %d clusters, audio=%.1fs, elapsed=%.1fs, RTF=%.3f',
        len(spans),
        len({s['speaker'] for s in spans}),
        audio_dur,
        elapsed,
        elapsed / audio_dur if audio_dur else 0,
    )

    return {
        'spans': spans,
        'audio_duration_sec': audio_dur,
        'elapsed_sec': round(elapsed, 1),
        'rtf': round(elapsed / audio_dur, 4) if audio_dur else None,
        'num_clusters': len({s['speaker'] for s in spans}),
    }
