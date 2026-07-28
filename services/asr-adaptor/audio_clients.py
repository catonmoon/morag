"""HTTP-клиенты к аудио-бэкендам на Маке: diarize(:8090), ASR(:8123), CAM++(:8126).

Блокирующие (requests) — оркестратор зовёт их через asyncio.to_thread. Apple-Silicon-bound модели
остаются на Маке; сервис их только дёргает.
"""
from __future__ import annotations

import json

import requests

from config import CFG


def diarize(wav_path: str, min_spk: int = 1, max_spk: int = 10) -> list[dict]:
    headers = {'Authorization': f'Bearer {CFG.diarizer_key}'} if CFG.diarizer_key else {}
    with open(wav_path, 'rb') as f:
        r = requests.post(CFG.diarizer_url, files={'audio': f},
                          data={'min_speakers': str(min_spk), 'max_speakers': str(max_spk)},
                          headers=headers, timeout=900)
    r.raise_for_status()
    d = r.json()
    if isinstance(d, list):
        return d
    return d.get('spans') or d.get('segments') or next((v for v in d.values() if isinstance(v, list)), [])


def asr(wav_path: str, prompt: str = '') -> dict:
    """podlodka через transcribe_backend, всегда verbose_json → {'text', 'segments'}.

    Сегменты нужны ОБОИМ пассам: пасс-1 по ним нарезает чанки, пасс-2 — чтобы было видно, сколько
    звука кусок реально покрыл. Без них потеря речи невидима: текст перескакивает через кусок, а
    тайм-коды соседних реплик не рвутся (см. stages/coverage.py). Расшифровка та же — verbose_json
    лишь подробнее отвечает, поэтому отдельной ветки «только текст» больше нет.
    """
    data = {'model': CFG.asr_model, 'language': 'ru', 'response_format': 'verbose_json'}
    if prompt:
        data['prompt'] = prompt
    headers = {'Authorization': f'Bearer {CFG.asr_key}'} if CFG.asr_key else {}
    with open(wav_path, 'rb') as f:
        r = requests.post(CFG.asr_url, data=data, files={'file': f}, headers=headers, timeout=300)
    r.raise_for_status()
    j = r.json()
    return {'text': (j.get('text') or '').strip(), 'segments': j.get('segments') or []}


def campp(wav_path: str, spans: list[dict]) -> tuple[dict, dict]:
    """CAM++ центроиды substantial-кластеров → (centroids{cluster:[float]}, air{cluster:sec})."""
    headers = {'Authorization': f'Bearer {CFG.campp_key}'} if CFG.campp_key else {}
    with open(wav_path, 'rb') as f:
        r = requests.post(CFG.campp_url, files={'file': f}, data={'spans': json.dumps(spans)},
                          headers=headers, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d.get('centroids', {}), d.get('air', {})


def health() -> dict:
    """Пинг всех downstream-бэкендов (для /health сервиса)."""
    out = {}
    for name, url in (('diarizer', CFG.diarizer_url), ('asr', CFG.asr_url), ('campp', CFG.campp_url)):
        base = url.rsplit('/', 1)[0] if name == 'campp' else url.split('/v1')[0] if '/v1' in url else url.rsplit('/', 1)[0]
        try:
            requests.get(base.rstrip('/') + '/health', timeout=5)
            out[name] = 'ok'
        except Exception as e:
            out[name] = f'err: {str(e)[:40]}'
    return out
