"""CAM++ embed-эндпоинт (Mac, sherpa-onnx / CoreML): wav + spans → центроиды кластеров + air-time.

Держит CAM++-зависимость (Apple-Silicon-bound) НА МАКЕ; morag-сервис asr-adaptor берёт центроиды по HTTP
и сам ведёт глобальный Speaker_N-реестр. Инлайн `make_embedder/embed/cluster_centroids` — паттерн из
diarizer-service/build_transcript.py (без его тяжёлого import-хвоста). Порт :8126, за Caddy при желании.

POST /embed-centroids (multipart: file=wav 16k mono, spans=JSON [{start,end,speaker}]) →
  {"centroids": {cluster: [float...]}, "air": {cluster: sec}}  — только для substantial-кластеров (≥2с).
Запуск (Mac, diarizer-onnx venv): CAMPP_API_KEY=... uvicorn app:app --host 0.0.0.0 --port 8126
"""
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

EMB = os.environ.get('CAMPP_MODEL') or str(
    Path(os.environ.get('ASR_STACK_HOME') or (Path.home() / 'asr-stack')) / 'models'
    / '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx')
API_KEY = os.environ.get('CAMPP_API_KEY')
MIN_DUR = 2.0   # сегменты короче не берём в центроид
MAX_SEGS = 20   # самых длинных сегментов на центроид
SEG_CAP = 20.0  # окно на сегмент: длинный спан целиком валит CoreML («dynamically resizing for
                # sequence length»), а для центроида голоса 20с с запасом хватает

_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, provider='coreml', num_threads=4))

app = FastAPI(title='CAM++ embed backend')


def _embed(samples: np.ndarray, sr: int) -> np.ndarray:
    st = _ext.create_stream()
    st.accept_waveform(sample_rate=sr, waveform=samples)
    st.input_finished()
    v = np.array(_ext.compute(st), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


GUARD_S = 0.3  # отступ от чужого отрезка: граница диаризации неточна, у края звучит сосед


def _minus(a: float, e: float, others: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Отрезок [a, e] без чужих отрезков, расширенных на GUARD_S с каждой стороны."""
    parts = [(a, e)]
    for o0, o1 in others:
        o0, o1 = o0 - GUARD_S, o1 + GUARD_S
        nxt = []
        for p0, p1 in parts:
            if o1 <= p0 or o0 >= p1:
                nxt.append((p0, p1))
                continue
            if o0 > p0:
                nxt.append((p0, o0))
            if o1 < p1:
                nxt.append((o1, p1))
        parts = nxt
    return parts


def _centroids(wav: str, spans: list[dict], clean: bool = False):
    """Центроид голоса по его отрезкам диаризации.

    `clean` — из отрезка вычитается чужая речь (наложение) и GUARD_S у границы с чужим отрезком:
    центроид, набранный из смеси двух голосов, узнаёт «среднего» человека и склеивает разных
    (идея — выбор «чистых» отрезков для образца голоса у VoiceStudio). Если после вычитания у
    голоса не осталось отрезка от MIN_DUR, берутся прежние: без центроида голос не узнать вовсе.
    `air` — всегда по сырым отрезкам: это эфир, а не качество образца.
    """
    audio, sr = sf.read(wav, dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    byspk = defaultdict(list)
    for s in spans:
        byspk[s['speaker']].append((float(s['start']), float(s['end'])))
    cents, air = {}, {}
    for spk, segs in byspk.items():
        pool = segs
        if clean:
            others = [x for k, v in byspk.items() if k != spk for x in v]
            cleaned = [p for a, e in segs for p in _minus(a, e, others)]
            if any(e - a >= MIN_DUR for a, e in cleaned):
                pool = cleaned
        longs = sorted([(a, e) for a, e in pool if e - a >= MIN_DUR],
                       key=lambda x: -(x[1] - x[0]))[:MAX_SEGS]
        if not longs:
            continue  # чистый шум — без центроида (как build_transcript)
        cap = int(SEG_CAP * sr)
        vecs = []
        for a, e in longs:
            seg = audio[int(a * sr):int(e * sr)][:cap]  # окно ≤SEG_CAP — CoreML не давится длиной
            try:
                vecs.append(_embed(seg, sr))
            except Exception:
                continue  # один плохой сегмент не валит весь центроид
        if not vecs:
            continue
        c = np.mean(vecs, axis=0)
        cents[spk] = (c / (np.linalg.norm(c) + 1e-9)).tolist()
        air[spk] = sum(e - a for a, e in segs)
    return cents, air


@app.get('/health')
def health():
    return {'status': 'ok', 'model': Path(EMB).name}


@app.post('/embed-centroids')
async def embed_centroids(file: UploadFile = File(...), spans: str = Form(...),
                          clean: str = Form(''),
                          authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f'Bearer {API_KEY}':
        raise HTTPException(401, 'unauthorized')
    sp = json.loads(spans)
    tmp = tempfile.mktemp(suffix='.wav')
    Path(tmp).write_bytes(await file.read())
    try:
        cents, air = _centroids(tmp, sp, clean=clean.strip().lower() in ('1', 'true', 'yes', 'on'))
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'centroids': cents, 'air': air}


@app.post('/embed-spans')
async def embed_spans(file: UploadFile = File(...), spans: str = Form(...),
                      authorization: Optional[str] = Header(None)):
    """Вектор на КАЖДЫЙ отрезок (`[{start, end}]` → `[[float] | null]`, в том же порядке).

    Нужен восстановлению склеенного диалога: там голоса ищутся по фразам, а не по кластерам.
    Отрезок короче 0.5 с или неудачный — `null`. ⚠️ Векторы — биометрия: адаптер держит их в
    памяти прогона и в артефакт не кладёт.
    """
    if API_KEY and authorization != f'Bearer {API_KEY}':
        raise HTTPException(401, 'unauthorized')
    sp = json.loads(spans)
    tmp = tempfile.mktemp(suffix='.wav')
    Path(tmp).write_bytes(await file.read())
    try:
        audio, sr = sf.read(tmp, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        out = []
        for s in sp:
            a, e = float(s['start']), float(s['end'])
            if e - a < 0.5:
                out.append(None)
                continue
            try:
                out.append(_embed(audio[int(a * sr):int(e * sr)][:int(SEG_CAP * sr)], sr).tolist())
            except Exception:
                out.append(None)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'vectors': out}
