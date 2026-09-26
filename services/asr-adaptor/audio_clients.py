"""HTTP-клиенты к аудио-бэкендам на Маке: diarize(:8090), ASR(:8123), CAM++(:8126).

Блокирующие (requests) — оркестратор зовёт их через asyncio.to_thread. Apple-Silicon-bound модели
остаются на Маке; сервис их только дёргает.
"""
from __future__ import annotations

import json

import requests

from config import CFG


def diarize(wav_path: str, min_spk: int | None = None, max_spk: int | None = None) -> list[dict]:
    # Диапазон голосов задаётся корпусом (ASR_MIN_SPEAKERS/ASR_MAX_SPEAKERS): у подкаста их 2-4,
    # у совещания бывает за десяток, и прежний зашитый потолок 10 схлопывал лишние голоса в чужие.
    min_spk = CFG.min_speakers if min_spk is None else min_spk
    max_spk = CFG.max_speakers if max_spk is None else max_spk
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


# Пасс-1 слушает файл ЦЕЛИКОМ, поэтому его таймаут обязан расти с длиной записи. Прежний
# фиксированный потолок в 300 с молча резал ровно длинные лекции: замерено на курсе QA — шесть
# записей от 172 до 328 минут сорвались на `Read timed out`, и все шесть после конвейера.
# ⚠️ Делитель взят по ЗАМЕРУ, а не наугад: на этой машине пасс-1 идёт 31-34× реального времени
# (самая медленная из 27 записей — 31.2×). Берём 8× как «худший мыслимый случай» — четырёхкратный
# запас к измеренному. Записи на 328 минут это даёт 41 минуту потолка при нужных десяти.
# ⓘ Пол в 300 с оставлен для коротких кусков пасса-2: там время уходит не на счёт, а на очередь.
_ASR_TIMEOUT_MIN = 300
_ASR_WORST_SPEED = 8


def _asr_timeout(wav_path: str) -> int:
    try:
        from stages.coverage import wav_duration
        return max(_ASR_TIMEOUT_MIN, int(wav_duration(wav_path) / _ASR_WORST_SPEED))
    except Exception:      # не смогли прочесть заголовок — прежнее поведение
        return _ASR_TIMEOUT_MIN


def asr(wav_path: str, prompt: str = '', temperature: str = '', words: bool = False,
        model: str = '') -> dict:
    """podlodka через transcribe_backend, всегда verbose_json → {'text', 'segments'}.

    Сегменты нужны ОБОИМ пассам: пасс-1 по ним нарезает чанки, пасс-2 — чтобы было видно, сколько
    звука кусок реально покрыл. Без них потеря речи невидима: текст перескакивает через кусок, а
    тайм-коды соседних реплик не рвутся (см. stages/coverage.py). Расшифровка та же — verbose_json
    лишь подробнее отвечает, поэтому отдельной ветки «только текст» больше нет.
    """
    # `model` — ДРУГАЯ модель прослушивания (второе ухо, ADR-0030); пусто — модель профиля.
    data = {'model': model or CFG.asr_model, 'language': 'ru', 'response_format': 'verbose_json'}
    if prompt:
        data['prompt'] = prompt
    # ⚠️ Температура — ЛЕСЕНКА отступа («0,0.2,0.4»), а не число: по ней у бэкенда включаются его
    # датчики галлюцинации. Пусто — бэкенд решает сам (у него дефолт «0», то есть детерминированно).
    # ⚠️⚠️ Полную лесенку (до 1.0) не просить: замерено, что на невнятном звуке она не спасает, а
    # сочиняет — 3 ответа из 18 пришли текстом на чужих языках там, где при 0 модель молчала.
    if temperature:
        data['temperature'] = temperature
    # Времена слов от декодера (`segments[].words`): приблизительные, но доступные сразу — см.
    # бэкенд. По умолчанию не просим: ответ и поведение прежние байт в байт.
    if words:
        data['word_timestamps'] = '1'
    headers = {'Authorization': f'Bearer {CFG.asr_key}'} if CFG.asr_key else {}
    with open(wav_path, 'rb') as f:
        r = requests.post(CFG.asr_url, data=data, files={'file': f}, headers=headers,
                          timeout=_asr_timeout(wav_path))
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
