"""Глобальный кросс-эпизодный реестр спикеров по голосу → стабильный `Speaker_N`.

Хранилище — JSON ЛОКАЛЬНО на машине транскрибации (НЕ Qdrant; см. CLAUDE.md). Голос (CAM++ центроид,
L2-normed, с Mac-эндпоинта) → integer ID: новый голос (cos<threshold ко всем) → next_id++; тот же голос
в другом выпуске → тот же Speaker_N. `best_match` — порт diarizer-service/build_transcript.py.
Персист: fcntl.flock (read-modify-write) + атомарный os.replace + .bak. Прогон корпуса — последовательно.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import shutil
from pathlib import Path

import numpy as np

# Новый голос длиннее → отдельный Speaker; короче и нет матча → fold к ближайшему.
# ⚠️ Это доменное предположение, а не универсальное: «участник говорит хотя бы пару минут» верно
# для подкаста и ломается на записи митапа, где вопрос из зала длится десятки секунд — тогда КАЖДЫЙ
# спрашивающий приклеивается к докладчику. Замерено на 13-минутной записи: диаризатор нашёл шесть
# голосов, реестр свёл их в один, и подъём ASR_MATCH_THRESHOLD не помогал вовсе — отсекала именно
# эта константа. Поэтому она читается из окружения; дефолт прежний.
MIN_GUEST_MIN = float(os.environ.get('ASR_MIN_GUEST_MIN', '2.0'))


def _empty() -> dict:
    return {'version': 1, 'next_id': 0, 'speakers': {}}


def _today() -> str:
    return _dt.date.today().isoformat()


def names(registry_path: str) -> dict[str, str]:
    """{'Speaker_N': имя} для спикеров реестра с проставленным `name` (глобальные имена → fallback
    для авто-наминга). Пусто, если реестра/имён нет."""
    path = Path(registry_path)
    if not path.exists():
        return {}
    try:
        reg = json.loads(path.read_text())
    except Exception:
        return {}
    return {f'Speaker_{sid}': rec['name'] for sid, rec in reg.get('speakers', {}).items()
            if rec.get('name')}


def best_match(cent: np.ndarray, speakers: dict) -> tuple[str | None, float]:
    """(speaker_id, max_cosine) по всем центроидам всех спикеров. Векторы L2-normed → cos = dot."""
    best_id, best = None, -1.0
    for sid, rec in speakers.items():
        for c in rec.get('centroids', []):
            cos = float(np.dot(cent, np.asarray(c, dtype=np.float32)))
            if cos > best:
                best_id, best = sid, cos
    return best_id, best


def _prov(episode: str, cluster: str, air: float) -> dict:
    return {'episode': episode, 'cluster': cluster, 'air_sec': round(air, 1), 'added': _today()}


def assign(cents: dict, air: dict, episode: str, registry_path: str,
           threshold: float = 0.55, max_centroids: int = 8) -> dict:
    """cents {cluster: centroid[]} + air {cluster: sec} (substantial-кластеры от CAM++-эндпоинта)
    → {cluster: 'Speaker_N'}. Мутирует+персистит реестр под flock. Кластеры по air-time desc (детерминизм)."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix('.lock')
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        reg = _empty()
        if path.exists():
            try:
                reg = json.loads(path.read_text())
            except Exception:
                reg = _empty()
        sp = reg.setdefault('speakers', {})
        reg.setdefault('next_id', 0)
        mapping = {}
        for cl in sorted(air, key=lambda c: -air[c]):
            cent = np.asarray(cents[cl], dtype=np.float32)
            pid, cos = best_match(cent, sp)
            if pid is not None and cos >= threshold:
                mapping[cl] = f'Speaker_{pid}'
                rec = sp[pid]
                if len(rec['centroids']) < max_centroids:  # обогащаем голос новым центроидом (cap)
                    rec['centroids'].append(cent.tolist())
                rec['provenance'].append(_prov(episode, cl, air[cl]))
            elif pid is None or air[cl] >= MIN_GUEST_MIN * 60:
                nid = str(reg['next_id'])
                reg['next_id'] += 1
                sp[nid] = {'centroids': [cent.tolist()], 'provenance': [_prov(episode, cl, air[cl])]}
                mapping[cl] = f'Speaker_{nid}'
            else:  # короткий неузнанный кластер при непустом реестре → fold к ближайшему, без записи
                mapping[cl] = f'Speaker_{pid}'
        # атомарная запись + .bak
        if path.exists():
            try:
                shutil.copy2(path, path.with_suffix('.bak'))
            except Exception:
                pass
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=1))
        tmp.replace(path)
        fcntl.flock(lf, fcntl.LOCK_UN)
    return mapping
