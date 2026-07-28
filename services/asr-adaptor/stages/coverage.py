"""Покрытие звука расшифровкой: сколько речи конвейер потерял и где.

Дефект, ради которого написан модуль, невидим глазами: текст перескакивает через произнесённый
кусок, а тайм-коды соседних реплик при этом не рвутся. Замерено на корпусе «Капитанского мостика»
(55 выпусков, 72.6 ч): пропуски в 41 выпуске, 37 минут речи (0.85%), худший одиночный 101.7 с.

Механизмов потери ДВА, и по готовой расшифровке они неразличимы — поэтому мерим оба:
  1. пасс-2 вернул на кусок пусто (тишина в начале, наложение речи, музыка) — текст сшивается без него;
  2. **дыра в покрытии пасса-1**: чанки нарезаются по ЕГО сегментам (`chunking.load_words`), поэтому
     там, где он промолчал, чанка не возникло вообще и пасс-2 этот звук никогда не слышал.

Здесь только арифметика отрезков — ни сети, ни моделей, поэтому всё проверяется тестами.
"""
from __future__ import annotations

import contextlib
import wave

Interval = tuple[float, float]


def wav_duration(path: str) -> float:
    """Длительность wav по заголовку. Конвейер уже держит 16 кГц моно — внешних зависимостей не надо."""
    with contextlib.closing(wave.open(path, 'rb')) as w:
        rate = w.getframerate() or 1
        return w.getnframes() / float(rate)


def merge(intervals) -> list[Interval]:
    """Отрезки → непересекающиеся, по времени. Сегменты ASR соседних кусков умеют налезать друг
    на друга (повтор с паддингом), а двойной счёт завысил бы покрытие."""
    out: list[list[float]] = []
    for a, b in sorted((min(x, y), max(x, y)) for x, y in intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def covered(intervals, start: float = float('-inf'), end: float = float('inf')) -> float:
    """Сколько секунд окна [start, end] покрыто отрезками."""
    return sum(min(b, end) - max(a, start) for a, b in merge(intervals)
               if min(b, end) > max(a, start))


def holes(intervals, start: float, end: float, min_gap: float = 0.0) -> list[Interval]:
    """Непокрытые промежутки ≥ min_gap внутри [start, end].

    Одна функция обслуживает оба механизма: «пасс-1 против всего звука» и «сегменты реплики против
    её окна».
    """
    res, cur = [], start
    for a, b in merge(intervals):
        if b <= start:
            continue
        if a >= end:
            break
        a, b = max(a, start), min(b, end)
        if a > cur and a - cur >= min_gap:
            res.append((round(cur, 2), round(a, 2)))
        cur = max(cur, b)
    if end - cur >= min_gap and end > cur:
        res.append((round(cur, 2), round(end, 2)))
    return res


def turn_windows(turns, audio_sec: float) -> list[Interval]:
    """Окно реплики — от её начала до начала следующей (последняя — до конца звука).

    Конца у реплики нет, и брать его по последнему распознанному сегменту нельзя: так спрячется
    ровно то, что мы ищем, — потеря в хвосте реплики. По этой же конвенции считает выравнивание,
    поэтому окна для обоих строятся здесь.
    """
    out = []
    for i, t in enumerate(turns):
        start = float(t['start'])
        end = float(turns[i + 1]['start']) if i + 1 < len(turns) else max(audio_sec, start)
        out.append((start, max(end, start)))
    return out


def turn_segments(turn) -> list[Interval]:
    """Отрезки, реально распознанные внутри реплики (сегменты пасса-2 всех её чанков)."""
    return [(s['start'], s['end']) for c in turn.get('chunks') or [] for s in c.get('segments') or []]


def summarize(audio_sec: float, pass1_segments, chunks, turns, min_gap: float) -> dict:
    """Блок `coverage` для ответа: где потеряно, сколько и что из этого вернула страховка."""
    p1_holes = holes([(s['start'], s['end']) for s in pass1_segments], 0.0, audio_sec, min_gap)
    recovered = [c for c in chunks if c.get('recovered')]
    empty = [c for c in chunks if not (c.get('raw') or '').strip()]

    # Что пасс-2 не услышал бы БЕЗ страховки: дыры между обычными чанками. Не путать с дырами
    # пасса-1 — те частью лежат ВНУТРИ чанков, а чанк переслушивается целиком.
    unheard = holes([(c['start'], c['end']) for c in chunks if not c.get('recovered')],
                    0.0, audio_sec, min_gap)

    windows = turn_windows(turns, audio_sec)
    if turns:
        tail = turn_segments(turns[-1])
        if tail:  # хвост звука после последнего слова — заставка/музыка, а не потерянная речь
            windows[-1] = (windows[-1][0], max(e for _, e in tail))

    lost = []
    for turn, (a, b) in zip(turns, windows):
        for h0, h1 in holes(turn_segments(turn), a, b, min_gap):
            lost.append({'start': h0, 'sec': round(h1 - h0, 1), 'speaker': turn.get('speaker', '')})
    lost.sort(key=lambda x: -x['sec'])

    return {
        'audio_sec': round(audio_sec, 1),
        'min_gap_s': min_gap,
        'pass1_covered_sec': round(covered([(s['start'], s['end']) for s in pass1_segments],
                                           0.0, audio_sec), 1),
        'pass1_holes': [[a, b] for a, b in p1_holes],
        'pass1_hole_sec': round(sum(b - a for a, b in p1_holes), 1),
        'unheard': [[a, b] for a, b in unheard],
        'unheard_sec': round(sum(b - a for a, b in unheard), 1),
        'recovered_chunks': len(recovered),
        'recovered_sec': round(sum(covered(
            [(s['start'], s['end']) for s in c.get('segments') or []]) for c in recovered), 1),
        'retried_chunks': sum(1 for c in chunks if c.get('retried')),
        'empty_chunks': [[round(c['start'], 1), round(c['end'], 1)] for c in empty],
        'lost_sec': round(sum(x['sec'] for x in lost), 1),
        'lost_spots': lost[:20],
    }
