"""Разрез длинного звука на части для модели без длинного режима и сборка сегментов из слов.

Чистый numpy (+ razdel для границ предложений), без torch и mlx: тестируется без весов и переиспользуется
другим бэкендом (CUDA, Windows) — контракт ответа один (`verbose_json`: `segments[].words = [{word, start, end}]`, как у mlx-whisper).

Зачем. GigaAM-v3 берёт до 25 с звука за вызов, а длинный режим пакета требует pyannote. Для черновика
записи целиком режем сами: граница части — САМАЯ ТИХАЯ точка окна [20, 24] с (RMS по кадрам 20 мс), а не
сетка времени, чтобы слово не рвалось на шве. Замер 16.10 на пяти записях (185 мин, 507 частей): разрезов
в речи — 10, все на самой плотной записи; слова с временами (RNN-T, покадрово) — длительность медиана 0.2 с,
паузы ≥ 0.6 с — одна на 11 слов, то есть сегментам есть где кончаться.
"""
from __future__ import annotations

import numpy as np

SENTENCE_END = ('.', '!', '?', '…')


def quiet_cuts(audio: np.ndarray, sr: int, lo: float = 20.0, hi: float = 24.0,
               frame: float = 0.02) -> list[tuple[float, float]]:
    """Границы частей `[(start_s, end_s)]`, каждая ≤ `hi` с: разрез — минимум RMS в окне [pos+lo, pos+hi].

    Звук короче `hi` — одна часть. Хвост короче `lo` остаётся отдельной частью (модель берёт и короткое)."""
    total = len(audio) / sr
    if total <= hi:
        return [(0.0, total)]
    n = max(1, int(frame * sr))
    nfr = len(audio) // n
    rms = np.sqrt(np.mean(audio[:nfr * n].astype(np.float32).reshape(nfr, n) ** 2, axis=1) + 1e-12)
    out: list[tuple[float, float]] = []
    pos = 0.0
    while total - pos > hi:
        a, b = int((pos + lo) / frame), int((pos + hi) / frame)
        b = min(b, nfr)
        k = a + int(np.argmin(rms[a:b])) if b > a else b
        cut = k * frame
        out.append((pos, cut))
        pos = cut
    out.append((pos, total))
    return out


def sentence_ends(words: list[dict]) -> set[int]:
    """Индексы слов, на которых кончается предложение — по razdel (правило владельца 17.10: границы предложений —
    библиотекой, не точкой: «т.е.», инициалы, числа — не конец). Нет razdel — запасное правило по знаку конца."""
    texts = [str(w.get('word') or '').strip() for w in words]
    try:
        from razdel import sentenize
    except ImportError:                       # запасной ход: знак конца предложения
        return {i for i, t in enumerate(texts) if t.endswith(SENTENCE_END)}
    joined = ' '.join(texts)
    # Конец каждого слова в символах склеенного текста → индекс слова по концу предложения razdel.
    ends: list[int] = []
    pos = 0
    for t in texts:
        pos += len(t)
        ends.append(pos)
        pos += 1
    out: set[int] = set()
    for sent in sentenize(joined):
        stop = sent.stop
        # слово, чей конец ближе всего к концу предложения (razdel не рвёт слова, разница — хвостовые пробелы)
        k = min(range(len(ends)), key=lambda i: abs(ends[i] - stop)) if ends else None
        if k is not None:
            out.add(k)
    return out


def words_to_segments(words: list[dict], pause: float = 0.6) -> list[dict]:
    """Слова `{word, start, end}` (уже в абсолютном времени) → сегменты `{id, start, end, text, words}`.

    Граница сегмента — конец предложения по razdel (`sentence_ends`) или пауза ≥ `pause` с перед следующим словом.
    Разрез на куски читает `words` сегмента и режет по паузам между словами сам — сегменты здесь нужны ради
    `start`/`end`/`text` для глоссария, ленты и покрытия."""
    segs: list[dict] = []
    cur: list[dict] = []
    ends = sentence_ends(words)

    def flush() -> None:
        if cur:
            segs.append({'id': len(segs), 'start': float(cur[0]['start']), 'end': float(cur[-1]['end']),
                         'text': ' '.join(w['word'].strip() for w in cur), 'words': list(cur)})
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if i in ends or (nxt is not None and float(nxt['start']) - float(w['end']) >= pause):
            flush()
    flush()
    return segs


def verbose_json(words: list[dict], language: str, duration: float, text: str | None = None) -> dict:
    """Ответ бэкенда той же формы, что у mlx-whisper (`verbose_json`), по словам с временами."""
    segs = words_to_segments(words)
    return {'task': 'transcribe', 'language': language, 'duration': round(float(duration), 3),
            'text': (text if text is not None else ' '.join(s['text'] for s in segs)).strip(),
            'segments': segs}
