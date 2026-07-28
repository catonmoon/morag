"""Диаризация-первичный чанкер (порт adventures/podlodka-asr/pass2_chunk.py на объекты, не файлы).

Надёжность границ: смена спикера (диаризация) > крупная пауза (word-gap) > пунктуация Whisper.
Чанк ≤MAX_S, НИКОГДА не пересекает турн спикера; разрыв по наибольшей паузе, по слову.
Вход: words (пасс-1 сегменты/слова) + spans (pyannote [{start,end,speaker}]). Выход: [{start,end,speaker,text}].
"""
from __future__ import annotations

import math
from collections import defaultdict

MAX_S = 28.0
SHORT_S = 2.0  # короче — бэкчэннел/обрывок; клеим к соседу, если влезает в MAX_S


def load_words(words):
    """words = пасс-1 (dict с 'segments' или список сегментов) → [(text,start,end)] пословно либо
    по сегментам (fallback, когда нет word_timestamps)."""
    segs = words['segments'] if isinstance(words, dict) else words
    out = []
    for s in segs:
        ws = s.get('words') or []
        if ws:
            for w in ws:
                t = (w.get('word') or '').strip()
                if t and w.get('start') is not None and w.get('end') is not None:
                    out.append((t, float(w['start']), float(w['end'])))
        else:
            t = (s.get('text') or '').strip()
            if t and s.get('start') is not None and s.get('end') is not None:
                out.append((t, float(s['start']), float(s['end'])))
    return out


def speaker_of(t0, t1, spans):
    """Спикер по максимальному перекрытию span'ов."""
    ov = defaultdict(float)
    for s in spans:
        o = max(0.0, min(s['end'], t1) - max(s['start'], t0))
        if o:
            ov[s['speaker']] += o
    return max(ov.items(), key=lambda kv: kv[1])[0] if ov else None


def _split_turn(words, max_s):
    """words: [(text,start,end)] одного спикера → чанки ≤max_s, разрез по НАИБОЛЬШЕЙ паузе, по слову."""
    chunks, i, n = [], 0, len(words)
    while i < n:
        start = words[i][1]
        j = i
        while j + 1 < n and words[j + 1][2] - start <= max_s:
            j += 1
        if j + 1 < n:
            best_k, best_gap = j, -1.0
            for k in range(i, j):
                gap = words[k + 1][1] - words[k][2]
                if gap > best_gap:
                    best_gap, best_k = gap, k
            cut = best_k
        else:
            cut = j
        chunks.append((words[i][1], words[cut][2], ' '.join(w[0] for w in words[i:cut + 1])))
        i = cut + 1
    return chunks


def _merge_short(chunks, max_s=MAX_S, short=SHORT_S):
    """Короткий чанк (бэкчэннел) клеим к соседу при итоге ≤max_s. Проход1 — к пред., проход2 — к след."""
    p1 = []
    for c in chunks:
        if p1 and (c['end'] - c['start']) < short and (c['end'] - p1[-1]['start']) <= max_s:
            p1[-1]['end'] = c['end']
            p1[-1]['text'] = (p1[-1]['text'] + ' ' + c['text']).strip()
        else:
            p1.append(dict(c))
    res, i = [], 0
    while i < len(p1):
        c = p1[i]
        if (c['end'] - c['start']) < short and i + 1 < len(p1) \
                and (p1[i + 1]['end'] - c['start']) <= max_s:
            nxt = p1[i + 1]
            nxt['start'] = c['start']
            nxt['text'] = (c['text'] + ' ' + nxt['text']).strip()
            i += 1
        else:
            res.append(c); i += 1
    return res


def chunk(words, spans, max_s: float = MAX_S):
    """words (пасс-1) + spans (диаризация) → [{start,end,speaker,text}], single-speaker, ≤max_s."""
    ws = load_words(words)
    raw = [speaker_of(a, b, spans) for _, a, b in ws]
    # forward/back-fill: слово без span'а наследует ближайшего известного спикера (не плодим 'unknown')
    last = None
    for k in range(len(raw)):
        if raw[k] is None:
            raw[k] = last
        else:
            last = raw[k]
    nxt = None
    for k in range(len(raw) - 1, -1, -1):
        if raw[k] is None:
            raw[k] = nxt
        else:
            nxt = raw[k]
    tagged = [(t, a, b, raw[i] or 'unknown') for i, (t, a, b) in enumerate(ws)]
    out, cur, cur_spk = [], [], None
    for t, a, b, spk in tagged:
        if spk != cur_spk and cur:
            for s, e, txt in _split_turn(cur, max_s):
                out.append({'start': s, 'end': e, 'speaker': cur_spk, 'text': txt})
            cur = []
        cur_spk = spk
        cur.append((t, a, b))
    if cur:
        for s, e, txt in _split_turn(cur, max_s):
            out.append({'start': s, 'end': e, 'speaker': cur_spk, 'text': txt})
    return _merge_short(out, max_s)


def gap_chunks(holes, spans, max_s: float = MAX_S) -> list[dict]:
    """Чанки на промежутки, которые пасс-1 не покрыл своими сегментами.

    Нарезка идёт ПО НЕМУ, поэтому там, где он промолчал, чанка не возникает — и пасс-2 этот звук
    никогда не услышит, сколько бы речи в нём ни было. Здесь такой промежуток режется на куски
    ≤max_s и отдаётся пассу-2 наравне с остальными; текста от пасса-1 у них нет (`text: ''`).
    Помечены `recovered`, чтобы их вклад был виден в отчёте о покрытии.

    **Добираем только то, где диаризация слышит речь.** Заставка, музыка и тишина — тоже
    непокрытые промежутки, а Whisper на них склонен фантазировать: без фильтра страховка от потери
    речи сама дописывала бы в транскрипт то, чего не звучало. Диаризация уже посчитана, и это её
    прямая компетенция. Если спанов нет вовсе (диаризация не отработала), фильтровать нечем —
    добираем промежуток целиком.
    """
    out = []
    for a, b in holes:
        for s0, e0 in (_speech_parts(a, b, spans) if spans else [(a, b)]):
            n = max(1, math.ceil((e0 - s0) / max_s))
            step = (e0 - s0) / n
            for k in range(n):
                s, e = s0 + k * step, s0 + (k + 1) * step
                out.append({'start': s, 'end': e, 'speaker': speaker_of(s, e, spans) or 'unknown',
                            'text': '', 'recovered': True})
    return out


def _speech_parts(a: float, b: float, spans) -> list[tuple[float, float]]:
    """Куски [a, b], которые диаризация считает речью (объединённые спаны, по времени)."""
    parts: list[list[float]] = []
    for s in sorted(spans, key=lambda x: x['start']):
        s0, e0 = max(a, float(s['start'])), min(b, float(s['end']))
        if e0 <= s0:
            continue
        if parts and s0 <= parts[-1][1]:
            parts[-1][1] = max(parts[-1][1], e0)
        else:
            parts.append([s0, e0])
    return [(s, e) for s, e in parts]
