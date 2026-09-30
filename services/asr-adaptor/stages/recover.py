"""Склеенный диалог: диаризация дала ОДИН голос там, где людей двое.

Случай: два ведущих в быстром диалоге, голоса похожие — pyannote сводит их в один кластер, и
расшифровка выходит монологом, в котором человек сам себя перебивает. Перерезка по словам
(`resplit`) тут бессильна: ей не на что опереться, отрезков второго голоса нет.

Ход (идея — «восстановление по эмбеддингам фраз» из VoiceStudio): фразы пасса-2 (сегменты от
`MIN_PHRASE_S`) получают по вектору CAM++, векторы делятся на ДВА кластера, и результат берётся,
только если голоса действительно разные:

- разделение = средний косинус внутри кластеров − средний между ними ≥ `MIN_SEPARATION`
  (у одного человека фразы расходятся шумом, и деление на двое даёт разделение около нуля);
- в каждом кластере не меньше `MIN_PHRASES` фраз и `MIN_AIR_S` секунд.

Иначе ничего не меняется. Узел срабатывает только когда ЗНАНИЕ О ЗАПИСИ говорит «людей больше
одного» (имена в подсказках или `ASR_MIN_SPEAKERS ≥ 2`): делить каждый монолог на двое ради
проверки — значит однажды разрезать одного докладчика.
"""
from __future__ import annotations

import numpy as np

MIN_PHRASE_S = 0.75
MIN_PHRASES = 2
MIN_AIR_S = 1.5
MIN_SEPARATION = 0.12
SUBSTANTIAL_S = 30.0   # кластер с таким эфиром считается «голосом записи»


def lone_voice(spans) -> str | None:
    """Если у диаризации ровно один голос с заметным эфиром — его метка, иначе None."""
    air: dict[str, float] = {}
    for s in spans or ():
        air[s['speaker']] = air.get(s['speaker'], 0.0) + float(s['end']) - float(s['start'])
    big = [k for k, v in air.items() if v >= SUBSTANTIAL_S]
    return big[0] if len(big) == 1 else None


def phrases(chunks) -> list[dict]:
    """Фразы для голосов: сегменты пасса-2 не короче MIN_PHRASE_S, с текстом."""
    out = []
    for c in chunks:
        for s in c.get('segments') or ():
            a, b = float(s['start']), float(s['end'])
            if b - a >= MIN_PHRASE_S and (s.get('text') or '').strip():
                out.append({'start': a, 'end': b})
    return out


def _two_means(x: np.ndarray, iters: int = 30) -> np.ndarray:
    """Сферический k-means на двоих; старт — самая далёкая пара (детерминированно)."""
    sim = x @ x.T
    i, j = np.unravel_index(np.argmin(sim), sim.shape)
    c = np.stack([x[i], x[j]])
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, lab) and _:
            break
        lab = new
        for k in (0, 1):
            m = x[lab == k].mean(axis=0) if (lab == k).any() else c[k]
            c[k] = m / (np.linalg.norm(m) + 1e-9)
    return lab


def split_two(vectors, durations) -> tuple[list[int] | None, dict]:
    """Векторы фраз → метки 0/1 или None (не доказано), плюс замер для журнала."""
    keep = [k for k, v in enumerate(vectors) if v is not None]
    info = {'phrases': len(vectors), 'usable': len(keep)}
    if len(keep) < 2 * MIN_PHRASES:
        return None, {**info, 'why': 'мало фраз'}
    x = np.asarray([vectors[k] for k in keep], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    lab = _two_means(x)
    dur = np.asarray([durations[k] for k in keep])
    counts = [int((lab == k).sum()) for k in (0, 1)]
    airs = [float(dur[lab == k].sum()) for k in (0, 1)]
    info.update(counts=counts, air=[round(a, 1) for a in airs])
    if min(counts) < MIN_PHRASES or min(airs) < MIN_AIR_S:
        return None, {**info, 'why': 'второй голос слишком мал'}
    sim = x @ x.T
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(x), 1)
    within, cross = sim[iu][same[iu]], sim[iu][~same[iu]]
    sep = float(within.mean() - cross.mean()) if len(within) and len(cross) else 0.0
    info['separation'] = round(sep, 3)
    if sep < MIN_SEPARATION:
        return None, {**info, 'why': 'голоса не разошлись'}
    out = [-1] * len(vectors)
    for k, l in zip(keep, lab.tolist()):
        out[k] = l
    return out, info


def relabel(chunks, spans, phr, labels, lone: str) -> list[dict]:
    """Новые отрезки диаризации по фразам и новый голос у кусков пасса-2.

    Больший по эфиру кластер сохраняет прежнюю метку (`lone`), второй получает `<lone>_B`.
    Кусок получает голос большинства своих фраз; фразы без метки голоса не меняют.
    """
    air = [0.0, 0.0]
    for p, l in zip(phr, labels):
        if l >= 0:
            air[l] += p['end'] - p['start']
    main = 0 if air[0] >= air[1] else 1
    name = {main: lone, 1 - main: f'{lone}_B'}
    new_spans = [{'start': p['start'], 'end': p['end'], 'speaker': name[l]}
                 for p, l in zip(phr, labels) if l >= 0]
    for c in chunks:
        vote: dict[str, float] = {}
        for sp in new_spans:
            o = min(float(c['end']), sp['end']) - max(float(c['start']), sp['start'])
            if o > 0:
                vote[sp['speaker']] = vote.get(sp['speaker'], 0.0) + o
        if vote:
            c['speaker'] = max(vote, key=vote.get)
    # Прочие кластеры (короткие) остаются как были: их отрезки не пересекаются с фразами главного.
    rest = [s for s in spans or () if s['speaker'] != lone]
    return sorted(new_spans + rest, key=lambda s: s['start'])
