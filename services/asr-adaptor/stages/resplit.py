"""Голоса по словам: реплика режется там, где по диаризации сменился говорящий.

Зачем. Голос куску пасса-2 назначается целиком — по сегменту пасса-1 (у него нет времён слов) и
после склейки коротких кусков (`chunking._merge_short`). Короткая реплика второго голоса («ага»,
вставка собеседника) поэтому уезжает в реплику первого, и в читалке один человек сам себе
поддакивает. Здесь, в самом конце, у каждого слова уже есть ТОЧНОЕ время (выравнивание MMS_FA),
и голос решается пословно по отрезкам диаризации.

Правила (каждое — против известной ошибки):

- **Реплика, все слова которой одного голоса, не меняется ни на байт.** Однодикторные записи и
  подкаст ничего не замечают.
- **Одиночное слово чужого голоса НЕ сглаживается.** Сглаживание «одно слово внутри чужой серии —
  шум» стирает ровно то, ради чего всё затеяно, — «ага» поверх чужой речи.
- **Наложение:** слово внутри отрезков двух голосов отдаётся КОРОТКОМУ отрезку (перебивающему),
  если тот не длиннее `OVERLAP_SHORT_S`; иначе — голосу с наибольшим перекрытием. Долгий отрезок
  хозяина покрывает поддакивание целиком, и «наибольшее перекрытие» всегда отдавало бы его хозяину.
- Слово без покрытия (пауза в отрезках) — голос реплики: чужой голос нужно ДОКАЗАТЬ отрезком.
- Кусок, отрезанный от реплики, приклеивается к соседней реплике того же голоса: иначе хвост
  «ага» и следующая реплика того же человека стояли бы двумя строками.

Времена слов — формата morag-words-v1 (`[слово, начало, конец]`), слова выравнивания — это
`text.split()` реплики; если их число разошлось, реплика не трогается (журнал говорит об этом).
"""
from __future__ import annotations

import difflib
import re

OVERLAP_SHORT_S = 3.0  # отрезок не длиннее — «перебивающий»: слово в наложении отдаётся ему
# Голос, у которого во всей записи эфира меньше, смену говорящего внутри реплики НЕ доказывает
# (вызывающий отдаёт для него None в `label_of`). Замерено 30.09 на «События, а не БД»: кластер в
# 13 с эфира смешал зрителя и ведущего и отнял у них по слову («Вот, и», «отличное»).
WEAK_AIR_S = 20.0
SNAP_WORDS = 3         # насколько далеко граница голоса тянется к концу предложения
_SENT_END = re.compile(r'[.!?…]["»)\]]*$')

_TOKEN = re.compile(r'[\w-]+', re.UNICODE)


def snap(ids: list, toks: list[str]) -> list:
    """Притянуть границы смены голоса к концу предложения.

    Граница голоса у диаризации дрожит на слово-два, и реплика резалась посреди фразы: «…
    продиагностируем. Ещё | вопрос.» (владелец, 30.09). Граница, которая стоит не после конца
    предложения, переезжает к ближайшему концу предложения в пределах SNAP_WORDS слов.
    ⚠️ Короткий кусок (≤ 2 слов — «ага», «да») граница может только РАСШИРИТЬ и лишь на одно
    слово: иначе поддакивание либо исчезало бы, либо отбирало у соседа полфразы.
    """
    ids = list(ids)
    k = 1
    while k < len(ids):
        if ids[k] == ids[k - 1] or _SENT_END.search(toks[k - 1]):
            k += 1
            continue
        r = k
        while r < len(ids) and ids[r] == ids[k]:
            r += 1
        lft = k - 1
        while lft > 0 and ids[lft - 1] == ids[k - 1]:
            lft -= 1
        short_right, short_left = r - k <= 2, k - lft <= 2
        lo = max(lft + 1, k - (1 if short_right else SNAP_WORDS))
        hi = min(r - 1, k + (1 if short_left else SNAP_WORDS))
        cands = [p for p in range(lo, hi + 1) if p != k and _SENT_END.search(toks[p - 1])
                 and not (p > k and short_right) and not (p < k and short_left)]
        if cands:
            p = min(cands, key=lambda p: (abs(p - k), p))
            if p < k:
                ids[p:k] = [ids[k]] * (k - p)
            else:
                ids[k:p] = [ids[k - 1]] * (p - k)
            k = max(p, k) + 1
        else:
            k += 1
    return ids


def _norm(w: str) -> str:
    m = _TOKEN.findall(w.casefold().replace('ё', 'е'))
    return ''.join(m)


def word_cluster(a: float, b: float, spans) -> str | None:
    """Кластер диаризации для слова [a, b] — по правилам модуля; None — слово вне отрезков."""
    mid = (a + b) / 2.0
    cover = [s for s in spans if float(s['start']) <= mid <= float(s['end'])]
    if not cover:
        ov: dict[str, float] = {}
        for s in spans:
            o = min(b, float(s['end'])) - max(a, float(s['start']))
            if o > 0:
                ov[s['speaker']] = ov.get(s['speaker'], 0.0) + o
        return max(ov, key=ov.get) if ov else None
    if len({s['speaker'] for s in cover}) == 1:
        return cover[0]['speaker']
    shortest = min(cover, key=lambda s: float(s['end']) - float(s['start']))
    if float(shortest['end']) - float(shortest['start']) <= OVERLAP_SHORT_S:
        return shortest['speaker']
    ov = {}
    for s in cover:
        o = min(b, float(s['end'])) - max(a, float(s['start']))
        ov[s['speaker']] = ov.get(s['speaker'], 0.0) + max(o, 0.0)
    return max(ov, key=ov.get)


def _raw_cuts(text_words: list[str], raw: str, cuts: list[int]) -> list[str]:
    """Разрезать `raw` (текст до правок) в местах, соответствующих разрезам `text` по словам."""
    raw_words = raw.split()
    if not raw_words:
        return [''] * (len(cuts) + 1)
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in text_words], b=[_norm(w) for w in raw_words],
                                 autojunk=False)
    # позиция слова text → позиция в raw (монотонно, по совпавшим блокам с интерполяцией)
    pos = [0] * (len(text_words) + 1)
    anchors = [(0, 0)]
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            anchors.append((blk.a + k, blk.b + k))
    anchors.append((len(text_words), len(raw_words)))
    for i in range(len(text_words) + 1):
        lo = max((p for p in anchors if p[0] <= i), key=lambda p: p[0])
        hi = min((p for p in anchors if p[0] >= i), key=lambda p: p[0])
        if hi[0] == lo[0]:
            pos[i] = lo[1]
        else:
            pos[i] = round(lo[1] + (hi[1] - lo[1]) * (i - lo[0]) / (hi[0] - lo[0]))
    edges = [0] + [pos[c] for c in cuts] + [len(raw_words)]
    edges = [max(edges[k - 1] if k else 0, e) for k, e in enumerate(edges)]  # монотонность
    return [' '.join(raw_words[edges[k]:edges[k + 1]]) for k in range(len(edges) - 1)]


def resplit(out_turns: list[dict], word_turns: list[dict], spans,
            label_of, pinned=frozenset()) -> tuple[list[dict], list[dict], dict]:
    """(реплики артефакта, реплики words-документа) → то же, разрезанное по голосам слов.

    `label_of(cluster)` → `(speaker_id, speaker)` — номер голоса и подпись (имя, если есть).
    `pinned` — пары (реплика, слово) БЕЗ своего голоса: берут голос слова слева. Нужно, когда
    голоса переносят на выверенный текст: слово, которое правка человека удаляет (задвоение на
    шве), иначе могло стать отдельной репликой, а пустой реплику правка сделать не может.
    Возвращает новые списки и журнал: сколько реплик разрезано, какие места (для отчёта человеку).
    """
    if not spans or len(out_turns) != len(word_turns):
        return out_turns, word_turns, {}
    pieces: list[tuple[dict, dict, bool]] = []  # (реплика, words-реплика, «рождена разрезом»)
    moments, skipped = [], 0
    # 1. Голос каждого слова. Слово без доказанного голоса (вне отрезков, слабый голос) берёт голос
    # СОСЕДНИХ слов своей реплики — сперва слева, потом справа, — и только если голоса нет ни у
    # одного, голос реплики. ⚠️ Не голос реплики сразу: при переносе на выверенный текст подпись
    # старой реплики и есть ошибка («Вот, и» зрителя уезжало к докладчику — владелец, 30.09).
    usable, per_ids, names = [], [], {}
    for n_turn, (t, wt) in enumerate(zip(out_turns, word_turns)):
        words = wt.get('words') or []
        ok = bool(words) and len(words) == len((t.get('text') or '').split())
        skipped += bool(words) and not ok
        usable.append(ok)
        if not ok:
            per_ids.append([])
            continue
        own = t.get('speaker_id') or t.get('speaker')
        names.setdefault(own, t.get('speaker') or own)
        labs = []
        for w in words:
            cl = word_cluster(float(w[1]), float(w[2]), spans)
            labs.append(label_of(cl) if cl is not None else None)
        for lab in labs:
            if lab:
                names.setdefault(lab[0], lab[1])
        ids = [lab[0] if lab else None for lab in labs]
        for k in range(1, len(ids)):
            ids[k] = ids[k] or ids[k - 1]
        for k in range(len(ids) - 2, -1, -1):
            ids[k] = ids[k] or ids[k + 1]
        ids = [i or own for i in ids]
        for k in range(len(ids)):
            if (n_turn, k) in pinned:
                ids[k] = ids[k - 1] if k else own
        per_ids.append(ids)
    # 2. Границы голосов — к концу предложения, по СПЛОШНОЙ последовательности слов: граница
    # старой реплики ничего не значит («… продиагностируем. Ещё» | «вопрос.» — разные реплики).
    # Реплика с расхождением слов и текста — барьер: притягивание через неё не идёт.
    block: list[int] = []
    for n_turn in list(range(len(out_turns))) + [None]:
        if n_turn is not None and usable[n_turn]:
            block.append(n_turn)
            continue
        if block:
            flat = [i for b in block for i in per_ids[b]]
            toks = [w for b in block for w in out_turns[b]['text'].split()]
            flat = snap(flat, toks)
            # ⚠️ Слова без своего голоса — снова за левым соседом ПОСЛЕ притягивания: иначе граница,
            # притянутая к концу предложения, отрезала удаляемое правкой слово от остальной правки
            # («дьос ну» уцелело в «Демо» — правка не может сделать реплику пустой).
            where = [(b, k) for b in block for k in range(len(per_ids[b]))]
            for g, key in enumerate(where):
                if key in pinned and g:
                    flat[g] = flat[g - 1]
            pos = 0
            for b in block:
                n = len(per_ids[b])
                per_ids[b], pos = flat[pos:pos + n], pos + n
        block = []
    # 3. Разрез реплик по голосам слов.
    for n_turn, (t, wt) in enumerate(zip(out_turns, word_turns)):
        if not usable[n_turn]:
            pieces.append((t, wt, False))
            continue
        words = wt.get('words') or []
        text_words = (t.get('text') or '').split()
        own = t.get('speaker_id') or t.get('speaker')
        ids = per_ids[n_turn]
        labels = [(i, names.get(i, i)) for i in ids]
        if all(i == own for i in ids):
            pieces.append((t, wt, False))
            continue
        runs: list[list[int]] = []
        for k, i in enumerate(ids):
            if runs and ids[runs[-1][0]] == i:
                runs[-1].append(k)
            else:
                runs.append([k])
        cuts = [r[0] for r in runs[1:]]
        raws = _raw_cuts(text_words, t.get('raw') or '', cuts)
        starts = [float(words[r[0]][1]) for r in runs]
        seg_of: list[list[dict]] = [[] for _ in runs]
        for s in t.get('segments') or ():
            mid = (float(s['start']) + float(s['end'])) / 2
            seg_of[max([k for k, a0 in enumerate(starts) if a0 <= mid] or [0])].append(s)
        for n, (run, raw) in enumerate(zip(runs, raws)):
            sid = ids[run[0]]
            name = next((lab[1] for k in run if (lab := labels[k]) and lab[0] == sid), None)
            if sid == own:
                name = t.get('speaker')
            ws = [words[k] for k in run]
            a = t['start'] if n == 0 else round(float(ws[0][1]), 1)
            b = t['end'] if n == len(runs) - 1 else round(float(ws[-1][2]), 2)
            piece = dict(t)
            piece.update({'speaker': name or sid, 'speaker_id': sid, 'start': a, 'end': b,
                          'text': ' '.join(text_words[k] for k in run), 'raw': raw,
                          'segments': seg_of[n]})
            wpiece = dict(wt)
            wpiece.update({'start': float(ws[0][1]) if n else wt['start'],
                           'end': float(ws[-1][2]) if n < len(runs) - 1 else wt['end'],
                           'speaker': piece['speaker'], 'words': ws})
            pieces.append((piece, wpiece, True))
            if sid != own:
                moments.append({'at': round(float(ws[0][1]), 1), 'from': own, 'to': sid,
                                'words': len(ws), 'text': piece['text'][:80]})
    # кусок, рождённый разрезом, приклеить к соседу того же голоса
    merged: list[tuple[dict, dict, bool]] = []
    for p in pieces:
        if merged and merged[-1][0]['speaker_id'] == p[0]['speaker_id'] and (merged[-1][2] or p[2]):
            t0, w0, _ = merged[-1]
            t1, w1, _ = p
            t0 = dict(t0, end=t1['end'], text=f"{t0['text']} {t1['text']}".strip(),
                      raw=f"{t0.get('raw') or ''} {t1.get('raw') or ''}".strip(),
                      segments=list(t0.get('segments') or ()) + list(t1.get('segments') or ()))
            w0 = dict(w0, end=w1['end'], words=list(w0['words']) + list(w1['words']))
            merged[-1] = (t0, w0, True)
        else:
            merged.append(p)
    log = {}
    if moments or skipped:
        log = {'n_moved': len(moments), 'n_turns_before': len(out_turns),
               'n_turns_after': len(merged), 'moments': moments[:500],
               **({'skipped_mismatch': skipped} if skipped else {})}
    return [m[0] for m in merged], [m[1] for m in merged], log
