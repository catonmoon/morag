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
- **Наложение:** слово внутри отрезков двух голосов — голосу с наибольшим перекрытием (хозяину).
  Короткому (перебивающему) отрезку оно отдаётся, только если САМО слово — поддакивание из списка
  (`backchannel`, задаёт профиль). ⚠️ Прежнее «всегда короткому» было неверно (владелец, 01.10):
  кто-то меньше секунды сказал «да» поверх лектора, whisper записал слово ЛЕКТОРА («мне,»), и
  правило отдало его перебившему. ASR пишет того, кто громче и дольше.
- Слово без покрытия (пауза в отрезках) — голос соседних слов: чужой голос нужно ДОКАЗАТЬ отрезком.
- **Второй свидетель — голос** (`witness`): кусок, который перерезка отдаёт другому голосу,
  сверяется вектором CAM++ с голосами соседей. Звучит как сосед — остаётся соседу. Кусок короче
  `WITNESS_MIN_S` голосом не проверить — он у соседа, если это не поддакивание. Без этого мелкие
  кластеры pyannote (20–60 с) забирали целые фразы лектора — «у каждого рода правки свой свидетель».
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
WITNESS_MIN_S = 1.0    # кусок короче голосом не проверить: остаётся соседу (кроме поддакивания)
CENTROID_MIN_S = 2.0   # куски голоса от этой длины идут в его центроид для свидетеля
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


def word_cluster(a: float, b: float, spans, backchannel: bool = False) -> str | None:
    """Кластер диаризации для слова [a, b] — по правилам модуля; None — слово вне отрезков.

    `backchannel` — само слово поддакивание: в наложении оно отдаётся короткому отрезку.
    """
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
    if backchannel and float(shortest['end']) - float(shortest['start']) <= OVERLAP_SHORT_S:
        return shortest['speaker']
    # наибольшее перекрытие; ничья (слово целиком внутри обоих) — долгому отрезку, хозяину
    best: dict[str, tuple[float, float]] = {}
    for s in cover:
        o = max(min(b, float(s['end'])) - max(a, float(s['start'])), 0.0)
        ln = float(s['end']) - float(s['start'])
        prev = best.get(s['speaker'], (0.0, 0.0))
        best[s['speaker']] = (prev[0] + o, max(prev[1], ln))
    return max(best, key=lambda k: best[k])


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


SAME_AS = 0.5          # cos с голосом соседа, от которого кусок «звучит как сосед» (своего центроида нет)
MAX_CENTROID_RUNS = 20


def _cos(a, b) -> float:
    na = sum(x * x for x in a) ** 0.5 or 1e-9
    nb = sum(x * x for x in b) ** 0.5 or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _mean(vs):
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    return [sum(c) / len(vs) for c in zip(*vs)]


def witnessed(ids: list, toks: list[str], times: list, embed, backchannel=frozenset(),
              log: list | None = None) -> list:
    """Проверить каждую смену голоса внутри последовательности вторым свидетелем — голосом.

    Кусок (серия слов одного голоса) с соседями другого голоса: короче WITNESS_MIN_S — отдаётся
    более длинному соседу (кроме поддакивания); длиннее — сверяется вектором с голосами соседей
    (центроид — по длинным кускам того же голоса В ЭТОЙ ЖЕ последовательности, БЕЗ проверяемого
    куска: у мелкого кластера иначе «свой» голос — это сам кусок, и он всегда выигрывал бы).
    Звучит как сосед не хуже, чем как «свой» голос, — отдаётся соседу. `embed([(t0, t1)])` →
    векторы (или None); без него — только правило коротких кусков.
    """
    ids = list(ids)

    def runs_of(ids):
        out = []
        for k, i in enumerate(ids):
            if out and out[-1][2] == i:
                out[-1][1] = k + 1
            else:
                out.append([k, k + 1, i])
        return out

    def span(r):
        return float(times[r[0]][0]), float(times[r[1] - 1][1])

    def dur(r):
        a, b = span(r)
        return b - a

    def neighbours(runs, j):
        return {runs[j - 1][2] if j else None, runs[j + 1][2] if j + 1 < len(runs) else None} - {runs[j][2], None}

    # ⚠️ Порядок: СНАЧАЛА голос по длинным кускам, потом — пересчёт кусков, и только потом правило
    # коротких. Решения по одной исходной раскладке противоречили друг другу: короткий кусок лектора
    # между двумя кусками мелкого кластера уходил к нему, а свидетель тут же возвращал соседей лектору.
    runs = runs_of(ids)
    queries = [(j, neighbours(runs, j)) for j, r in enumerate(runs)
               if neighbours(runs, j) and dur(r) >= WITNESS_MIN_S]
    if queries and embed is not None:
        ids = _by_voice(ids, runs, queries, span, dur, embed, log)
    runs = runs_of(ids)
    for j, r in enumerate(runs):
        if not neighbours(runs, j) or dur(r) >= WITNESS_MIN_S:
            continue
        # Короткий кусок голосом не проверить. Отдаём его соседу, только если он ЗАЖАТ между кусками
        # одного и того же голоса («некая фишка.» внутри речи лектора): на стыке настоящего диалога
        # (A, потом B) короткая реплика остаётся за диаризацией.
        if backchannel and all(_norm(t) in backchannel for t in toks[r[0]:r[1]]):
            continue
        left = runs[j - 1][2] if j else None
        right = runs[j + 1][2] if j + 1 < len(runs) else None
        if left is None or left != right:
            continue
        if log is not None:
            log.append({'at': round(span(r)[0], 1), 'was': r[2], 'now': left, 'why': 'короткий'})
        ids[r[0]:r[1]] = [left] * (r[1] - r[0])
    return ids


def _by_voice(ids, runs, queries, span, dur, embed, log):
    """Свидетель-голос для длинных кусков (см. `witnessed`)."""
    ids = list(ids)
    # векторы: проверяемые куски + длинные куски голосов, участвующих в проверках
    voices = {runs[j][2] for j, _ in queries} | {v for _, c in queries for v in c}
    long_runs = {}
    for j, r in enumerate(runs):
        if r[2] in voices and dur(r) >= CENTROID_MIN_S:
            long_runs.setdefault(r[2], []).append(j)
    for v in long_runs:
        long_runs[v] = sorted(long_runs[v], key=lambda j: -dur(runs[j]))[:MAX_CENTROID_RUNS]
    need = sorted({j for j, _ in queries} | {j for js in long_runs.values() for j in js})
    vecs = dict(zip(need, embed([span(runs[j]) for j in need])))

    def centroid(v, without):
        return _mean([vecs.get(j) for j in long_runs.get(v, []) if j != without])

    for j, cand in queries:
        r, q = runs[j], vecs.get(j)
        if q is None:
            continue
        own_c = centroid(r[2], j)
        own = _cos(q, own_c) if own_c is not None else None
        sc = {v: _cos(q, c) for v in cand if (c := centroid(v, j)) is not None}
        if not sc:
            continue
        best = max(sc, key=sc.get)
        move = (own is None and sc[best] >= SAME_AS) or (own is not None and sc[best] >= own)
        if log is not None:
            log.append({'at': round(span(r)[0], 1), 'was': r[2], 'now': best if move else r[2],
                        'why': 'голос' if move else 'голос: оставлен', 'neighbour': best,
                        'cos_own': None if own is None else round(own, 3),
                        'cos_neighbour': round(sc[best], 3), 'sec': round(dur(r), 1)})
        if move:
            ids[r[0]:r[1]] = [best] * (r[1] - r[0])
    return ids


BIG_AIR_S = 120.0      # чужие голоса ищем только внутри голоса с таким эфиром в записи
FOREIGN_COS = 0.5      # предложение дальше этого от центроида голоса — чужое
FOREIGN_SAME = 0.6     # чужие отрезки ближе этого друг к другу — один и тот же человек
FOREIGN_CORE = 0.6     # второй проход центроида — по предложениям не дальше этого от первого
RETURN_MARGIN = 0.05   # к другому большому голосу — если ближе к нему, чем к своему, на столько
SHORT_VOICE_S = 0.8    # короткое рядом с чужим отрезком от этой длины решает голос, а не положение
SENT_MIN_S = 2.0       # предложение короче голосом не судим (идёт за соседним длинным)
SENT_MIN_RATE = 1.0    # слов в секунду: реже — почти тишина, вектор врёт («Мне всё.» на 5.8 с)


def foreign_voices(word_turns: list[dict], embed, new_label) -> tuple[list, list]:
    """Чужие голоса ВНУТРИ большого голоса записи: вопросы из зала, склеенные с докладчиком.

    Диаризатор кладёт короткие вопросы из зала в кластер докладчика (замерено 01.10, «События, а
    не БД»: три зрителя в голосе докладчика). Каждое длинное предложение голоса — вектор CAM++;
    центроид — по всем ним; предложения дальше FOREIGN_COS — чужие (там: зрители 0.19–0.41, сам
    докладчик ≥ 0.60, медиана 0.91 — зазор чистый). Короткие предложения идут за следующим длинным
    той же реплики (приветствие «Всем привет, …» — со своим вопросом, «Да, хороший вопрос.» после
    него — с ответом). Чужие отрезки группируются по похожести (FOREIGN_SAME) — разные зрители
    разные люди. Возвращает [(реплика, слово_с, слово_по, метка)] и журнал. `new_label(n)` — метка
    n-го найденного голоса (номер выдаёт реестр потом).
    """
    by_voice: dict[str, list] = {}
    air: dict[str, float] = {}
    for n, t in enumerate(word_turns):
        ws = t.get('words') or []
        if not ws:
            continue
        v = t.get('speaker_id') or t.get('speaker')
        air[v] = air.get(v, 0.0) + float(ws[-1][2]) - float(ws[0][1])
        cur = []
        for k, w in enumerate(ws):
            cur.append(k)
            if _SENT_END.search(str(w[0])) or k == len(ws) - 1:
                a, b = float(ws[cur[0]][1]), float(ws[cur[-1]][2])
                by_voice.setdefault(v, []).append({'turn': n, 'i0': cur[0], 'i1': cur[-1] + 1,
                                                   't0': a, 't1': b, 'nw': len(cur)})
                cur = []
    # 1. Векторы длинных предложений ВСЕХ голосов записи (короткие и почти тишина не судятся).
    allsents = [s | {'voice': v} for v, ss in by_voice.items() for s in ss]
    judged = [s for s in allsents if s['t1'] - s['t0'] >= SENT_MIN_S
              and s['nw'] / (s['t1'] - s['t0']) >= SENT_MIN_RATE]
    if not judged:
        return [], []
    for s, vec in zip(judged, embed([(s['t0'], s['t1']) for s in judged])):
        s['vec'] = vec
    # 2. Центроиды БОЛЬШИХ голосов — в два прохода (первый «размазан» чужими: вопрос Ольги на 60 с
    # не дотягивал до порога, пока центроид не очистили — замерено 01.10).
    core: dict[str, list] = {}
    for v in by_voice:
        if air.get(v, 0.0) < BIG_AIR_S:
            continue
        vs = [s['vec'] for s in judged if s['voice'] == v and s.get('vec') is not None]
        if len(vs) < 5:
            continue
        cen = _mean(vs)
        tight = [x for x in vs if _cos(x, cen) >= FOREIGN_CORE]
        core[v] = _mean(tight) if len(tight) >= 5 else cen
    if not core:
        return [], []
    # 3. Куда предложение: свой большой голос — остаётся; звучит как ДРУГОЙ большой голос (от
    # FOREIGN_CORE и ближе своего) — к нему (речь докладчика в мелком кластере зрителя: ответ Даши
    # под меткой Ольги); в большом голосе и ни на кого не похоже — чужой голос.
    for s in judged:
        if s.get('vec') is None:
            continue
        own = _cos(s['vec'], core[s['voice']]) if s['voice'] in core else None
        others = {v: _cos(s['vec'], c) for v, c in core.items() if v != s['voice']}
        best = max(others, key=others.get) if others else None
        s['cos'] = own
        if best is not None and others[best] >= FOREIGN_CORE and \
                others[best] >= (own if own is not None else -1.0) + RETURN_MARGIN:
            # ⚠️ Сравнение, а не порог «своего»: центроид мелкого голоса бывает смесью (у Ольги в
            # кластере сидели и ответы Даши), и ответ Даши похож на смесь не меньше 0.5.
            s['dest'] = best
            s['cos'] = others[best]
        elif own is None or own >= FOREIGN_COS:
            s['dest'] = None
        else:
            s['dest'] = '?'          # чужой голос
    # 4а. Короткое предложение РЯДОМ с чужим отрезком решает голос: по положению не различить хвост
    # вопроса («Вот именно, которые в K8s.») и начало ответа («Какой сложный вопрос!») — замерено
    # 01.10, правило «за соседом» в любую сторону ломало одно из двух. Ближе к соседнему чужому
    # предложению, чем к своему голосу, — уходит с ним.
    cand = []
    for n in {s['turn'] for s in allsents}:
        row = sorted((s for s in allsents if s['turn'] == n), key=lambda s: s['i0'])
        for k, s in enumerate(row):
            if 'dest' in s or s['t1'] - s['t0'] < SHORT_VOICE_S or s['voice'] not in core:
                continue
            nb = [x for x in (row[k - 1] if k else None, row[k + 1] if k + 1 < len(row) else None)
                  if x is not None and x.get('dest') is not None and x.get('vec') is not None]
            if nb:
                cand.append((s, nb[0]))
    if cand:
        for (s, nb), vec in zip(cand, embed([(s['t0'], s['t1']) for s, _ in cand])):
            if vec is None:
                continue
            own, other = _cos(vec, core[s['voice']]), _cos(vec, nb['vec'])
            s['vec'], s['dest'], s['cos'] = vec, (nb['dest'] if other > own else None), max(own, other)
    # 4. Короткие — за следующим судимым своей реплики (нет его — остаются): «Всем привет, …» идёт
    # со своим вопросом, «Да, хороший вопрос.» после него — с ответом.
    for n in {s['turn'] for s in allsents}:
        row = sorted((s for s in allsents if s['turn'] == n), key=lambda s: s['i0'])
        nxt = None
        for s in reversed(row):
            if 'dest' in s:
                nxt = s['dest']
            else:
                s['dest_inh'] = nxt
        # ⚠️ Назад не наследуем: короткое в конце реплики («Какой сложный вопрос!» после вопроса
        # зрителя) — уже ответ, а не хвост вопроса (владелец, 01.10).
    # 5. Отрезки: подряд идущие предложения одной реплики с одним назначением.
    regions = []
    for s in sorted(allsents, key=lambda s: (s['turn'], s['i0'])):
        dest = s.get('dest', s.get('dest_inh'))
        if dest is None:
            continue
        r = regions[-1] if regions else None
        if r and r['turn'] == s['turn'] and r['i1'] == s['i0'] and r['dest'] == dest:
            r['i1'], r['t1'] = s['i1'], s['t1']
            r['sents'].append(s)
        else:
            regions.append({'turn': s['turn'], 'i0': s['i0'], 'i1': s['i1'], 't0': s['t0'],
                            't1': s['t1'], 'dest': dest, 'voice': s['voice'], 'sents': [s]})
    out, log, groups = [], [], []
    for r in regions:
        judged_here = [s for s in r['sents'] if s.get('vec') is not None and 'dest' in s]
        if not judged_here:
            continue
        label = r['dest']
        if label == '?':
            vec = _mean([s['vec'] for s in judged_here])
            g = next((g for g in groups if _cos(vec, g['vec']) >= FOREIGN_SAME), None)
            if g is None:
                g = {'vec': vec, 'label': new_label(len(groups))}
                groups.append(g)
            label = g['label']
        out.append((r['turn'], r['i0'], r['i1'], label))
        log.append({'at': round(r['t0'], 1), 'to': round(r['t1'], 1), 'from': r['voice'], 'label': label,
                    'kind': 'чужой' if r['dest'] == '?' else 'вернул большому',
                    'cos': round(min(s['cos'] for s in judged_here if s.get('cos') is not None), 3)})
    return out, log


def resplit(out_turns: list[dict], word_turns: list[dict], spans,
            label_of, pinned=frozenset(), embed=None,
            backchannel=frozenset()) -> tuple[list[dict], list[dict], dict]:
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
    witness_log: list[dict] = []
    backchannel = frozenset(_norm(w) for w in backchannel)
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
            cl = word_cluster(float(w[1]), float(w[2]), spans,
                              backchannel=bool(backchannel) and _norm(str(w[0])) in backchannel)
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
            times = [(w[1], w[2]) for b in block for w in word_turns[b]['words']]
            flat = witnessed(flat, toks, times, embed, backchannel, witness_log)
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
    if moments or skipped or witness_log:
        log = {'n_moved': len(moments), 'n_turns_before': len(out_turns),
               'n_turns_after': len(merged), 'moments': moments[:500],
               **({'skipped_mismatch': skipped} if skipped else {}),
               **({'witness': witness_log[:500]} if witness_log else {})}
    return [m[0] for m in merged], [m[1] for m in merged], log
