"""Шов с соседями: окно с запасом слышит ЧУЖИЕ слова — оставляем только свои.

Повтор пустого куска (±0.3 с) и переслушивание (±1.5 с) режут звук с запасом: модели нужен
разбег, иначе она теряет первое и последнее слово. Но запас звучит речью соседа, и раньше
весь услышанный текст ложился в кусок: слово на стыке появлялось в записи дважды — у соседа и
здесь (жалоба владельца: «этим объясняются некоторые ошибки»).

Правило — **середина внутри окна**: слово принадлежит куску, если его середина лежит в
[начало, конец] куска. Слова без времён (модель их не дала) — то же правило по сегменту целиком,
грубее, но лучше, чем брать всё. Правило то же, что у починки корпуса (`relisten_fix._piece`),
где оно замерено: без него 13 % правок несли задвоенное слово.

Второй рубеж — `dedupe_head`: если начало куска слово в слово повторяет конец предыдущего (от трёх
слов подряд), повтор снимается. Одиночное совпадение не режем никогда: «да, да» и «ну, ну» —
живая речь, а не шов.

⚠️⚠️ Куски добора дыр (`recovered`) сюда НЕ относятся — замерено 30.09 на пяти эталонах: запас и
отсечка по середине на них теряли живую речь (WER 3.28 → 3.95 %, 34 из 40 пропусков одной записи —
у дыр пасса-1). Граница такого куска — край дыры, где пасс-1 речи не слышал, соседа там нет, а
времена слов декодера на коротком куске неточны.
"""
from __future__ import annotations

import re

MIN_DUP = 3      # от стольких совпавших слов подряд повтор на стыке считается швом
TAIL_WORDS = 12  # сколько слов конца соседа сравнивать

_TOKEN = re.compile(r'[\w-]+', re.UNICODE)


def _mid(x: dict) -> float:
    return (float(x.get('start') or 0.0) + float(x.get('end') or 0.0)) / 2.0


def keep_inside(segments, off: float, a: float, b: float) -> list[dict]:
    """Сегменты ответа модели (время от начала вырезки) → АБСОЛЮТНЫЕ сегменты куска [a, b].

    Слово остаётся, если его середина внутри [a, b]; сегмент без времён слов — если внутри его
    середина. Метрики сегмента (`avg_logprob` и т. п.) сохраняются: они про звук, не про границу.
    """
    out = []
    for s in segments or ():
        seg = dict(s)
        seg['start'] = round(off + float(s.get('start') or 0.0), 2)
        seg['end'] = round(off + float(s.get('end') or 0.0), 2)
        ws = s.get('words') or []
        if ws:
            kept = []
            for w in ws:
                w2 = dict(w)
                w2['start'] = round(off + float(w.get('start') or 0.0), 2)
                w2['end'] = round(off + float(w.get('end') or 0.0), 2)
                if a <= _mid(w2) <= b:
                    kept.append(w2)
            if not kept:
                continue
            seg['words'] = kept
            seg['text'] = ' '.join(str(w.get('word') or '').strip() for w in kept).strip()
            seg['start'], seg['end'] = kept[0]['start'], kept[-1]['end']
        else:
            if not a <= _mid(seg) <= b:
                continue
            seg['text'] = (s.get('text') or '').strip()
        if seg['text']:
            out.append(seg)
    return out


def text_of(segments) -> str:
    return ' '.join(s['text'] for s in segments if s.get('text')).strip()


def _norm(t: str) -> str:
    return t.casefold().replace('ё', 'е')


def dedupe_head(prev_text: str, text: str) -> tuple[str, int]:
    """Снять с начала `text` слова, повторяющие конец `prev_text` (≥ MIN_DUP подряд).

    Возвращает (новый текст, сколько слов снято). Сравнение — по словам без знаков и регистра,
    а снимается префикс исходного текста до конца последнего совпавшего слова, чтобы пунктуация
    остального не пострадала.
    """
    tail = [_norm(m.group()) for m in _TOKEN.finditer(prev_text or '')][-TAIL_WORDS:]
    head = list(_TOKEN.finditer(text or ''))
    best = 0
    for k in range(min(len(tail), len(head)), MIN_DUP - 1, -1):
        if tail[-k:] == [_norm(m.group()) for m in head[:k]]:
            best = k
            break
    if not best:
        return text, 0
    rest = text[head[best - 1].end():].lstrip(' ,.;:—-…')
    return rest.strip(), best


def dedupe_chunks(chunks) -> list[dict]:
    """Страховка после пасса-2: у куска, декодированного С ЗАПАСОМ, снять повтор конца соседа.

    Только у кусков с запасом (`retried`, `relistened`): обычный кусок и кусок добора режутся без
    запаса, и совпадение там — это речь, повторённая говорящим. Журнал — в ответе.
    """
    log = []
    for i in range(1, len(chunks)):
        c, p = chunks[i], chunks[i - 1]
        if not (c.get('retried') or c.get('relistened')):
            continue
        new, n = dedupe_head(p.get('raw') or '', c.get('raw') or '')
        if not n:
            continue
        log.append({'start': round(float(c['start']), 2), 'dropped': n,
                    'was': (c.get('raw') or '')[:80]})
        c['raw'] = new
        segs = c.get('segments') or []
        if segs:
            first = segs[0]
            t, k = dedupe_head(p.get('raw') or '', first.get('text') or '')
            if k:
                first['text'] = t
                if first.get('words'):
                    first['words'] = first['words'][k:]
                if not t:
                    segs.pop(0)
    return log
