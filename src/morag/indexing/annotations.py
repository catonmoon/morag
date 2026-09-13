"""Аннотации к документу (ADR-0027): какие элементы сайдкара относятся к чанку и как их печатать.

Элемент — словарь `{kind, …}` с якорем по времени (`t0`/`t1` отрезок, `at` точка) или по
символьному смещению (`o0`/`o1`, `offset`). Ядро не знает, что стоит за `kind`: поле чанка
называет конфиг (`indexing.annotations.field`), привязки — всегда `kind == 'ref'`.
"""
from __future__ import annotations

from morag.indexing.token_counter import TokenCounter

# Хвост прежнего экрана, оставшийся в чанке от притяжения границы к концу предложения (±12 с),
# в чанк соседа не берём: отрезок считается действующим, если покрывает начало чанка не меньше
# 5 с (или половины своей длины, если он короче 10 с).
_MIN_OVERLAP_SEC = 5.0


def _span(item: dict) -> tuple[float, float, str] | None:
    """(начало, конец, 'sec' | 'char') якоря элемента; None — якоря нет."""
    for a, b, unit in (('t0', 't1', 'sec'), ('at', 'at', 'sec'), ('o0', 'o1', 'char'), ('offset', 'offset', 'char')):
        if item.get(a) is not None:
            start = float(item[a])
            end = float(item[b]) if item.get(b) is not None else start
            return start, max(start, end), unit
    return None


def item_text(item: dict) -> str:
    """Текст элемента для векторов, лексики и контекста: заголовок + текст; у привязки — цитата и
    разрешение («вот здесь» → что там было)."""
    if item.get('kind') == 'ref':
        quote, text = item.get('quote') or '', item.get('text') or ''
        if quote and text:
            return f'«{quote}» — {text}'
        return f'«{quote}»' if quote else text
    return '\n'.join(str(x) for x in (item.get('title'), item.get('text')) if x)


def annotation_text(items: list[dict]) -> str:
    return '\n'.join(t for t in (item_text(it) for it in items) if t)


def select_annotations(
    items: list[dict], field: str, *,
    start_sec: float | None = None, end_sec: float | None = None,
    char_start: int | None = None, char_end: int | None = None,
    counter: TokenCounter | None = None, max_tokens: int = 0,
) -> list[dict]:
    """Элементы `kind == field` и привязки `ref`, относящиеся к чанку, по времени.

    Отрезок берётся, если сменился ВНУТРИ чанка или ещё действовал в его начале (см.
    `_MIN_OVERLAP_SEC`); точечный (`t0 == t1`) — если лежит внутри; привязка — если её `at`
    внутри. Якорь по символам — те же правила по смещениям. Порядок — по времени. `max_tokens`
    (при `counter`) — потолок суммарного текста: лишние элементы отбрасываются с конца, а
    первый, не влезший целиком, режется.
    """
    out: list[tuple[float, dict]] = []
    for it in items:
        kind = it.get('kind')
        if kind not in (field, 'ref'):
            continue
        span = _span(it)
        if span is None:
            continue
        a, b, unit = span
        lo, hi = (start_sec, end_sec) if unit == 'sec' else (char_start, char_end)
        if lo is None:
            continue
        hi = float('inf') if hi is None else hi
        tol = 1.0 if unit == 'sec' else 0.0
        if kind == 'ref' or a == b:
            if lo - tol <= a < hi:
                out.append((a, it))
            continue
        overlap = min(b, hi) - max(a, lo)
        threshold = min(_MIN_OVERLAP_SEC, 0.5 * (b - a)) if unit == 'sec' else 0.5 * (b - a)
        if lo - tol <= a < hi or overlap >= threshold:
            out.append((a, it))
    out.sort(key=lambda x: x[0])
    selected = [it for _, it in out]
    if not (counter and max_tokens):
        return selected
    kept: list[dict] = []
    used = 0
    for it in selected:
        t = counter.count(item_text(it))
        if used + t <= max_tokens:
            kept.append(it)
            used += t
            continue
        if not kept and it.get('text'):
            cut = dict(it)
            cut['text'] = counter.truncate(str(it['text']), max(1, max_tokens - counter.count(str(it.get('title') or ''))))
            kept.append(cut)
        break
    return kept
