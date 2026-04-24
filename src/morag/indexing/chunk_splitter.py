"""Post-chunk splitter для больших markdown-таблиц.

Работает ПОСЛЕ chunker'а, ДО ChunkProcessors. Для каждого чанка содержащего
markdown-таблицу с > max_rows data-строк режет чанк на последовательность
sub-чанков:

    preamble + header+separator+rows[0:N]  → sub-chunk 1
    header+separator+rows[N:2N]            → sub-chunk 2
    header+separator+rows[2N:3N] + tail    → sub-chunk 3

Sub-чанки наследуют метаданные родителя (doc_id, path, context, updated_at,
source_type, payload). `order` перенумеровывается у всех чанков документа,
`total` пересчитывается. Sub-чанкам добавляются payload-маркеры для retrieval
и будущей реконструкции таблицы:

    payload.content_kind = 'table'
    payload.table_part = {
        'anchor': 'hash identifying this table within doc',
        'index': 2,         # 2-й из 5
        'total': 5,
        'headers': ['Col1', 'Col2'],
    }

Риски обработки: chunker может предварительно разрезать таблицу между своими
чанками. Мы срабатываем только на чанках, где таблица полноформатная
(есть `header + separator + data`). «Хвосты» без header оставляем как есть.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from morag.sources.base import Chunk

_SEPARATOR_RE = re.compile(r'^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$')


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped.startswith('|') and bool(_SEPARATOR_RE.match(stripped))


def _is_table_row(line: str) -> bool:
    return line.strip().startswith('|')


def _find_table(lines: list[str], start: int = 0) -> tuple[int, int, int, int] | None:
    """Найти первую markdown-таблицу в строках начиная с start.

    Возвращает (table_start, data_start, data_end, table_end) или None.
    - table_start: индекс строки-header
    - data_start: индекс первой data-строки (после separator)
    - data_end: индекс сразу после последней data-строки (exclusive)
    - table_end: == data_end (для симметрии)

    Игнорирует code-fence блоки.
    """
    in_code_fence = False
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_fence = not in_code_fence
            i += 1
            continue
        if in_code_fence:
            i += 1
            continue
        if _is_table_row(line) and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            data_start = i + 2
            data_end = data_start
            while data_end < len(lines) and _is_table_row(lines[data_end]):
                data_end += 1
            return i, data_start, data_end, data_end
        i += 1
    return None


def _parse_headers(header_line: str) -> list[str]:
    cells = header_line.strip().strip('|').split('|')
    return [c.strip() for c in cells]


def _make_anchor(doc_id: str, header_line: str, table_start_line: int) -> str:
    """Стабильный идентификатор таблицы в пределах документа.

    Используем хэш (doc_id + header + first line index) — устойчив между
    индексациями если структура не меняется.
    """
    src = f'{doc_id}|{header_line.strip()}|{table_start_line}'.encode('utf-8')
    return hashlib.md5(src).hexdigest()[:10]


def _split_chunk_tables(chunk: Chunk, max_rows: int) -> list[Chunk]:
    """Попытаться порезать один чанк по table-boundaries.

    Возвращает список из одного (неизменённый) или нескольких sub-чанков.
    """
    if max_rows <= 0 or not chunk.text.strip():
        return [chunk]

    lines = chunk.text.split('\n')
    located = _find_table(lines)
    if located is None:
        return [chunk]

    table_start, data_start, data_end, _ = located
    data_rows = data_end - data_start
    if data_rows <= max_rows:
        return [chunk]

    header_line = lines[table_start]
    separator_line = lines[table_start + 1]
    headers = _parse_headers(header_line)
    anchor = _make_anchor(chunk.doc_id, header_line, table_start)

    preamble = lines[:table_start]
    tail_after_table = lines[data_end:]  # что идёт после таблицы в том же чанке
    rows = lines[data_start:data_end]

    # Если после таблицы ещё есть другая таблица — пока игнорируем (редкий кейс,
    # обработается следующей итерацией splitter'а в будущем).
    # Сейчас tail прикрепляется к последнему sub-чанку.

    parts: list[list[str]] = []
    for offset in range(0, len(rows), max_rows):
        parts.append(rows[offset:offset + max_rows])

    sub_chunks: list[Chunk] = []
    total_parts = len(parts)
    for idx, part_rows in enumerate(parts):
        lines_out: list[str] = []
        if idx == 0 and preamble:
            lines_out.extend(preamble)
            # гарантируем пустую строку между preamble и таблицей
            if lines_out and lines_out[-1].strip():
                lines_out.append('')
        lines_out.append(header_line)
        lines_out.append(separator_line)
        lines_out.extend(part_rows)
        if idx == total_parts - 1 and tail_after_table:
            lines_out.append('')
            lines_out.extend(tail_after_table)
        text = '\n'.join(lines_out)

        # Копируем payload/vectors так чтобы изменения в одном sub-чанке не
        # влияли на другие (shallow copy dict'ов)
        sub_payload = dict(chunk.payload)
        sub_payload['content_kind'] = 'table'
        sub_payload['table_part'] = {
            'anchor': anchor,
            'index': idx + 1,
            'total': total_parts,
            'headers': headers,
        }
        sub = replace(
            chunk,
            text=text,
            payload=sub_payload,
            vectors={},  # векторы будут пересчитаны chunk_processors'ом
            id=chunk.id if idx == 0 else _new_chunk_id(),
        )
        sub_chunks.append(sub)
    return sub_chunks


def _new_chunk_id() -> str:
    import uuid
    return str(uuid.uuid4())


def split_table_chunks(chunks: list[Chunk], max_rows: int) -> list[Chunk]:
    """Применить split к каждому чанку, вернуть обновлённый список.

    Перенумерует order/total у всех чанков (sub-чанки вставляются на место
    оригинала).
    """
    if max_rows <= 0 or not chunks:
        return chunks

    result: list[Chunk] = []
    for chunk in chunks:
        result.extend(_split_chunk_tables(chunk, max_rows))

    # Перенумеровать order + обновить total
    total = len(result)
    renumbered: list[Chunk] = []
    for i, c in enumerate(result):
        renumbered.append(replace(c, order=i, total=total))
    return renumbered
