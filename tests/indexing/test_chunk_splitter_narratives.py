"""Тесты add_table_narratives — дублирующее покрытие таблиц per-row narrative-чанками.

См. ADR-0013.
"""
from datetime import datetime

from morag.indexing.chunk_splitter import add_table_narratives
from morag.sources.base import Chunk


def _chunk(text: str, doc_id: str = 'doc1', payload: dict | None = None, chunk_id: str = 'parent-uuid') -> Chunk:
    return Chunk(
        doc_id=doc_id,
        path=['Test/Path'],
        order=0,
        total=1,
        text=text,
        updated_at=datetime(2026, 5, 1),
        source_type='confluence',
        context='parent context',
        payload=dict(payload or {}),
        vectors={},
        id=chunk_id,
    )


def test_add_narratives_basic():
    """Таблица 5 строк → 5 narrative-чанков с правильным parent_chunk_id и форматом."""
    text = (
        '| Термин | Определение |\n'
        '| --- | --- |\n'
        '| ТС | Торговая Сеть |\n'
        '| УИ | Упрощенная Идентификация |\n'
        '| Флоу | Сценарий оформления |\n'
        '| ЧД | Чистое Досье |\n'
        '| Жёсткая типизация | Полное доверие к сервису |\n'
    )
    parent = _chunk(text, payload={'content_kind': 'table', 'table_part': {'anchor': 'x'}})

    result = add_table_narratives([parent], min_rows=5)

    # Parent + 5 narratives
    assert len(result) == 6
    assert result[0] is parent  # parent неизменён, сохраняет identity

    narratives = result[1:]
    assert all(n.payload['chunk_type'] == 'table_row_narrative' for n in narratives)
    assert all(n.payload['parent_chunk_id'] == 'parent-uuid' for n in narratives)
    # parent-specific метки не копируются в narrative
    assert all('content_kind' not in n.payload for n in narratives)
    assert all('table_part' not in n.payload for n in narratives)
    # Метаданные parent — наследуются
    assert all(n.doc_id == 'doc1' for n in narratives)
    assert all(n.path == ['Test/Path'] for n in narratives)
    assert all(n.source_type == 'confluence' for n in narratives)
    # Narrative-специфика
    assert all(n.order == -1 for n in narratives)
    assert all(n.context == '' for n in narratives)
    assert all(n.vectors == {} for n in narratives)
    # IDs уникальные
    ids = {n.id for n in narratives}
    assert len(ids) == 5
    assert 'parent-uuid' not in ids  # narrative не разделяет id с parent

    # Текст narrative — newline-joined Header: value
    expected_first = 'Термин: ТС\nОпределение: Торговая Сеть'
    assert narratives[0].text == expected_first
    # Последняя строка содержит русские пробелы корректно
    assert 'Жёсткая типизация' in narratives[-1].text
    assert 'Полное доверие к сервису' in narratives[-1].text


def test_skip_small_table():
    """Таблица <min_rows строк — narratives не генерим."""
    text = (
        '| A | B |\n'
        '| --- | --- |\n'
        '| 1 | 2 |\n'
        '| 3 | 4 |\n'
    )
    parent = _chunk(text)
    result = add_table_narratives([parent], min_rows=5)
    assert result == [parent]  # без изменений


def test_skip_empty_cells():
    """Пустые ячейки и '-' пропускаются из narrative."""
    text = (
        '| Имя | Значение | Заметка |\n'
        '| --- | --- | --- |\n'
        '| A | 1 | comment |\n'
        '| B |   | - |\n'
        '| C | 2 |  |\n'
        '| D | - | x |\n'
        '| E | 3 | y |\n'
    )
    parent = _chunk(text)
    result = add_table_narratives([parent], min_rows=5)
    narratives = result[1:]
    assert len(narratives) == 5
    # Полная строка
    assert narratives[0].text == 'Имя: A\nЗначение: 1\nЗаметка: comment'
    # B: пустая ячейка и '-' пропущены
    assert narratives[1].text == 'Имя: B'
    # C: одна пустая
    assert narratives[2].text == 'Имя: C\nЗначение: 2'
    # D: '-' значение и заметка с x
    assert narratives[3].text == 'Имя: D\nЗаметка: x'


def test_chunk_without_table_skipped():
    """Чанк без таблицы — narratives не генерим."""
    parent = _chunk('Some plain text without any table.')
    result = add_table_narratives([parent], min_rows=5)
    assert result == [parent]


def test_skip_row_with_all_empty_cells():
    """Если все ячейки строки пустые — narrative для неё не создаётся."""
    text = (
        '| A | B |\n'
        '| --- | --- |\n'
        '| 1 | 2 |\n'
        '| 3 | 4 |\n'
        '|   |   |\n'
        '| 5 | 6 |\n'
        '| 7 | 8 |\n'
    )
    parent = _chunk(text)
    result = add_table_narratives([parent], min_rows=5)
    narratives = result[1:]
    # 5 data-rows, но одна пустая → 4 narrative
    assert len(narratives) == 4


def test_min_rows_zero_or_negative_returns_unchanged():
    """min_rows < 1 → no-op."""
    parent = _chunk('| A | B |\n| --- | --- |\n| 1 | 2 |\n')
    assert add_table_narratives([parent], min_rows=0) == [parent]
    assert add_table_narratives([parent], min_rows=-1) == [parent]


def test_payload_run_versioning_inherited():
    """Поля run_number/version/indexed_at наследуются от parent (pipeline их потом
    перештампует, но базово они должны быть скопированы из payload)."""
    parent = _chunk(
        '| A | B |\n| --- | --- |\n' + '| x | y |\n' * 5,
        payload={'version': 3, 'run_number': 42, 'creator': 'user1'},
    )
    result = add_table_narratives([parent], min_rows=5)
    narratives = result[1:]
    assert all(n.payload.get('version') == 3 for n in narratives)
    assert all(n.payload.get('run_number') == 42 for n in narratives)
    assert all(n.payload.get('creator') == 'user1' for n in narratives)
