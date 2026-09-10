"""Разворот аргументов агента: короткий код → `doc_id`, и три РАЗНЫХ отказа.

Почему это отдельный файл с тестами, а не «и так понятно»: замерено, что на единственное «не
найдено» агент отвечает повтором того же испорченного идентификатора четыре раза подряд (ADR-0025).
Ценность правки не в том, что код разворачивается, а в том, что случаи «ты опечатался», «такого
документа нет» и «это раздел, а не документ» разведены — каждому нужно своё действие.

⚠️ Разворот обязан происходить ДО инструмента: `get_descendant_doc_ids` кладёт переданный
`section_id` в результат строкой, без проверки, — нерезолвнутый код прошёл бы там за валидный
`doc_id` и отфильтровал бы выдачу в ноль.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from morag import shortid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402

DOC = 'local:demo:talks/2024/alpha.md'
SECTION = 'local:demo:talks/2024/'
CODE_DOC = shortid.make(DOC)
CODE_SECTION = shortid.make(SECTION, structural=True)


def pipe() -> Pipeline:
    """Pipeline без живых клиентов: подменяем только доступ к таблице кодов."""
    p = object.__new__(Pipeline)
    p._short_maps = lambda: (
        {CODE_DOC: DOC, CODE_SECTION: SECTION},
        {DOC: CODE_DOC, SECTION: CODE_SECTION},
    )
    p._cite_units = {}
    return p


def test_code_resolves_to_the_long_doc_id():
    assert pipe()._resolve_id_arg(CODE_DOC, want='doc') == (DOC, None)


def test_long_doc_id_still_accepted():
    """Совместимость: `find_section` у документных корпусов печатает длинные id, конфиги их хранят,
    подкаст на них живёт. Приём обеих форм — не удобство, а условие, что ничего не сломается."""
    assert pipe()._resolve_id_arg(DOC, want='doc') == (DOC, None)
    assert pipe()._resolve_id_arg('confluence:demo:att:12345', want='doc') == (
        'confluence:demo:att:12345', None)


def test_corrupted_code_says_it_is_corrupted():
    """Первый из трёх отказов: контрольный знак не сошёлся ⇒ виновата ПЕРЕПИСЬ, а не корпус."""
    broken = CODE_DOC[:-1] + ('0' if CODE_DOC[-1] != '0' else '1')
    doc_id, note = pipe()._resolve_id_arg(broken, want='doc')
    assert doc_id is None
    assert 'испорчен' in note and 'дословно' in note


def test_wellformed_but_unknown_code_says_the_document_is_absent():
    """Второй отказ: код безупречен, документа нет. Действие другое — искать, а не переписывать."""
    other = shortid.make('local:demo:talks/2024/never-indexed.md')
    doc_id, note = pipe()._resolve_id_arg(other, want='doc')
    assert doc_id is None
    assert 'такого документа' in note and 'search' in note


def test_section_passed_as_document_is_named_as_such():
    """Третий отказ. Раньше раздел в `doc_ids` давал пустой фильтр и обвинение агенту в
    выдумывании идентификатора — на код, который мы сами ему и напечатали."""
    doc_id, note = pipe()._resolve_id_arg(CODE_SECTION, want='doc')
    assert doc_id is None
    assert 'РАЗДЕЛ' in note and 'section_ids' in note


def test_document_passed_as_section_is_allowed_silently():
    """Обратное сочетание безвредно: документ в `section_ids` сужает поиск до него самого.
    Ошибкой не считаем и не шумим — лишний отказ дороже безвредной вольности."""
    assert pipe()._resolve_id_arg(CODE_DOC, want='section') == (DOC, None)


def test_empty_reference_is_refused():
    doc_id, note = pipe()._resolve_id_arg('   ', want='doc')
    assert doc_id is None and note


# --- нормализация аргументов инструмента ------------------------------------


def test_get_doc_argument_is_replaced_before_the_tool_runs():
    args, notes = pipe()._normalize_tool_ids('get_doc', {'doc_id': CODE_DOC, 'query': 'что'})
    assert args['doc_id'] == DOC
    assert args['query'] == 'что', 'чужие аргументы не трогаем'
    assert notes == []


def test_search_lists_are_resolved_and_bad_refs_reported():
    """Часть скоупа развернулась, часть нет: искать по остальному надо, но молчать нельзя —
    сегодня неизвестный идентификатор выбрасывается из сужения МОЛЧА."""
    broken = CODE_DOC[:-1] + ('0' if CODE_DOC[-1] != '0' else '1')
    args, notes = pipe()._normalize_tool_ids(
        'search', {'query': 'x', 'doc_ids': [CODE_DOC, broken], 'section_ids': [CODE_SECTION]},
    )
    assert args['doc_ids'] == [DOC]
    assert args['section_ids'] == [SECTION]
    assert len(notes) == 1 and 'испорчен' in notes[0]


def test_normalize_does_not_mutate_the_callers_arguments():
    """Аргументы приходят из разобранного tool_call и уходят в лог — портить их нельзя."""
    original = {'doc_id': CODE_DOC, 'query': 'что'}
    pipe()._normalize_tool_ids('get_doc', original)
    assert original['doc_id'] == CODE_DOC


@pytest.mark.parametrize('name', ['find_section', 'catalog'])
def test_tools_without_identifier_arguments_are_untouched(name):
    args, notes = pipe()._normalize_tool_ids(name, {'query': 'x'})
    assert args == {'query': 'x'} and notes == []
