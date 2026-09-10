"""Рендер retrieval-выдачи агенту: идентификаторы документов и гарантии формата.

Первый тест на `_render_chunks_block` в репозитории. Причина завести его именно сейчас:
схема `get_doc` требует «ID документа из результатов find_section/search», а выдача моментов
(`timestamp_citations`) идентификатора не содержала вовсе. Агенту нечего было подставить, и он
собирал id по аналогии — замерено на корпусе расшифровок: 57% вызовов `get_doc` уходили в
несуществующий документ.

Размещение выбрано замером (ADR-0025): по умолчанию `grouped`, `inline` оставлен как вариант.
Тесты фиксируют не «как красивее», а инварианты, которые обязаны держаться при любом значении:
документная ветка не меняется, документ не дублируется, id пригоден для `get_doc`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag import shortid  # noqa: E402
from morag_pipeline import (  # noqa: E402
    Pipeline,
    _call_signature,
    _group_by_doc,
    _moment_head,
    _print_order,
    _repeat_note,
)

DOC_A = 'local:demo:talks/2024/alpha.md'
DOC_B = 'local:demo:talks/2024/beta.md'
# Агенту печатается КОД, а не длинный идентификатор (ADR-0025) — тесты проверяют именно его.
CODE_A = shortid.make(DOC_A)
CODE_B = shortid.make(DOC_B)


def block(chunks: list[dict], mode: str = 'none', timestamps: bool = True) -> str:
    """Рендер настоящим методом, но без живых клиентов.

    `title` у чанка задан намеренно: тогда `_get_doc_title` не зовётся и поисковик не нужен —
    проверяется ровно форматирование, а не окружение.
    """
    pipe = object.__new__(Pipeline)
    pipe._s = {'timestamp_citations': timestamps, 'doc_ids_in_results': mode}
    pipe._doc_numbering = {}
    pipe._cite_units = {}
    pipe._live_titles = {}
    # Таблица кодов живёт в кэше searcher; здесь подменяем сам доступ к ней — живых клиентов
    # у объекта нет, а проверяем мы форматирование.
    pipe._short_maps = lambda: ({CODE_A: DOC_A, CODE_B: DOC_B},
                                {DOC_A: CODE_A, DOC_B: CODE_B})
    return pipe._render_chunks_block(chunks)


def moment(doc_id: str, sec: int, text: str, title: str = 'Доклад') -> dict:
    return {'doc_id': doc_id, 'start_sec': sec, 'text': text, 'title': title,
            'speakers': ['Кузнецова'], 'path': [doc_id.split(':')[-1]]}


def document(doc_id: str, text: str, order: int = 0) -> dict:
    return {'doc_id': doc_id, 'text': text, 'title': 'Страница', 'order': order,
            'path': [doc_id.split(':')[-1]]}


# --- инвариант ADR-0015: документная ветка не меняется -----------------------


@pytest.mark.parametrize('mode', ['none', 'inline', 'grouped'])
def test_document_branch_is_byte_identical(mode):
    """Документным корпусам обещан прежний рендер байт в байт — при ЛЮБОМ значении."""
    chunks = [document(DOC_A, 'первый'), document(DOC_B, 'второй')]
    assert block(chunks, mode=mode, timestamps=False) == block(chunks, mode='none', timestamps=False)


def test_off_keeps_previous_moment_format():
    """Выключенный переключатель — прежний вывод: номер, метка, текст. И ничего больше."""
    out = block([moment(DOC_A, 30, 'реплика')], mode='none')
    assert 'id:' not in out and DOC_A not in out and CODE_A not in out
    assert '[1] Доклад · 0:30 · Кузнецова' in out


# --- варианты размещения ----------------------------------------------------


def test_inline_puts_id_on_every_moment():
    out = block([moment(DOC_A, 10, 'раз'), moment(DOC_A, 90, 'два')], mode='inline')
    assert out.count(f'id: {CODE_A}') == 2


def test_grouped_keeps_relevance_order_of_documents():
    """Группировка не имеет права переставить выдачу: документ встаёт туда, где его лучший момент."""
    out = block([moment(DOC_B, 5, 'самый релевантный'), moment(DOC_A, 10, 'слабее'),
                 moment(DOC_B, 90, 'ещё из B')], mode='grouped')
    assert out.index(CODE_B) < out.index(CODE_A)
    assert out.count(f'id: {CODE_B}') == 1, 'документ получает одну шапку, а не по одной на момент'


def test_grouped_keeps_id_on_its_own_line():
    """⚠️ Идентификатор отдельной строкой, а не в скобках после заголовка.

    Ловилось на живом прогоне: из формы «Документ: Заголовок (id: …)» модель копировала в
    `get_doc` всю строку целиком, вместе с заголовком и скобкой, — и документ не находился.
    """
    out = block([moment(DOC_A, 10, 'раз', title='Доклад с. точкой')], mode='grouped')
    assert f'\nid: {CODE_A}' in out
    assert f'(id: {CODE_A})' not in out


def test_grouped_does_not_regroup_mixed_output():
    """Смешанная выдача (моменты + документы) не перестраивается: порядок там несёт релевантность."""
    chunks = [moment(DOC_A, 10, 'момент'), document(DOC_B, 'страница')]
    assert block(chunks, mode='grouped') == block(chunks, mode='none')


# --- номера в блоке ---------------------------------------------------------


@pytest.mark.parametrize('mode', ['none', 'inline', 'grouped'])
def test_numbers_run_in_print_order(mode):
    """⚠️ Номера обязаны идти по возрастанию СВЕРХУ ВНИЗ блока.

    Своя же группировка это и сломала: моменты печатались по документам, а номера
    раздавались в порядке релевантности, и агент читал «[1] [2] [3] [4] [6] [8] [5]…».
    Номер под моментом в такой выдаче не значит ничего — а он и есть то, чем модель
    ссылается на источник.
    """
    chunks = [moment(DOC_B, 5, 'B раз'), moment(DOC_A, 10, 'A раз'),
              moment(DOC_B, 90, 'B два'), moment(DOC_A, 200, 'A два')]
    printed = [int(m) for m in re.findall(r'^\[(\d+)\]', block(chunks, mode=mode), re.M)]
    assert printed == sorted(printed), printed
    assert printed == [1, 2, 3, 4]


# --- id обязан быть пригоден для get_doc ------------------------------------


@pytest.mark.parametrize('mode', ['inline', 'grouped'])
def test_printed_id_is_the_one_get_doc_accepts(mode):
    """⚠️ Гарантия контракта: напечатанное обязано разворачиваться обратно в тот же документ.

    Печатается код документа, а не ключ момента (`{doc_id}#{сек}`) и не длинный `doc_id`. Код
    проходит проверку контрольным знаком и резолвится по таблице — ровно то, что делает `get_doc`.
    """
    out = block([moment(DOC_A, 30, 'реплика')], mode=mode)
    assert CODE_A in out
    assert DOC_A not in out, 'длинный идентификатор агенту больше не показывается'
    assert f'{CODE_A}#30' not in out
    assert shortid.parse(CODE_A) is not None


# --- чистые помощники -------------------------------------------------------


def test_moment_head_adds_id_only_inline():
    assert _moment_head(1, 'метка', DOC_A, 'none') == '[1] метка'
    assert _moment_head(1, 'метка', DOC_A, 'grouped') == '[1] метка'
    assert _moment_head(1, 'метка', DOC_A, 'inline').endswith(f'· id: {DOC_A}')


def test_print_order_touches_only_grouped_moments():
    """Переупорядочивание — плата за группировку, и больше нигде оно не оправдано."""
    units = {'b#5': {'meta': {'doc_id': DOC_B, 'start_sec': 5}},
             'a#10': {'meta': {'doc_id': DOC_A, 'start_sec': 10}},
             'b#90': {'meta': {'doc_id': DOC_B, 'start_sec': 90}}}
    keys = ['b#5', 'a#10', 'b#90']
    assert _print_order(keys, units, 'grouped') == ['b#5', 'b#90', 'a#10']
    assert _print_order(keys, units, 'inline') == keys
    assert _print_order(keys, units, 'none') == keys
    # документ среди моментов — блок не перестраивается вовсе
    mixed = dict(units, doc={'meta': {'doc_id': DOC_A}})
    assert _print_order([*keys, 'doc'], mixed, 'grouped') == [*keys, 'doc']


def test_group_by_doc_preserves_first_appearance():
    units = {
        'b#5': {'meta': {'doc_id': DOC_B}},
        'a#10': {'meta': {'doc_id': DOC_A}},
        'b#90': {'meta': {'doc_id': DOC_B}},
    }
    assert _group_by_doc(['b#5', 'a#10', 'b#90'], units) == [
        (DOC_B, ['b#5', 'b#90']), (DOC_A, ['a#10'])]


# --- отказ get_doc: чем помочь агенту, который опечатался в id --------------


def _pipe_with_seen(keys: list[tuple[str, str]]):
    """Pipeline с уже отданными единицами цитирования (ключ → doc_id), без живых клиентов."""
    pipe = object.__new__(Pipeline)
    pipe._cite_units = {k: {'doc_id': d} for k, d in keys}
    docs = {d for _, d in keys}
    pipe._short_maps = lambda: ({shortid.make(d): d for d in docs},
                                {d: shortid.make(d) for d in docs})
    return pipe


def test_refusal_lists_ids_already_offered():
    """Замерено: на отказ агент звал ТОТ ЖЕ испорченный id четыре раза подряд — он не знает,
    что опечатался. Перечень уже отданных кодов разрывает цикл, ничего не угадывая."""
    hint = _pipe_with_seen([(f'{DOC_A}#10', DOC_A), (f'{DOC_A}#90', DOC_A),
                            (f'{DOC_B}#5', DOC_B)])._ids_offered_hint()
    assert hint.count(CODE_A) == 1, 'документ упомянут один раз, а не по разу на момент'
    assert CODE_B in hint
    assert DOC_A not in hint, 'в подсказке те же коды, что в выдаче, — иначе она ей противоречит'


def test_refusal_says_nothing_when_nothing_was_offered():
    """Ретрив ещё ничего не дал — подсказывать нечем, и выдумывать нельзя."""
    assert _pipe_with_seen([])._ids_offered_hint() == ''


def test_refusal_hint_is_capped():
    """Перечень ограничен: отказ не имеет права стоить дороже полезной выдачи."""
    many = [(f'local:demo:doc{i}.md', f'local:demo:doc{i}.md') for i in range(50)]
    hint = _pipe_with_seen(many)._ids_offered_hint()
    assert len([ln for ln in hint.splitlines() if shortid.parse(ln.strip())]) == 20
    assert 'и ещё 30' in hint


# --- гейт против зацикливания на повторном вызове ----------------------------


def test_call_signature_ignores_argument_order():
    """⚠️ Модель перечисляет аргументы в произвольном порядке. Без нормализации гейт против
    зацикливания пропускал бы половину повторов, оставаясь при этом «работающим»."""
    a = _call_signature('search', {'query': 'x', 'limit': 5})
    b = _call_signature('search', {'limit': 5, 'query': 'x'})
    assert a == b
    assert _call_signature('search', {'query': 'y'}) != a
    assert _call_signature('catalog', {}) != _call_signature('search', {})


def test_repeat_note_says_which_tool_and_what_to_do():
    """Замерено: `catalog args={}` десять раз подряд сожгли весь лимит итераций. Отказ обязан
    сказать не только «повтор», но и куда смотреть вместо него."""
    note = _repeat_note('catalog')
    assert 'catalog' in note
    assert 'выше' in note


def test_repeat_gate_lets_a_shrunk_result_be_asked_again():
    """⚠️ Гейт против зацикливания отказывает словами «результат выше в диалоге». Но бюджет
    контекста ПЕРЕ-РЕНДЕРИТ уже выданные результаты и может вытеснить часть чанков — тогда выше
    лежит урезанный блок, и отказ стал бы враньём, а повтор законен: агент возвращает то, что у
    него забрали.

    Проверяем сам признак: бюджет обязан пометить урезанный результат его `tool_call_id`.
    """
    pipe = object.__new__(Pipeline)
    # Окно просторное: вытеснять нечего, сработает только дедуп между сообщениями.
    pipe._s = {'agent_context_window': 32768, 'agent_max_tokens': 100,
               'doc_ids_in_results': 'none', 'timestamp_citations': True}
    pipe._doc_numbering, pipe._cite_units, pipe._live_titles = {}, {}, {}
    pipe._short_maps = lambda: ({}, {})
    pipe._shrunk_tool_calls = set()
    # Два поиска, второй возвращает те же чанки — дедуп заберёт их у второго сообщения.
    shared = [dict(moment(DOC_A, 10, 'раз'), chunk_id='c1'),
              dict(moment(DOC_A, 90, 'два'), chunk_id='c2')]
    pipe._retrieval_entries = [
        {'tool_call_id': 'call_1', 'chunks': shared},
        {'tool_call_id': 'call_2', 'chunks': shared},
    ]
    messages = [
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'старое'},
        {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'старое'},
    ]
    pipe._budget_agent_context(messages)
    assert pipe._shrunk_tool_calls == set(), (
        'дедуп между сообщениями — НЕ потеря: чанк показан под первым поиском, повторять нечего')

    # А теперь окно, в которое не влезает даже один чанк сверх первого, — настоящее вытеснение.
    pipe._shrunk_tool_calls = set()
    pipe._s['agent_context_window'] = 3200
    pipe._retrieval_entries = [
        {'tool_call_id': 'call_1', 'chunks': [dict(moment(DOC_A, 10, 'раз' * 400), chunk_id='c1'),
                                              dict(moment(DOC_A, 90, 'два' * 400), chunk_id='c2')]},
    ]
    pipe._budget_agent_context([{'role': 'tool', 'tool_call_id': 'call_1', 'content': 'старое'}])
    assert 'call_1' in pipe._shrunk_tool_calls, 'вытесненный чанк обязан быть помечен'
