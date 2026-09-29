"""Реестр инструментов графа: схема отвергает лишнее и неверное, журнал и счётчик ведутся."""
from __future__ import annotations

import pytest

from conveyor.tools import Registry, Tool, ToolError, validate

LISTEN = {'properties': {'ear': {'type': 'string', 'enum': ['second', 'clean']},
                         'window': {'type': 'string', 'enum': ['same', 'chunk']},
                         'pad': {'type': 'number'}},
          'required': ['ear']}


def test_schema_rejects_extra_missing_wrongly_typed_and_out_of_enum_arguments():
    validate(LISTEN, {'ear': 'second'})
    validate(LISTEN, {'ear': 'clean', 'window': 'chunk', 'pad': 1.5})
    with pytest.raises(ToolError, match='лишние аргументы: model'):
        validate(LISTEN, {'ear': 'second', 'model': 'x'})
    with pytest.raises(ToolError, match='нет обязательного аргумента «ear»'):
        validate(LISTEN, {'window': 'same'})
    with pytest.raises(ToolError, match='не из перечня') as e:
        validate(LISTEN, {'ear': 'third'})
    assert 'second, clean' in e.value.hint, 'подсказка называет допустимые значения'
    with pytest.raises(ToolError, match='должен быть number'):
        validate(LISTEN, {'ear': 'second', 'pad': True})   # bool — не число


def _registry():
    journal: list = []
    meter: dict = {}

    async def listen(ear: str, window: str = 'same') -> dict:
        return {'ear': ear, 'text': 'слово ' * 100, 'a': 10.0, 'b': 22.5}

    async def boom() -> dict:
        raise RuntimeError('бэкенд упал')

    async def reader() -> dict:
        return {'suspicious': []}

    reg = Registry([Tool('listen', 'слушать', listen, schema=LISTEN,
                         cost=lambda a, r: {'audio_s': r['b'] - r['a']}),
                    Tool('boom', 'падает', boom),
                    Tool('reader_flags', 'читатель', reader, cost=lambda a, r: {'llm_calls': 1})],
                   journal=journal, meter=meter, place='chunk:3')
    return reg, journal, meter


async def test_call_journals_the_call_and_counts_the_cost():
    reg, journal, meter = _registry()
    res = await reg.call('listen', ear='second')
    assert res['ear'] == 'second'
    row = journal[-1]
    assert row['place'] == 'chunk:3' and row['tool'] == 'listen' and row['args'] == {'ear': 'second'}
    assert row['result']['text'].endswith('…') and len(row['result']['text']) < 130, 'сводка, не сам текст'
    assert 'ms' in row
    await reg.call('reader_flags')
    assert meter['tool_calls'] == 2 and meter['tool.listen'] == 1 and meter['tool.reader_flags'] == 1
    assert meter['audio_s'] == 12.5 and meter['llm_calls'] == 1


async def test_unknown_tool_and_bad_arguments_are_tool_errors_with_hints():
    reg, journal, meter = _registry()
    with pytest.raises(ToolError, match='нет инструмента «guess»') as e:
        await reg.call('guess')
    assert 'listen' in e.value.hint
    with pytest.raises(ToolError):
        await reg.call('listen', ear='fourth')
    assert not journal, 'отказ схемы — до вызова, в журнал не попадает'


async def test_a_crashing_tool_is_journaled_and_re_raised():
    reg, journal, meter = _registry()
    with pytest.raises(RuntimeError):
        await reg.call('boom')
    assert journal[-1]['error'].startswith('RuntimeError') and meter['tool_failures'] == 1


def test_schemas_are_in_function_calling_form():
    reg, _, _ = _registry()
    s = reg.schemas()
    assert [x['function']['name'] for x in s] == ['listen', 'boom', 'reader_flags']
    params = s[0]['function']['parameters']
    assert params['type'] == 'object' and params['additionalProperties'] is False
    assert params['properties']['ear']['enum'] == ['second', 'clean'] and params['required'] == ['ear']
