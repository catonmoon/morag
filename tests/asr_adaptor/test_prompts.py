"""Реестр промптов и файл переопределений (prompts.py)."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import prompts

SERVICE = Path(prompts.__file__).resolve().parent


@pytest.fixture(autouse=True)
def restore():
    """Переопределение меняет константы стадий — после теста вернуть встроенные."""
    before = prompts.builtins()
    yield
    prompts.apply(before)


def test_every_prompt_constant_of_the_stages_is_registered():
    """Страховка от забытого промпта: новая константа `*_SYS` в stages/ обязана получить имя."""
    registered = {(m, a) for m, a in prompts.REGISTRY.values()}
    found = set()
    for path in sorted((SERVICE / 'stages').glob('*.py')):
        mod = f'stages.{path.stem}'
        for name in re.findall(r'^(_?[A-Z][A-Z_]*_SYS[A-Z_]*)\s*=', path.read_text(encoding='utf-8'), re.M):
            if isinstance(getattr(importlib.import_module(mod), name), str):
                found.add((mod, name))
    assert found <= registered, f'промпты без имени в реестре: {sorted(found - registered)}'
    for name in prompts.REGISTRY:
        assert isinstance(prompts.builtin(name), str) and prompts.builtin(name)


def test_dump_is_a_valid_override_file_that_changes_nothing(tmp_path):
    import tomllib
    text = prompts.dump()
    parsed = tomllib.loads(text)
    assert set(parsed) == set(prompts.REGISTRY)
    path = tmp_path / 'p.toml'
    path.write_text(text, encoding='utf-8')
    assert prompts.parse(str(path)) == prompts.builtins()
    assert prompts.install(str(path)) == [], 'заготовка равна встроенным — применять нечего'


def test_override_is_applied_to_the_stage_and_nested_tables_work(tmp_path):
    import stages.arbitrate as A
    path = tmp_path / 'p.toml'
    path.write_text('[arbitrate.reader]\nsystem = """Читай внимательно."""\n', encoding='utf-8')
    assert prompts.install(str(path)) == ['arbitrate.reader.system']
    assert A.READER_SYS == 'Читай внимательно.'
    assert prompts.builtin('arbitrate.reader.system') == 'Читай внимательно.'


def test_unknown_name_missing_slot_and_non_string_are_refused(tmp_path):
    path = tmp_path / 'p.toml'
    path.write_text('"final.correct.systm" = "opa"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='неизвестный промпт'):
        prompts.parse(str(path))
    path.write_text('"final.correct.system" = "Вычитай фрагмент @CORPUS@ и верни JSON."\n', encoding='utf-8')
    with pytest.raises(ValueError, match='потерян слот'):
        prompts.parse(str(path))
    path.write_text('"final.recall.system" = 3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='должен быть строкой'):
        prompts.parse(str(path))


def test_without_a_file_nothing_is_touched():
    before = prompts.builtins()
    assert prompts.install('') == []
    assert prompts.builtins() == before
