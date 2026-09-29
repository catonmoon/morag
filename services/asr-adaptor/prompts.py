"""Реестр промптов стадий и файл переопределений: `ASR_PROMPTS=<путь>.toml`.

Промпты стадий — константы модулей `stages/*` (`_SYS`, `_CORRECT_SYS`, `READER_SYS`…), и стадии
читают их В МОМЕНТ ВЫЗОВА. Поэтому переопределение — это `setattr` при старте, без правки стадий:
без файла константы не тронуты, и любой чужой профиль работает как раньше байт в байт.

Зачем файл, а не env: промпт — многострочный текст с кавычками и JSON-скобками, и домен обязан
уметь его ПРАВИТЬ, а не подставлять фрагменты. Доменные тексты лежат у домена (в приватном
репозитории рядом с профилем корпуса), здесь — только generic-умолчания. Формат TOML: многострочные
литеральные строки ('''…''') не трогают ни кавычки, ни обратные слэши; читает `tomllib` из stdlib.

    python prompts.py --dump > asr-prompts.toml     # встроенные тексты как заготовка файла

⚠️ Имя промпта — стабильный ключ (`final.correct.system`), а не имя константы: константы —
внутренняя кухня, файл домена не должен ломаться от переименования. Неизвестное имя в файле —
отказ при старте: опечатка иначе молчала бы, и человек тюнил бы промпт, который не применяется.
⚠️ У промпта правки есть СЛОТЫ (`@CORPUS@`, `@EXAMPLE@`, `@NAMERULE@`), которые стадия подставляет
`.replace()`; переопределение обязано их сохранить — иначе описание корпуса и правило про имена
молча выпадут из промпта.
"""
from __future__ import annotations

import importlib
import sys

# имя промпта → (модуль, константа). Порядок — порядок в `--dump`.
REGISTRY: dict[str, tuple[str, str]] = {
    'glossary.system': ('stages.glossary', '_SYS'),
    'glossary.selflabel_suffix': ('stages.glossary', '_SYS_SELFLABEL'),
    'hints.seed.system': ('stages.hints', '_SEED_SYS'),
    'final.correct.system': ('stages.final_round', '_CORRECT_SYS'),
    'final.correct.name_rule': ('stages.final_round', '_NAME_RULE'),
    'final.recall.system': ('stages.final_round', '_RECALL_SYS'),
    'final.doc.system': ('stages.final_round', '_DOC_SYS'),
    'final.doc_merge.system': ('stages.final_round', '_DOC_MERGE_SYS'),
    'arbitrate.reader.system': ('stages.arbitrate', 'READER_SYS'),
    'namer.intro.system': ('stages.namer', '_SYS'),
    'namer.guests.system': ('stages.namer', '_GUESTS_SYS'),
    'policy.orchestrator.system': ('graph.policy', 'ORCHESTRATOR_SYS'),
    'editor.system': ('graph.editor', 'EDITOR_SYS'),
}

# Слоты, которые переопределение обязано сохранить.
SLOTS: dict[str, tuple[str, ...]] = {
    'final.correct.system': ('@CORPUS@', '@EXAMPLE@', '@NAMERULE@'),
    'namer.intro.system': ('@CORPUS@',),
    'namer.guests.system': ('@CORPUS@',),
    'editor.system': ('@CORPUS@',),
}


def _target(name: str):
    mod, attr = REGISTRY[name]
    return importlib.import_module(mod), attr


def builtin(name: str) -> str:
    """Текст промпта, каким он сейчас стоит у стадии (после переопределений — переопределённый)."""
    m, attr = _target(name)
    return getattr(m, attr)


def builtins() -> dict[str, str]:
    return {name: builtin(name) for name in REGISTRY}


def _flatten(d: dict, prefix: str = '') -> dict[str, object]:
    """`[final.correct] system = …` и `"final.correct.system" = …` — одно и то же."""
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def parse(path: str) -> dict[str, str]:
    """Прочитать и проверить файл переопределений. Ошибка — исключение с причиной, не молчание."""
    import tomllib  # noqa: PLC0415 — stdlib 3.11+, адаптер живёт на 3.12

    with open(path, 'rb') as fh:
        raw = _flatten(tomllib.load(fh))
    out: dict[str, str] = {}
    for name, text in raw.items():
        if name not in REGISTRY:
            raise ValueError(f'{path}: неизвестный промпт «{name}»; известны: {", ".join(REGISTRY)}')
        if not isinstance(text, str):
            raise ValueError(f'{path}: промпт «{name}» должен быть строкой, а не {type(text).__name__}')
        missing = [s for s in SLOTS.get(name, ()) if s not in text]
        if missing:
            raise ValueError(f'{path}: в промпте «{name}» потерян слот {", ".join(missing)} — '
                             f'стадия подставляет в него описание корпуса и правила')
        out[name] = text
    return out


def apply(overrides: dict[str, str]) -> list[str]:
    """Подменить константы стадий. Возвращает имена применённых промптов."""
    applied = []
    for name, text in overrides.items():
        m, attr = _target(name)
        if getattr(m, attr) != text:
            setattr(m, attr, text)
            applied.append(name)
    return applied


def install(path: str) -> list[str]:
    """`ASR_PROMPTS` пуст — ничего не делать; иначе прочитать, проверить и применить."""
    if not path:
        return []
    return apply(parse(path))


def _toml_string(text: str) -> str:
    """Многострочная литеральная строка: ни кавычки, ни слэши не экранируются. Текст с `'''`
    внутри пришлось бы экранировать — таких промптов нет, и это проверяется явно."""
    if "'''" in text:
        raise ValueError("промпт содержит ''' — литеральной строкой TOML его не записать")
    return "'''\n" + text + "'''"


def dump() -> str:
    lines = ['# Промпты стадий распознавания — заготовка файла переопределений (ASR_PROMPTS=<этот файл>).',
             '# Ключ — имя промпта из реестра prompts.py; значение — текст целиком. Убери то, что не меняешь:',
             '# промпт, которого в файле нет, остаётся встроенным. Слоты @CORPUS@ / @EXAMPLE@ / @NAMERULE@',
             '# подставляет стадия — их надо сохранить.', '']
    for name in REGISTRY:
        lines.append(f'"{name}" = {_toml_string(builtin(name))}')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--dump' in sys.argv:
        sys.stdout.write(dump())
    else:
        sys.stderr.write('использование: python prompts.py --dump > asr-prompts.toml\n')
        sys.exit(2)
