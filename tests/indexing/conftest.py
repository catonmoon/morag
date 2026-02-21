from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


@pytest.fixture
def llm_overview_md() -> str:
    return (DATA_DIR / 'llm_overview.md').read_text(encoding='utf-8')


@pytest.fixture
def simple_md_with_headers() -> str:
    return """\
# Раздел первый

Текст первого раздела. Здесь описывается тема A.

## Подраздел 1.1

Детали подраздела 1.1.

# Раздел второй

Текст второго раздела. Здесь описывается тема B.

## Подраздел 2.1

Детали подраздела 2.1.
"""


@pytest.fixture
def md_with_large_table() -> str:
    rows = '\n'.join(f'| Строка {i} | Значение {i} | Описание {i} |' for i in range(1, 51))
    return f"""\
Текст перед таблицей.

| Название | Значение | Описание |
|---|---|---|
{rows}

Текст после таблицы.
"""


@pytest.fixture
def md_no_headers() -> str:
    return """\
Это обычный текст без заголовков.
Просто несколько предложений подряд.
Никаких заголовков здесь нет.
"""
