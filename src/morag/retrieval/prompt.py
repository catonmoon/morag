"""Сборка системного промпта агента из именованных секций.

Единый источник истины для pipeline (`services/pipeline/morag_pipeline.py`,
собирает финальный текст) и console (`/api/retrieval/prompt-preview`, показывает
структуру с подсветкой настраиваемых частей). Раньше скелет жил в монолитной
строке внутри pipeline — консоль его не видела.

Промпт = последовательность `PromptSection`:
- `kind='fixed'`    — зашитое ядро (язык, сохранение сущностей);
- `kind='config'`   — настраивается через `retrieval.prompts.*` (есть `field`);
- `kind='tool'`     — методика тулов (из слоя тулов, `core.CORE_EXECUTION_METHODOLOGY`);
- `kind='km'`       — Knowledge Map (авто из корпуса).

`build_system_prompt()` склеивает `.text` всех секций — байт-в-байт как прежний
монолит при дефолтных значениях (проверяется тестом).
"""
from __future__ import annotations

from dataclasses import dataclass

from morag.config import DEFAULT_ANSWER_STYLE, DEFAULT_CORPUS_DESCRIPTION
from morag.retrieval.tools.core import (
    CORE_EXECUTION_METHODOLOGY,
    FIND_SECTION_OPTIONAL_POLICY,
    FIND_SECTION_REQUIRED_POLICY,
)


@dataclass
class PromptSection:
    label: str               # человекочитаемое имя секции (для консоли)
    kind: str                # 'fixed' | 'config' | 'tool' | 'km'
    text: str                # резолвнутый текст секции (идёт в промпт как есть)
    field: str = ''          # для kind='config' — имя ручки (corpus_description / …)
    edit_kind: str = 'readonly'  # как правится в консоли: 'text' | 'toggle' | 'readonly'
    description: str = ''    # короткое пояснение для hover-тултипа (что это)
    enabled: bool = True     # для editor-режима: выключенная секция (призрак, text='')


# ── Зашитые куски между плейсхолдерами (перенесены из _SYSTEM_PROMPT дословно) ──
_INTRO = (
    'Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
)

# Сохранение сущностей + повторный find_section + объединение результатов переехали
# в CORE_EXECUTION_METHODOLOGY (core.py) — это методика тулов, а не отдельное ядро.

# Блок «### 3. ПРОВЕРКА ПОЛНОТЫ» — включается, если completeness_check (дефолт on).
# Тот же тумблер гейтит рантайм diversity-nudge. Завершается '\n\n' — стык с answer_rules.
COMPLETENESS_CHECK_SECTION = (
    '### 3. ПРОВЕРКА ПОЛНОТЫ\n'
    'После поисков проверь:\n'
    '- Найдена ли информация из РАЗНЫХ разделов/документов?\n'
    '- ⚠️ КРАСНЫЙ ФЛАГ: если все результаты из одного раздела — '
    'почти наверняка ты пропустил информацию в других местах. Ищи шире.\n'
    '- Если какая-то грань вопроса не покрыта — ищи в оставшихся разделах.\n'
    '- Делай несколько поисков. Качество важнее скорости.\n\n'
)

_ADMIN_HEADER = '\n\n## Обязательные инструкции администратора\n'
_KM_HEADER = '\n\nСтруктура базы знаний (используй для навигации):\n'


def build_prompt_sections(
    *,
    corpus_description: str = '',
    require_find_section: bool = True,
    tool_methodology: str = CORE_EXECUTION_METHODOLOGY,
    completeness_check: bool = True,
    answer_style: str = '',
    admin_instructions: str = '',
    knowledge_map: str = '',
    editor: bool = False,
) -> list[PromptSection]:
    """Структура системного промпта (для сборки в pipeline и превью/редактора в консоли).

    Настраиваемые секции (`kind='config'`) подставляют значение или дефолт; `field`
    указывает ручку, `edit_kind` — как правится (text/toggle/readonly), `description`
    — пояснение для тултипа. Пустые секции (completeness off, нет admin/KM) обычно НЕ
    добавляются — но при `editor=True` выключенный completeness отдаётся «призраком»
    (`enabled=False`, `text=''`), чтобы консоль показала его и дала включить.
    `knowledge_map` — реальный текст карты (pipeline) или нота-плейсхолдер (превью).
    """
    role = corpus_description or DEFAULT_CORPUS_DESCRIPTION
    policy = FIND_SECTION_REQUIRED_POLICY if require_find_section else FIND_SECTION_OPTIONAL_POLICY
    sections: list[PromptSection] = [
        PromptSection(
            'Роль агента', 'config', role + ' ', field='corpus_description', edit_kind='text',
            description='Доменная роль агента: кто он и как отвечает. Заменяет дефолт.',
        ),
        PromptSection(
            'Язык и доступ к инструментам', 'fixed', _INTRO,
            description='Язык ответа и наличие инструментов — фиксированное ядро.',
        ),
        PromptSection(
            'Политика find_section (' + ('обязателен' if require_find_section else 'опционален') + ')',
            'config', policy, field='require_find_section', edit_kind='toggle',
            description='find_section обязателен (иерархичный корпус) или опционален '
                        '(плоский корпус, агрегатные/точечные вопросы). Кликни → к карточке '
                        'find_section (там переключатель required и описание).',
        ),
        PromptSection(
            'Методика тулов', 'tool', tool_methodology,
            description='Как пользоваться инструментами: сохранение сущностей, повторный '
                        'find_section, несколько search, section_ids/doc_ids, get_doc, шум. '
                        'Кликни → к карточкам инструментов.',
        ),
    ]
    if completeness_check or editor:
        sections.append(PromptSection(
            'Проверка полноты (### 3)', 'config',
            COMPLETENESS_CHECK_SECTION if completeness_check else '',
            field='completeness_check', edit_kind='toggle', enabled=completeness_check,
            description='Проверять, что ответ собран из разных разделов (+ runtime-подсказка). '
                        'Для юристов/подкаста, где ответ из одного места норма, — выключи.',
        ))
    sections.append(PromptSection(
        'Правила ответа', 'config', answer_style or DEFAULT_ANSWER_STYLE,
        field='answer_style', edit_kind='text',
        description='Стиль и правила ответа: кратко, формат, анти-конфабуляция, свежесть.',
    ))
    if admin_instructions:
        sections.append(PromptSection(
            'Инструкции администратора', 'config', _ADMIN_HEADER + admin_instructions,
            field='admin_instructions', edit_kind='text',
            description='Произвольные инструкции администратора (вклеиваются в хвост промпта).',
        ))
    if knowledge_map:
        sections.append(PromptSection(
            'Knowledge Map', 'km', _KM_HEADER + knowledge_map,
            description='Навигационная карта корпуса — авто, подставляется в рантайме.',
        ))
    return sections


def build_system_prompt(**kwargs) -> str:
    """Финальный текст системного промпта — склейка `.text` всех секций.
    Аргументы — как у build_prompt_sections."""
    return ''.join(s.text for s in build_prompt_sections(**kwargs))
