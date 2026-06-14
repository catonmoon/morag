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
    label: str          # человекочитаемое имя секции (для консоли)
    kind: str           # 'fixed' | 'config' | 'tool' | 'km'
    text: str           # резолвнутый текст секции (идёт в промпт как есть)
    field: str = ''     # для kind='config' — имя ручки (corpus_description / answer_style / …)


# ── Зашитые куски между плейсхолдерами (перенесены из _SYSTEM_PROMPT дословно) ──
_INTRO = (
    'Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
)

_ENTITY_PRESERVATION = (
    'СОХРАНЯЙ имена, фамилии, названия, ID, специфические термины — это самые сильные '
    'различающие сигналы и их нельзя обобщать.\n'
    'Второй find_section вызывай ТОЛЬКО если первый не дал релевантных секций '
    '(пустой результат, либо найденные разделы явно мимо темы). При втором — '
    'варьируй УГОЛ вопроса (другой аспект, переставленные слова, синонимы тех же '
    'терминов), НО не подменяй конкретные сущности на категории-абстракции.\n'
    'ЗАПРЕЩЕНО:\n'
    '  - «Иван Петров» → «сотрудник Иван» (выбросил фамилию, добавил категорию)\n'
    '  - «TASK-123» → «задача разработки» (выбросил конкретный ID)\n'
    '  - «конкретное название» → «общая категория» (добавил категорию из общей эрудиции)\n'
    '⚠️ Если делал несколько find_section — **ОБЪЕДИНЯЙ** результаты, не замещай. '
    'В search передавай union section_ids и doc_ids от всех вызовов.\n\n'
)

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
) -> list[PromptSection]:
    """Структура системного промпта (для сборки в pipeline и превью в консоли).

    Настраиваемые секции (`kind='config'`) подставляют значение или дефолт;
    `field` указывает ручку. Пустые/выключенные секции (completeness off, нет
    admin/KM) просто не добавляются. `knowledge_map` — реальный текст карты
    (pipeline) или короткая нота-плейсхолдер (превью).
    """
    role = corpus_description or DEFAULT_CORPUS_DESCRIPTION
    policy = FIND_SECTION_REQUIRED_POLICY if require_find_section else FIND_SECTION_OPTIONAL_POLICY
    sections: list[PromptSection] = [
        PromptSection('Роль агента', 'config', role + ' ', field='corpus_description'),
        PromptSection('Язык и доступ к инструментам', 'fixed', _INTRO),
        PromptSection(
            'Политика find_section (' + ('обязателен' if require_find_section else 'опционален') + ')',
            'config', policy, field='require_find_section',
        ),
        PromptSection('Сохранение сущностей', 'fixed', _ENTITY_PRESERVATION),
        PromptSection('Методика тулов (### 2)', 'tool', tool_methodology),
    ]
    if completeness_check:
        sections.append(PromptSection(
            'Проверка полноты (### 3)', 'config', COMPLETENESS_CHECK_SECTION,
            field='completeness_check',
        ))
    sections.append(PromptSection(
        'Правила ответа', 'config', answer_style or DEFAULT_ANSWER_STYLE, field='answer_style',
    ))
    if admin_instructions:
        sections.append(PromptSection(
            'Инструкции администратора', 'config', _ADMIN_HEADER + admin_instructions,
            field='admin_instructions',
        ))
    if knowledge_map:
        sections.append(PromptSection(
            'Knowledge Map', 'km', _KM_HEADER + knowledge_map,
        ))
    return sections


def build_system_prompt(**kwargs) -> str:
    """Финальный текст системного промпта — склейка `.text` всех секций.
    Аргументы — как у build_prompt_sections."""
    return ''.join(s.text for s in build_prompt_sections(**kwargs))
