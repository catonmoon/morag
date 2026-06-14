"""Сборка системного промпта агента из именованных секций (WYSIWYG-модель).

Единый источник истины для pipeline (`services/pipeline/morag_pipeline.py` —
финальный текст) и console (`/api/retrieval/prompt-preview` — документ-редактор).

Промпт = последовательность `PromptSection`. КАЖДАЯ секция редактируема: у неё
стабильный `id`, дефолтный текст `default` и текущий `text` = оверрайд из
`section_overrides[id]` (если задан) либо дефолт. Зашитого-несменяемого нет —
только дефолты, которые можно вернуть («сбросить к дефолту» = убрать оверрайд).

`build_system_prompt()` склеивает `.text` всех секций — байт-в-байт как прежний
монолит при пустых оверрайдах (проверяется тестом).
"""
from __future__ import annotations

from dataclasses import dataclass

from morag.config import (
    ADMIN_HEADER,
    DEFAULT_ADMIN_INSTRUCTIONS,
    DEFAULT_ANSWER_STYLE,
    DEFAULT_CORPUS_DESCRIPTION,
)
from morag.retrieval.tools.core import (
    CORE_EXECUTION_METHODOLOGY,
    FIND_SECTION_OPTIONAL_POLICY,
    FIND_SECTION_REQUIRED_POLICY,
)


@dataclass
class PromptSection:
    id: str                  # стабильный ключ секции (ключ оверрайда)
    label: str               # человекочитаемое имя (для консоли)
    kind: str                # 'config' | 'tool' | 'km' (для цвета/группировки)
    text: str                # текущий текст = оверрайд или дефолт (идёт в промпт)
    default: str             # дефолтный текст (для reset + индикатора «изменено»)
    edit_kind: str           # как правится: 'text' | 'policy' | 'toggle' | 'readonly'
    description: str = ''     # пояснение для тултипа
    enabled: bool = True      # для editor: выключенная (completeness off) — призрак, text=''


# Язык + доступ к инструментам. Ведущий пробел — разделитель после «роли» (раньше
# жил как хвостовой пробел роли). Так роль-оверрайд = чистый текст роли.
_INTRO_DEFAULT = (
    ' Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
)

# Блок «### 3. ПРОВЕРКА ПОЛНОТЫ» — текст секции completeness (вкл/выкл — тумблер
# completeness_check; при выкл секции в промпте нет). Завершается '\n\n'.
COMPLETENESS_CHECK_SECTION = (
    '### 3. ПРОВЕРКА ПОЛНОТЫ\n'
    'После поисков проверь:\n'
    '- Найдена ли информация из РАЗНЫХ разделов/документов?\n'
    '- ⚠️ КРАСНЫЙ ФЛАГ: если все результаты из одного раздела — '
    'почти наверняка ты пропустил информацию в других местах. Ищи шире.\n'
    '- Если какая-то грань вопроса не покрыта — ищи в оставшихся разделах.\n'
    '- Делай несколько поисков. Качество важнее скорости.\n\n'
)

_KM_HEADER = '\n\nСтруктура базы знаний (используй для навигации):\n'

# Все редактируемые id (для миграции/валидации оверрайдов).
SECTION_IDS = (
    'role', 'intro', 'find_section_policy', 'tool_methodology',
    'completeness', 'answer_rules', 'admin',
)


def build_prompt_sections(
    *,
    section_overrides: dict[str, str] | None = None,
    require_find_section: bool = True,
    tool_methodology: str = CORE_EXECUTION_METHODOLOGY,
    completeness_check: bool = True,
    knowledge_map: str = '',
    editor: bool = False,
) -> list[PromptSection]:
    """Структура системного промпта. `section_overrides` (id → текст) перекрывает
    дефолты посекционно. `require_find_section`/`completeness_check` влияют на
    ДЕФОЛТЫ find_section-политики и наличие блока полноты. `knowledge_map` —
    реальный текст карты (pipeline) или нота-плейсхолдер (превью); km не редактируем.
    `editor=True` отдаёт выключённый completeness «призраком» (enabled=False, text='').
    """
    ov = section_overrides or {}

    def resolved(sid: str, default: str) -> str:
        return ov[sid] if sid in ov else default

    policy_default = FIND_SECTION_REQUIRED_POLICY if require_find_section else FIND_SECTION_OPTIONAL_POLICY
    admin_default = ADMIN_HEADER + DEFAULT_ADMIN_INSTRUCTIONS

    sections: list[PromptSection] = [
        PromptSection(
            'role', 'Роль агента', 'config',
            resolved('role', DEFAULT_CORPUS_DESCRIPTION), DEFAULT_CORPUS_DESCRIPTION, 'text',
            description='Доменная роль агента: кто он и как отвечает.',
        ),
        PromptSection(
            'intro', 'Язык и доступ к инструментам', 'config',
            resolved('intro', _INTRO_DEFAULT), _INTRO_DEFAULT, 'text',
            description='Язык ответа и упоминание инструментов.',
        ),
        PromptSection(
            'find_section_policy',
            'Политика find_section (' + ('обязателен' if require_find_section else 'опционален') + ')',
            'config', resolved('find_section_policy', policy_default), policy_default, 'policy',
            description='Когда и как звать find_section. Дефолт зависит от того, обязателен '
                        'ли find_section перед search (REQUIRED для иерархичного корпуса, '
                        'OPTIONAL для плоского). Правится как текст; пресеты и флаг — в редакторе.',
        ),
        PromptSection(
            'tool_methodology', 'Методика тулов', 'tool',
            resolved('tool_methodology', tool_methodology), tool_methodology, 'text',
            description='Как пользоваться инструментами: сохранение сущностей, повторный '
                        'find_section, несколько search, section_ids/doc_ids, get_doc, шум.',
        ),
    ]
    if completeness_check or editor:
        sections.append(PromptSection(
            'completeness', 'Проверка полноты (### 3)', 'config',
            resolved('completeness', COMPLETENESS_CHECK_SECTION) if completeness_check else '',
            COMPLETENESS_CHECK_SECTION, 'toggle', enabled=completeness_check,
            description='Проверять, что ответ собран из разных разделов (+ runtime-подсказка). '
                        'Тумблер вкл/выкл; для юристов/подкаста — выключи.',
        ))
    sections.append(PromptSection(
        'answer_rules', 'Правила ответа', 'config',
        resolved('answer_rules', DEFAULT_ANSWER_STYLE), DEFAULT_ANSWER_STYLE, 'text',
        description='Стиль и правила ответа: кратко, формат, анти-конфабуляция, свежесть.',
    ))
    sections.append(PromptSection(
        'admin', 'Инструкции администратора', 'config',
        resolved('admin', admin_default), admin_default, 'text',
        description='Произвольные инструкции администратора (хвост промпта).',
    ))
    if knowledge_map:
        sections.append(PromptSection(
            'km', 'Knowledge Map', 'km', _KM_HEADER + knowledge_map, '', 'readonly',
            description='Навигационная карта корпуса — авто, подставляется в рантайме.',
        ))
    return sections


def build_system_prompt(**kwargs) -> str:
    """Финальный текст системного промпта — склейка `.text` всех секций.
    Аргументы — как у build_prompt_sections."""
    return ''.join(s.text for s in build_prompt_sections(**kwargs))
