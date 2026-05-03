"""Опинионированные пресеты для wizard'а Console UI.

Цель — сократить onboarding до 2-3 полей вместо ручной правки config.yml.

После Stage 6 refactor (ADR-0012):
- LLM-presets выдают snippet формы `{name, base_url, model, api_key, capabilities, ...}`
  — добавляются в `llms[]` через add_to_pool().
- Source-presets выдают `{kind, name, ...source-fields...}` — добавляются в `sources[]`.
- Embedder больше не отдельный preset (один dense + sparse, не пул).

Apply-логика (services/console/routes/presets.py): list-append/replace by name,
не deep-merge top-level dict (т.к. lists в overlay перезаписываются целиком).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


PresetTarget = Literal['llm', 'source']


@dataclass(frozen=True)
class PresetField:
    """Описание одного поля формы пресета."""
    name: str
    label: str
    kind: Literal['text', 'password', 'number', 'checkbox'] = 'text'
    required: bool = True
    default: str | int | bool | None = None
    placeholder: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class Preset:
    """Шаблон для добавления одного инстанса (LLM или source) в pool/list."""
    id: str
    name: str
    target: PresetTarget
    fields: list[PresetField]
    build: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ''


# ---------------------------------------------------------------------------
# LLM presets — выдают одну entry для llms[] pool
# ---------------------------------------------------------------------------

def _capabilities(form: dict[str, Any]) -> list[str]:
    """Из формы → list capabilities."""
    if form.get('vision_capable') in (True, 'true', 'on', '1'):
        return ['text', 'vision']
    return ['text']


def _build_openai_compatible(form: dict[str, Any]) -> dict[str, Any]:
    """Любой OpenAI-совместимый endpoint: Grok, OpenRouter, vLLM, OpenAI."""
    out = {
        'name': form.get('name') or 'main',
        'base_url': form['base_url'],
        'model': form['model'],
        'api_key': form['api_key'],
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window') or 32768),
        'max_concurrent': int(form.get('max_concurrent') or 4),
    }
    return out


def _build_ollama_llm(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': form.get('name') or 'ollama',
        'base_url': form.get('base_url') or 'http://host.docker.internal:11434/v1',
        'model': form['model'],
        'api_key': 'ollama',                # Ollama игнорирует, но SDK требует
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window') or 32768),
        'max_concurrent': int(form.get('max_concurrent') or 1),
        'enable_thinking': False,           # Ollama qwen-модели думают по умолчанию — выключаем
    }


# Общие поля для обоих LLM-пресетов (name + vision-capable)
def _llm_common_fields(default_name: str = '') -> list[PresetField]:
    return [
        PresetField('name', 'Имя (уникальное в пуле)',
                    default=default_name,
                    help='Используется для ссылок в indexing.llm/.vision. '
                         'Lowercase, без пробелов (a-z 0-9 _ -).'),
        PresetField('vision_capable', 'Vision-capable (multimodal)',
                    kind='checkbox', required=False, default=False,
                    help='Отметьте, если модель умеет обрабатывать изображения '
                         '(qwen3.5, claude, gpt-4o, и т.п.). Только такие LLM '
                         'можно назначать на роль indexing.vision.'),
    ]


LLM_PRESETS: list[Preset] = [
    Preset(
        id='openai-compatible', name='OpenAI-compatible', target='llm',
        description='Любой OpenAI-совместимый endpoint: Grok, OpenRouter, vLLM, '
                    'OpenAI, Together, и др. В будущем — отдельные пресеты '
                    'для известных провайдеров с проверенными настройками.',
        fields=_llm_common_fields(default_name='main') + [
            PresetField('base_url', 'Base URL',
                        placeholder='https://api.x.ai/v1'),
            PresetField('model', 'Модель',
                        placeholder='grok-4-1-fast-non-reasoning'),
            PresetField('api_key', 'API key', kind='password',
                        placeholder='xai-... / sk-or-... / sk-...'),
            PresetField('context_window', 'Context window (токенов)',
                        kind='number', default=32768, required=False,
                        help='Размер окна модели. Для Grok ≈ 256000, '
                             'для большинства Claude/GPT ≈ 200000.'),
            PresetField('max_concurrent', 'Max concurrent (потолок параллельных запросов)',
                        kind='number', default=4, required=False,
                        help='Защита от перегрузки провайдера. 4-8 для облака, '
                             '1-2 для слабых vLLM-серверов.'),
        ],
        build=_build_openai_compatible,
    ),
    Preset(
        id='ollama', name='Ollama (локальный)', target='llm',
        description='Локальный Ollama-сервер. По умолчанию max_concurrent=1 — '
                    'Ollama сериализует запросы. Thinking-режим выключается '
                    'автоматически (для qwen3, которые думают по умолчанию).',
        fields=_llm_common_fields(default_name='ollama') + [
            PresetField('model', 'Модель', placeholder='qwen3.5:9b',
                        help='Имя модели как в выводе `ollama list`.'),
            PresetField('base_url', 'Base URL',
                        default='http://host.docker.internal:11434/v1', required=False,
                        help='Изнутри docker-compose — host.docker.internal. '
                             'Если консоль локально — http://localhost:11434/v1.'),
            PresetField('context_window', 'Context window (токенов)',
                        kind='number', default=32768, required=False),
            PresetField('max_concurrent', 'Max concurrent (потолок параллельных запросов)',
                        kind='number', default=1, required=False,
                        help='Ollama сериализует запросы — обычно 1. '
                             'Если CPU/GPU не даст просесть — можно повысить.'),
        ],
        build=_build_ollama_llm,
    ),
]


# ---------------------------------------------------------------------------
# Source presets — выдают одну entry для sources[] list
# ---------------------------------------------------------------------------

def _build_local_source(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'kind': 'local',
        'name': form.get('name') or 'docs',
        'path': form['path'],
    }


def _build_confluence(form: dict[str, Any]) -> dict[str, Any]:
    """Универсальный Confluence (Cloud или on-premise).

    Auth: одно из двух — `password` (on-prem) или `api_token` (Cloud).
    Pydantic-валидатор на уровне ConfluenceSourceConfig потребует ровно одно из них.
    """
    out = {
        'kind': 'confluence',
        'name': form.get('name') or 'main',
        'url': form['url'],
        'username': form['username'],
    }
    # Auth: api_token приоритетнее (если юзер заполнил оба, считаем что Cloud)
    if form.get('api_token'):
        out['api_token'] = form['api_token']
    elif form.get('password'):
        out['password'] = form['password']
    if form.get('spaces'):
        out['spaces'] = [s.strip() for s in form['spaces'].split(',') if s.strip()]
    if form.get('ancestor_ids'):
        out['ancestor_ids'] = [s.strip() for s in form['ancestor_ids'].split(',') if s.strip()]
    if form.get('attachments_enabled') in (True, 'true', 'on', '1'):
        out['attachments'] = {'enabled': True}
    return out


def _build_jira(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'kind': 'jira',
        'name': form.get('name') or 'main',
        'url': form['url'],
        'username': form['username'],
        'password': form['password'],
    }


SOURCE_PRESETS: list[Preset] = [
    Preset(
        id='local', name='Local folder', target='source',
        description='Markdown и PDF из локальной директории.',
        fields=[
            PresetField('name', 'Name', default='docs', required=False,
                        help='Уникальный id среди local-инстансов. Lowercase.'),
            PresetField('path', 'Path', placeholder='data/',
                        help='Путь внутри контейнера. По умолчанию ./data/.'),
        ],
        build=_build_local_source,
    ),
    Preset(
        id='confluence', name='Confluence', target='source',
        description='Atlassian Confluence — Cloud или on-premise. '
                    'Заполните либо API token (Cloud), либо password (on-prem).',
        fields=[
            PresetField('name', 'Имя', default='main', required=False,
                        help='Уникальный id (например "corp", "vendor").'),
            PresetField('url', 'URL',
                        placeholder='https://your-company.atlassian.net'
                                    ' или https://confluence.your-company.com'),
            PresetField('username', 'Username',
                        placeholder='email (Cloud) или логин (on-prem)'),
            PresetField('api_token', 'API token (для Cloud)',
                        kind='password', required=False,
                        help='Для atlassian.net: id.atlassian.com/manage-profile/security/api-tokens'),
            PresetField('password', 'Password (для on-premise)',
                        kind='password', required=False,
                        help='Для self-hosted Confluence Server / Data Center.'),
            PresetField('spaces', 'Spaces (через запятую)', required=False,
                        placeholder='DOCS, ENG'),
            PresetField('ancestor_ids', 'Ancestor IDs (через запятую)', required=False,
                        placeholder='123456, 789012',
                        help='Только потомки этих страниц. Приоритет над spaces.'),
            PresetField('attachments_enabled', 'Индексировать PDF-вложения',
                        kind='checkbox', required=False, default=False),
        ],
        build=_build_confluence,
    ),
    Preset(
        id='jira', name='Jira', target='source',
        description='Jira (on-premise). Задачи берутся по ссылкам в '
                    'уже-проиндексированных документах (Confluence/Local).',
        fields=[
            PresetField('name', 'Имя', default='main', required=False),
            PresetField('url', 'URL', placeholder='https://jira.your-company.com'),
            PresetField('username', 'Username'),
            PresetField('password', 'Password', kind='password'),
        ],
        build=_build_jira,
    ),
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

ALL_PRESETS: list[Preset] = LLM_PRESETS + SOURCE_PRESETS


def find_preset(target: PresetTarget, preset_id: str) -> Preset:
    for p in ALL_PRESETS:
        if p.target == target and p.id == preset_id:
            return p
    raise KeyError(f'No preset found: target={target}, id={preset_id}')


def apply_preset(target: PresetTarget, preset_id: str, form: dict[str, Any]) -> dict[str, Any]:
    """Собрать одну entry для добавления в pool/list.

    Возвращает item — НЕ обёрнутый в ключ. Caller (apply route) сам решает
    в какой список добавлять (`llms` или `sources`) на основе target.
    """
    preset = find_preset(target, preset_id)
    return preset.build(form)


def serialize_preset(p: Preset) -> dict[str, Any]:
    return {
        'id': p.id,
        'name': p.name,
        'target': p.target,
        'description': p.description,
        'fields': [
            {
                'name': f.name,
                'label': f.label,
                'kind': f.kind,
                'required': f.required,
                'default': f.default,
                'placeholder': f.placeholder,
                'help': f.help,
            }
            for f in p.fields
        ],
    }
