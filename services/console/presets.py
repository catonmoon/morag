"""Опинионированные пресеты для wizard'а Console UI.

Цель — сократить onboarding до 2-3 полей вместо целого config.example.yml.

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

def _common_llm_form_fields(*, default_name: str = '') -> list[PresetField]:
    """name + capabilities checkbox — общие для всех LLM-пресетов."""
    return [
        PresetField('name', 'Name (уникальное имя в пуле)',
                    default=default_name,
                    help='Используется для ссылок в indexing.llm/.vision. '
                         'Lowercase, без пробелов (a-z 0-9 _ -).'),
        PresetField('vision_capable', 'Vision-capable (multimodal)',
                    kind='checkbox', required=False, default=False,
                    help='Поставь галку если эта модель умеет обрабатывать картинки '
                         '(qwen2.5-vl, claude, gpt-4o). Тогда её можно использовать '
                         'для роли indexing.vision.'),
    ]


def _capabilities(form: dict[str, Any]) -> list[str]:
    """Из формы → list capabilities."""
    if form.get('vision_capable') in (True, 'true', 'on', '1'):
        return ['text', 'vision']
    return ['text']


def _build_grok(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': form.get('name') or 'grok',
        'base_url': 'https://api.x.ai/v1',
        'model': form.get('model', 'grok-4-1-fast-non-reasoning'),
        'api_key': form['api_key'],
        'capabilities': _capabilities(form),
        'context_window': 256000,
        'max_concurrent': 8,
    }


def _build_openrouter(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': form.get('name') or 'openrouter',
        'base_url': 'https://openrouter.ai/api/v1',
        'model': form['model'],
        'api_key': form['api_key'],
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window', 200000)),
        'max_concurrent': int(form.get('max_concurrent', 4)),
    }


def _build_ollama_llm(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': form.get('name') or 'ollama',
        'base_url': form.get('base_url', 'http://host.docker.internal:11434/v1'),
        'model': form['model'],
        'api_key': 'ollama',
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window', 32768)),
        'max_concurrent': int(form.get('max_concurrent', 1)),
        'enable_thinking': False,
    }


def _build_custom_llm(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'name': form['name'],
        'base_url': form['base_url'],
        'model': form['model'],
        'api_key': form['api_key'],
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window', 32768)),
    }
    if form.get('max_concurrent'):
        out['max_concurrent'] = int(form['max_concurrent'])
    return out


LLM_PRESETS: list[Preset] = [
    Preset(
        id='grok', name='Grok (xAI)', target='llm',
        description='xAI Grok. Большой контекст (256K), быстрый, дешёвый.',
        fields=_common_llm_form_fields(default_name='grok') + [
            PresetField('model', 'Модель', default='grok-4-1-fast-non-reasoning'),
            PresetField('api_key', 'API key', kind='password', placeholder='xai-...'),
        ],
        build=_build_grok,
    ),
    Preset(
        id='openrouter', name='OpenRouter', target='llm',
        description='OpenRouter — десятки моделей через один API.',
        fields=_common_llm_form_fields(default_name='openrouter') + [
            PresetField('model', 'Модель', placeholder='anthropic/claude-haiku-4.5'),
            PresetField('api_key', 'API key', kind='password', placeholder='sk-or-...'),
            PresetField('context_window', 'Context window', kind='number',
                        default=200000, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        default=4, required=False),
        ],
        build=_build_openrouter,
    ),
    Preset(
        id='ollama', name='Ollama (локальный)', target='llm',
        description='Локальный Ollama-сервер. Подходит для приватных деплоев.',
        fields=_common_llm_form_fields(default_name='ollama') + [
            PresetField('model', 'Модель', placeholder='qwen3:4b',
                        help='Имя модели как в `ollama list`.'),
            PresetField('base_url', 'Base URL',
                        default='http://host.docker.internal:11434/v1', required=False),
            PresetField('context_window', 'Context window', kind='number',
                        default=32768, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        default=1, required=False),
        ],
        build=_build_ollama_llm,
    ),
    Preset(
        id='custom', name='Custom (vLLM / OpenAI-compat)', target='llm',
        description='Любой OpenAI-совместимый endpoint. Все параметры — вручную.',
        fields=_common_llm_form_fields() + [
            PresetField('base_url', 'Base URL', placeholder='https://...'),
            PresetField('model', 'Модель'),
            PresetField('api_key', 'API key', kind='password'),
            PresetField('context_window', 'Context window', kind='number',
                        default=32768, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        required=False),
        ],
        build=_build_custom_llm,
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


def _build_confluence_cloud(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'kind': 'confluence',
        'name': form.get('name') or 'main',
        'url': form['url'],
        'username': form['username'],
        'api_token': form['api_token'],
    }
    if form.get('spaces'):
        out['spaces'] = [s.strip() for s in form['spaces'].split(',') if s.strip()]
    if form.get('ancestor_ids'):
        out['ancestor_ids'] = [s.strip() for s in form['ancestor_ids'].split(',') if s.strip()]
    if form.get('attachments_enabled') in (True, 'true', 'on', '1'):
        out['attachments'] = {'enabled': True}
    return out


def _build_confluence_onprem(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'kind': 'confluence',
        'name': form.get('name') or 'main',
        'url': form['url'],
        'username': form['username'],
        'password': form['password'],
    }
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
        id='confluence-cloud', name='Confluence (Cloud)', target='source',
        description='Atlassian Cloud — auth через api_token.',
        fields=[
            PresetField('name', 'Name', default='main', required=False,
                        help='Уникальный id (например "corp", "vendor").'),
            PresetField('url', 'URL', placeholder='https://your-company.atlassian.net'),
            PresetField('username', 'Username (email)', placeholder='you@example.com'),
            PresetField('api_token', 'API token', kind='password',
                        help='id.atlassian.com/manage-profile/security/api-tokens'),
            PresetField('spaces', 'Spaces (comma-separated)', required=False,
                        placeholder='DOCS, ENG'),
            PresetField('ancestor_ids', 'Ancestor IDs (comma-separated)', required=False,
                        placeholder='123456, 789012',
                        help='Только потомки этих страниц. Приоритет над spaces.'),
            PresetField('attachments_enabled', 'Index PDF attachments',
                        kind='checkbox', required=False, default=False),
        ],
        build=_build_confluence_cloud,
    ),
    Preset(
        id='confluence-onprem', name='Confluence (on-premise)', target='source',
        description='Confluence Server / Data Center — auth через password.',
        fields=[
            PresetField('name', 'Name', default='main', required=False),
            PresetField('url', 'URL', placeholder='https://confluence.your-company.com'),
            PresetField('username', 'Username'),
            PresetField('password', 'Password', kind='password'),
            PresetField('spaces', 'Spaces (comma-separated)', required=False),
            PresetField('ancestor_ids', 'Ancestor IDs (comma-separated)', required=False),
            PresetField('attachments_enabled', 'Index PDF attachments',
                        kind='checkbox', required=False, default=False),
        ],
        build=_build_confluence_onprem,
    ),
    Preset(
        id='jira', name='Jira (on-premise)', target='source',
        description='Только on-prem (password). Задачи берутся по ссылкам в '
                    'уже-проиндексированных доках (Confluence/Local).',
        fields=[
            PresetField('name', 'Name', default='main', required=False),
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
