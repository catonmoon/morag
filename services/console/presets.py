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

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from .config_io import make_commented_id_list


PresetTarget = Literal['llm', 'source', 'embedder']


# ---------------------------------------------------------------------------
# Name auto-generation (заменяет ручной ввод name в форме)
# ---------------------------------------------------------------------------

# Generic technical-host префиксы которые НЕ несут смысла для name —
# пропускаем при выборе subdomain'а.
_GENERIC_HOST_PARTS = frozenset({'www', 'api', 'confluence', 'jira', 'wiki', 'docs'})

# Известные облачные LLM-провайдеры по hostname (полный или substring matching).
_KNOWN_LLM_PROVIDERS: list[tuple[str, str]] = [
    ('x.ai', 'grok'),
    ('openrouter.ai', 'openrouter'),
    ('openai.com', 'openai'),
    ('anthropic.com', 'claude'),
    ('together.ai', 'together'),
    ('together.xyz', 'together'),
    ('deepinfra.com', 'deepinfra'),
    ('mistral.ai', 'mistral'),
    ('groq.com', 'groq'),
    ('host.docker.internal', 'ollama'),       # docker-compose default
    ('localhost', 'ollama'),                   # native default
]


def _sanitize_name(s: str) -> str:
    """Lowercase, оставить [a-z0-9_-], схлопнуть/обрезать дефисы."""
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9_-]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s


def _extract_subdomain(url: str) -> str:
    """Из `https://corp.atlassian.net/...` → `corp`. Skip generic-prefixes."""
    if not url:
        return ''
    try:
        host = urlparse(url).hostname or ''
    except Exception:
        return ''
    if not host:
        return ''
    parts = host.split('.')
    for p in parts:
        if p and p not in _GENERIC_HOST_PARTS and len(p) >= 2:
            return _sanitize_name(p)
    return _sanitize_name(parts[0]) if parts else ''


def suggest_source_name(kind: str, form: dict[str, Any]) -> str:
    """Базовое имя источника (без collision-suffix). Caller добавит -2/-3 если нужно."""
    if kind == 'local':
        return 'doc'                          # singleton; путь зашит в /app/data
    if kind in ('confluence', 'jira'):
        sub = _extract_subdomain(form.get('url') or '')
        return sub or kind                    # fallback на kind если URL нет
    return kind or 'main'


def suggest_llm_name(form: dict[str, Any]) -> str:
    """Базовое имя LLM. Provider hint для известных + модель-короткая."""
    base_url = form.get('base_url') or ''
    host = (urlparse(base_url).hostname or '').lower() if base_url else ''
    provider = ''
    for hint, name in _KNOWN_LLM_PROVIDERS:
        if hint in host:
            provider = name
            break
    if not provider:
        provider = _extract_subdomain(base_url) or 'main'
    return provider


def unique_name(base: str, existing: set[str]) -> str:
    """Если base уже занят — добавить -2/-3/...; иначе вернуть как есть."""
    if not base:
        base = 'main'
    if base not in existing:
        return base
    i = 2
    while f'{base}-{i}' in existing:
        i += 1
    return f'{base}-{i}'


@dataclass(frozen=True)
class PresetField:
    """Описание одного поля формы пресета.

    kind='chips' — список ID-чипов (page-id для Confluence). UI парсит вход
    с любыми разделителями, рендерит как теги, лениво подтягивает названия
    через resolver_endpoint. Значение в form: list[{id, comment}].
    variant='danger' — визуально красный (для skip/exclude-семантики).
    """
    name: str
    label: str
    kind: Literal['text', 'password', 'number', 'checkbox', 'chips'] = 'text'
    required: bool = True
    default: str | int | bool | None = None
    placeholder: str | None = None
    help: str | None = None
    variant: Literal['default', 'danger'] | None = None
    # Для kind='chips': URL endpoint'а для resolve названий по ID.
    # Получает {url, username, secret_key, secret_value, ids[]} → {id: {title, path, error}}.
    resolver_endpoint: str | None = None


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
    """Любой OpenAI-совместимый endpoint: Grok, OpenRouter, vLLM, OpenAI.

    Имя генерируется автоматически (apply route добавит -2/-3 при коллизии).
    api_key может быть пустым — это валидно при Edit-режиме (бэкенд подтянет
    существующий секрет из текущего конфига). См. routes/presets.py::apply.
    """
    out = {
        'name': form.get('name') or suggest_llm_name(form),
        'base_url': form['base_url'],
        'model': form['model'],
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window') or 32768),
        'max_concurrent': int(form.get('max_concurrent') or 4),
        # Thinking всегда выключаем по умолчанию — vLLM/qwen-семейство по дефолту
        # думают, без явного флага CoT улетает в reasoning-поле и засирает payload
        # (инцидент 2026-05). Юзер может включить в конфиге вручную если нужно.
        'enable_thinking': False,
    }
    if form.get('api_key'):
        out['api_key'] = form['api_key']
    return out


def _build_ollama_llm(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': form.get('name') or suggest_llm_name(
            {**form, 'base_url': form.get('base_url') or 'http://host.docker.internal:11434/v1'},
        ),
        'base_url': form.get('base_url') or 'http://host.docker.internal:11434/v1',
        'model': form['model'],
        'api_key': 'ollama',                # Ollama игнорирует, но SDK требует
        'capabilities': _capabilities(form),
        'context_window': int(form.get('context_window') or 32768),
        'max_concurrent': int(form.get('max_concurrent') or 1),
        'enable_thinking': False,           # Ollama qwen-модели думают по умолчанию — выключаем
    }


# Общие поля для обоих LLM-пресетов (только vision-флаг — имя генерится auto)
def _llm_common_fields() -> list[PresetField]:
    return [
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
        fields=_llm_common_fields() + [
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
        fields=_llm_common_fields() + [
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

# Жёсткий путь для local-source: всегда /app/data в контейнере (см. docker-compose.yml).
# Если юзеру нужна другая папка — править docker-compose.yml volume и config.local.yml.
LOCAL_SOURCE_PATH = '/app/data'


def _build_local_source(form: dict[str, Any]) -> dict[str, Any]:
    """Singleton: имя всегда 'doc', путь зашит в /app/data."""
    return {
        'kind': 'local',
        'name': 'doc',
        'path': LOCAL_SOURCE_PATH,
    }


def _build_confluence(form: dict[str, Any]) -> dict[str, Any]:
    """Универсальный Confluence (Cloud или on-premise).

    Auth: одно из двух — `password` (on-prem) или `api_token` (Cloud).
    Pydantic-валидатор на уровне ConfluenceSourceConfig потребует ровно одно из них.
    Секреты опциональны — при пустом значении бэкенд подтянет существующий.

    Имя инстанса генерируется из subdomain URL (apply route добавит -2/-3 при коллизии).

    ancestor_ids/skip_ancestor_ids приходят из UI как chip-list:
    [{id, comment}, ...] — конвертируем в CommentedSeq, чтобы YAML overlay
    содержал inline-комменты с breadcrumb-путём страницы.
    """
    out: dict[str, Any] = {
        'kind': 'confluence',
        'name': form.get('name') or suggest_source_name('confluence', form),
        'url': form['url'],
        'username': form['username'],
    }
    # Auth: api_token приоритетнее (если юзер заполнил оба, считаем что Cloud)
    if form.get('api_token'):
        out['api_token'] = form['api_token']
    elif form.get('password'):
        out['password'] = form['password']
    if form.get('spaces'):
        out['spaces'] = _parse_csv_field(form['spaces'])
    chips = _parse_chips(form.get('ancestor_ids'))
    if chips:
        out['ancestor_ids'] = make_commented_id_list(chips)
    skip_chips = _parse_chips(form.get('skip_ancestor_ids'))
    if skip_chips:
        out['skip_ancestor_ids'] = make_commented_id_list(skip_chips)
    if form.get('attachments_enabled') in (True, 'true', 'on', '1'):
        out['attachments'] = {'enabled': True}
    return out


def _parse_csv_field(value: Any) -> list[str]:
    """Универсальный парсер: list[str] возвращает as-is (фильтруя пустые),
    str разбивает по запятой/пробелу/переносам.
    """
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s).strip()]
    if isinstance(value, str):
        # любые из ',', ';', whitespace в качестве разделителя
        import re
        return [s.strip() for s in re.split(r'[,;\s]+', value) if s.strip()]
    return []


def _parse_chips(value: Any) -> list[dict[str, str]]:
    """UI шлёт chips как list[{id, comment}]. Старый формат (csv-строка) тоже
    поддерживается на случай ручной правки — там comment будет пустой.
    """
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict) and item.get('id'):
                result.append({
                    'id': str(item['id']).strip(),
                    'comment': str(item.get('comment') or '').strip(),
                })
            elif isinstance(item, (str, int)):
                s = str(item).strip()
                if s:
                    result.append({'id': s, 'comment': ''})
        return result
    if isinstance(value, str):
        return [{'id': s, 'comment': ''} for s in _parse_csv_field(value)]
    return []


def _build_jira(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'kind': 'jira',
        'name': form.get('name') or suggest_source_name('jira', form),
        'url': form['url'],
        'username': form['username'],
    }
    if form.get('password'):
        out['password'] = form['password']
    return out


SOURCE_PRESETS: list[Preset] = [
    Preset(
        id='local', name='Локальная папка', target='source',
        description='Markdown и PDF из локальной папки. Путь зашит на /app/data — '
                    'это volume в docker-compose, на хосте — ./data/. '
                    'Положите файлы туда; для смены пути правьте docker-compose.yml.',
        fields=[],                            # singleton — никаких полей
        build=_build_local_source,
    ),
    Preset(
        id='confluence', name='Confluence', target='source',
        description='Atlassian Confluence — Cloud или on-premise. '
                    'Заполните либо API token (Cloud), либо password (on-prem).',
        fields=[
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
            PresetField('spaces', 'Spaces', required=False,
                        placeholder='DOCS, ENG (через запятую/пробел)',
                        help='Space keys (как в URL Confluence /spaces/DOCS/...).'),
            PresetField('ancestor_ids', 'Включить разделы',
                        kind='chips', required=False,
                        placeholder='ID или URL страниц (любые разделители)',
                        resolver_endpoint='/api/setup/confluence-page-paths',
                        help='Загружаются ТОЛЬКО потомки этих страниц '
                             '(включая их сами). Приоритет над spaces. '
                             'Можно вставлять как ID (1234567), так и URL '
                             '(https://confluence/display/SPACE/Title или '
                             'https://confluence/pages/viewpage.action?pageId=...).'),
            PresetField('skip_ancestor_ids', 'Исключить разделы',
                        kind='chips', required=False, variant='danger',
                        placeholder='ID или URL страниц, которые НЕ загружать',
                        resolver_endpoint='/api/setup/confluence-page-paths',
                        help='Эти страницы и все их потомки пропускаются '
                             '(исключение применяется поверх ancestor_ids/spaces).'),
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
            PresetField('url', 'URL', placeholder='https://jira.your-company.com'),
            PresetField('username', 'Username'),
            PresetField('password', 'Password', kind='password'),
        ],
        build=_build_jira,
    ),
]


# ---------------------------------------------------------------------------
# Embedder presets — выдают snippet для indexing.dense_embedder (replace, не append)
# ---------------------------------------------------------------------------

def _build_embedder_ollama(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'base_url': form.get('base_url') or 'http://host.docker.internal:11434/v1',
        'model': form['model'],
        'api_key': 'ollama',          # Ollama игнорирует, но SDK требует
        'dim': int(form['dim']),
        'max_concurrent': int(form.get('max_concurrent') or 1),
    }
    if form.get('tokenizer'):
        out['tokenizer'] = form['tokenizer']
    return out


def _build_embedder_openai_compatible(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'base_url': form['base_url'],
        'model': form['model'],
        'dim': int(form['dim']),
        'max_concurrent': int(form.get('max_concurrent') or 4),
    }
    if form.get('api_key'):
        out['api_key'] = form['api_key']
    if form.get('tokenizer'):
        out['tokenizer'] = form['tokenizer']
    return out


EMBEDDER_PRESETS: list[Preset] = [
    Preset(
        id='ollama', name='Ollama (локальный)', target='embedder',
        description='Локальный Ollama-сервер. Подходит для qwen3-embedding, '
                    'nomic-embed-text, bge-m3 и др.',
        fields=[
            PresetField('model', 'Модель', placeholder='qwen3-embedding:4b',
                        help='Имя модели как в выводе `ollama list`. '
                             'Стандарт для morag — qwen3-embedding:4b.'),
            PresetField('base_url', 'Base URL',
                        default='http://host.docker.internal:11434/v1', required=False,
                        help='Изнутри docker-compose — host.docker.internal. '
                             'Если консоль локально — http://localhost:11434/v1.'),
            PresetField('dim', 'Размерность вектора (dim)', kind='number',
                        placeholder='2560',
                        help='qwen3-embedding:4b → 2560, nomic-embed-text → 768, '
                             'bge-m3 → 1024.'),
            PresetField('max_concurrent', 'Max concurrent',
                        kind='number', default=1, required=False,
                        help='Ollama сериализует запросы. Обычно 1.'),
            PresetField('tokenizer', 'HuggingFace tokenizer (опционально)',
                        required=False,
                        placeholder='Qwen/Qwen3-Embedding-4B',
                        help='Для точного подсчёта токенов в чанкере. '
                             'Если не задан — TikToken (приближение ±30%).'),
        ],
        build=_build_embedder_ollama,
    ),
    Preset(
        id='openai-compatible', name='OpenAI-compatible', target='embedder',
        description='Любой OpenAI-совместимый endpoint /v1/embeddings: '
                    'OpenAI, Together, vLLM, и др.',
        fields=[
            PresetField('base_url', 'Base URL',
                        placeholder='https://api.openai.com/v1'),
            PresetField('model', 'Модель',
                        placeholder='text-embedding-3-small'),
            PresetField('api_key', 'API key', kind='password',
                        placeholder='sk-...'),
            PresetField('dim', 'Размерность вектора (dim)', kind='number',
                        placeholder='1536',
                        help='text-embedding-3-small → 1536, '
                             'text-embedding-3-large → 3072.'),
            PresetField('max_concurrent', 'Max concurrent',
                        kind='number', default=4, required=False),
            PresetField('tokenizer', 'HuggingFace tokenizer (опционально)',
                        required=False,
                        placeholder='Qwen/Qwen3-Embedding-4B',
                        help='Для точного подсчёта токенов в чанкере. '
                             'Если не задан — TikToken (приближение ±30%).'),
        ],
        build=_build_embedder_openai_compatible,
    ),
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

ALL_PRESETS: list[Preset] = LLM_PRESETS + SOURCE_PRESETS + EMBEDDER_PRESETS


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
                'variant': f.variant,
                'resolver_endpoint': f.resolver_endpoint,
            }
            for f in p.fields
        ],
    }
