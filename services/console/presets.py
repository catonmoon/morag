"""Опинионированные пресеты провайдеров для wizard'а.

Цель — сократить onboarding до 2-3 полей вместо целого config.example.yml.
Каждый пресет описывает: target (куда подкладывать в YAML), какие поля
спросить у юзера, и как собрать итоговый overlay-snippet.

Не покрывает экзотику — только массовые сценарии. Для нестандартных кейсов
есть пресет 'custom', где все поля редактируются вручную.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


PresetTarget = Literal['llm', 'llm_vision', 'dense_embedder', 'sparse_embedder']


@dataclass(frozen=True)
class PresetField:
    """Описание одного поля формы пресета."""
    name: str                            # ключ в form_data
    label: str                           # подпись в UI
    kind: Literal['text', 'password', 'number'] = 'text'
    required: bool = True
    default: str | int | None = None
    placeholder: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class Preset:
    """Шаблон для конкретного провайдера + типа вектора/LLM."""
    id: str
    name: str
    target: PresetTarget
    fields: list[PresetField]
    build: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ''


# ---------------------------------------------------------------------------
# LLM presets
# ---------------------------------------------------------------------------

def _build_grok(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'base_url': 'https://api.x.ai/v1',
        'model': form.get('model', 'grok-4-1-fast-non-reasoning'),
        'api_key': form['api_key'],
        'context_window': 256000,
        'max_concurrent': 8,
    }


def _build_openrouter(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'base_url': 'https://openrouter.ai/api/v1',
        'model': form['model'],
        'api_key': form['api_key'],
        'context_window': int(form.get('context_window', 32768)),
        'max_concurrent': int(form.get('max_concurrent', 4)),
    }


def _build_ollama_llm(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'base_url': form.get('base_url', 'http://host.docker.internal:11434/v1'),
        'model': form['model'],
        'api_key': 'ollama',  # OpenAI-compat требует непустое значение, Ollama его игнорирует
        'context_window': int(form.get('context_window', 32768)),
        'max_concurrent': int(form.get('max_concurrent', 1)),
        'enable_thinking': False,
    }


def _build_vllm_custom_llm(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'base_url': form['base_url'],
        'model': form['model'],
        'api_key': form['api_key'],
        'context_window': int(form.get('context_window', 32768)),
    }
    if form.get('max_concurrent'):
        out['max_concurrent'] = int(form['max_concurrent'])
    return out


LLM_PRESETS: list[Preset] = [
    Preset(
        id='grok',
        name='Grok (xAI)',
        target='llm',
        description='xAI Grok через api.x.ai. Большой контекст, дёшево, быстро.',
        fields=[
            PresetField('model', 'Модель', default='grok-4-1-fast-non-reasoning',
                        help='grok-4-1-fast-non-reasoning — рекомендуется. Reasoning-варианты — медленнее и дороже.'),
            PresetField('api_key', 'API key', kind='password',
                        placeholder='xai-...'),
        ],
        build=_build_grok,
    ),
    Preset(
        id='openrouter',
        name='OpenRouter',
        target='llm',
        description='Единая точка входа в десятки моделей. Удобно для экспериментов.',
        fields=[
            PresetField('model', 'Модель', placeholder='anthropic/claude-haiku-4.5'),
            PresetField('api_key', 'API key', kind='password',
                        placeholder='sk-or-...'),
            PresetField('context_window', 'Context window (токены)', kind='number',
                        default=200000, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        default=4, required=False),
        ],
        build=_build_openrouter,
    ),
    Preset(
        id='ollama',
        name='Ollama (локальный)',
        target='llm',
        description='Локальный Ollama-сервер. Подходит для приватных деплоев без внешних API.',
        fields=[
            PresetField('model', 'Модель', placeholder='qwen3:4b',
                        help='Имя модели как в `ollama list`. Например qwen3:4b, llama3.1:8b, mistral.'),
            PresetField('base_url', 'Base URL',
                        default='http://host.docker.internal:11434/v1', required=False,
                        help='host.docker.internal — для доступа из контейнера к Ollama на хост-машине.'),
            PresetField('context_window', 'Context window (токены)', kind='number',
                        default=32768, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        default=1, required=False,
                        help='Ollama обычно сериализует запросы — ставь 1, если не настраивал параллелизм.'),
        ],
        build=_build_ollama_llm,
    ),
    Preset(
        id='custom',
        name='Custom (vLLM / OpenAI-compat)',
        target='llm',
        description='Любой OpenAI-совместимый endpoint. Все параметры — вручную.',
        fields=[
            PresetField('base_url', 'Base URL', placeholder='https://...'),
            PresetField('model', 'Модель'),
            PresetField('api_key', 'API key', kind='password'),
            PresetField('context_window', 'Context window (токены)', kind='number',
                        default=32768, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        required=False),
        ],
        build=_build_vllm_custom_llm,
    ),
]


# ---------------------------------------------------------------------------
# Dense embedder presets
# ---------------------------------------------------------------------------

# Стандартный Qwen3-Embedding instruct-шаблон для query-side.
# document-side префикс не нужен — Qwen3 без него работает корректно.
QWEN3_QUERY_TEMPLATE = (
    'Instruct: Given a web search query, retrieve relevant passages that answer the query\n'
    'Query: {text}'
)


def _build_ollama_embedder(form: dict[str, Any]) -> dict[str, Any]:
    return {
        'model': form.get('model', 'qwen3-embedding:4b'),
        'base_url': form.get('base_url', 'http://host.docker.internal:11434/v1'),
        'dim': int(form.get('dim', 2560)),
        'tokenizer': 'tiktoken',
        'document_template': '{text}',
        'query_template': QWEN3_QUERY_TEMPLATE,
        'max_concurrent': int(form.get('max_concurrent', 4)),
    }


def _build_custom_embedder(form: dict[str, Any]) -> dict[str, Any]:
    out = {
        'base_url': form['base_url'],
        'model': form['model'],
        'dim': int(form['dim']),
        'tokenizer': form.get('tokenizer') or 'tiktoken',
    }
    if form.get('api_key'):
        # SDK требует key (даже фиктивный для self-hosted) — но конфиг не имеет поля api_key
        # для embedder; шаблоны на это не влияют. Игнорируем здесь.
        pass
    if form.get('document_template'):
        out['document_template'] = form['document_template']
    if form.get('query_template'):
        out['query_template'] = form['query_template']
    return out


DENSE_EMBEDDER_PRESETS: list[Preset] = [
    Preset(
        id='ollama-qwen3',
        name='Ollama: Qwen3-Embedding-4B',
        target='dense_embedder',
        description='Рекомендуемая локальная конфигурация. dim=2560, контекст 32K.',
        fields=[
            PresetField('model', 'Модель', default='qwen3-embedding:4b', required=False),
            PresetField('base_url', 'Base URL',
                        default='http://host.docker.internal:11434/v1', required=False),
            PresetField('dim', 'Dim', kind='number', default=2560, required=False),
            PresetField('max_concurrent', 'Max concurrent', kind='number',
                        default=4, required=False),
        ],
        build=_build_ollama_embedder,
    ),
    Preset(
        id='custom',
        name='Custom (OpenAI-compat)',
        target='dense_embedder',
        description='Любой OpenAI-совместимый embeddings endpoint.',
        fields=[
            PresetField('base_url', 'Base URL', placeholder='https://...'),
            PresetField('model', 'Модель'),
            PresetField('dim', 'Dim', kind='number',
                        help='Размерность вектора. Должна совпадать с моделью; меняется только пересозданием коллекции.'),
            PresetField('tokenizer', 'Tokenizer', default='tiktoken', required=False,
                        help='HF-имя или "tiktoken" как fallback.'),
        ],
        build=_build_custom_embedder,
    ),
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

ALL_PRESETS: list[Preset] = LLM_PRESETS + DENSE_EMBEDDER_PRESETS


def find_preset(target: PresetTarget, preset_id: str) -> Preset:
    """Найти пресет по target + id. Кидает KeyError если нет."""
    for p in ALL_PRESETS:
        if p.target == target and p.id == preset_id:
            return p
    raise KeyError(f'No preset found: target={target}, id={preset_id}')


def apply_preset(target: PresetTarget, preset_id: str, form: dict[str, Any]) -> dict[str, Any]:
    """Собрать overlay-snippet от заполненной формы.

    Возвращает структуру вида {'llm': {...}} или {'indexing': {'dense_embedder': {...}}},
    готовую к merge в config.local.yml через config_io.patch_local.
    """
    preset = find_preset(target, preset_id)
    snippet = preset.build(form)

    if target == 'llm':
        return {'llm': snippet}
    if target == 'llm_vision':
        return {'llm_vision': snippet}
    if target == 'dense_embedder':
        return {'indexing': {'dense_embedder': snippet}}
    if target == 'sparse_embedder':
        return {'indexing': {'sparse_embedder': snippet}}
    raise ValueError(f'Unknown target: {target}')


def serialize_preset(p: Preset) -> dict[str, Any]:
    """JSON-friendly представление для GET /api/presets."""
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
