"""GET /api/setup/checklist — onboarding-проверка готовности окружения.

Возвращает per-component статус: Qdrant reachability + Ollama installed-models
(сравниваем models из config со списком installed). Цель — на старте дать
юзеру actionable список вида «ollama pull qwen3.5:9b», а не молча падать
посреди индексации.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from morag.config import load_config
from morag.setup_gate import is_setup_complete

logger = logging.getLogger(__name__)
router = APIRouter()

# Patterns в base_url которые мы считаем «это Ollama».
# Не делаем content-sniffing (GET /api/tags на любой URL) — только если URL
# явно похож на Ollama, чтобы не дёргать чужие API без причины.
OLLAMA_HOST_HINTS = ('11434', 'host.docker.internal:11434', 'localhost:11434', '127.0.0.1:11434')

PROBE_TIMEOUT = 5.0  # короткий — checklist должен отдаваться быстро даже при сетевых проблемах


RECOMMENDED_OLLAMA_LLM_MODEL = 'qwen3.5:9b'
RECOMMENDED_OLLAMA_EMBEDDER_MODEL = 'qwen3-embedding:4b'
RECOMMENDED_OLLAMA_EMBEDDER_DIM = 2560
DEFAULT_OLLAMA_BASE_URL = 'http://host.docker.internal:11434/v1'


LOCAL_SOURCE_PATH = '/app/data'   # должно совпадать с presets.LOCAL_SOURCE_PATH


@router.get('/local-source-files')
async def local_source_files() -> dict[str, Any]:
    """Список файлов в /app/data (mount хостовой ./data) — для UI «Проверить»."""
    from pathlib import Path
    root = Path(LOCAL_SOURCE_PATH)
    if not root.exists():
        return {
            'path': LOCAL_SOURCE_PATH, 'exists': False, 'count': 0, 'files': [],
            'error': f'Папка {LOCAL_SOURCE_PATH} не существует. '
                     f'Проверьте volume mount в docker-compose.yml.',
        }
    # Считаем все обычные файлы рекурсивно (без скрытых)
    files = [
        str(p.relative_to(root))
        for p in root.rglob('*')
        if p.is_file() and not any(part.startswith('.') for part in p.parts)
    ]
    return {
        'path': LOCAL_SOURCE_PATH, 'exists': True,
        'count': len(files),
        'files': sorted(files)[:20],   # первые 20 для preview
        'truncated': len(files) > 20,
    }


@router.get('/quickstart-status')
async def quickstart_status() -> dict[str, Any]:
    """Готовность к одно-кликовой установке рекомендуемого Ollama-стека.

    Проверяет: Ollama доступна, обе рекомендуемые модели installed.
    Возвращает status каждой модели + can_apply (когда всё ОК).
    """
    api_root = _ollama_api_root(DEFAULT_OLLAMA_BASE_URL)
    installed: set[str] = set()
    ollama_error: str | None = None
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            r = await client.get(f'{api_root}/api/tags')
            r.raise_for_status()
            installed = {m['name'] for m in r.json().get('models', [])}
    except Exception as e:
        ollama_error = f'{type(e).__name__}: {e}'

    models = [
        {
            'role': 'llm',
            'model': RECOMMENDED_OLLAMA_LLM_MODEL,
            'installed': RECOMMENDED_OLLAMA_LLM_MODEL in installed,
            'pull_cmd': f'ollama pull {RECOMMENDED_OLLAMA_LLM_MODEL}',
        },
        {
            'role': 'embedder',
            'model': RECOMMENDED_OLLAMA_EMBEDDER_MODEL,
            'installed': RECOMMENDED_OLLAMA_EMBEDDER_MODEL in installed,
            'pull_cmd': f'ollama pull {RECOMMENDED_OLLAMA_EMBEDDER_MODEL}',
        },
    ]
    can_apply = ollama_error is None and all(m['installed'] for m in models)
    return {
        'ollama_available': ollama_error is None,
        'ollama_error': ollama_error,
        'ollama_base_url': DEFAULT_OLLAMA_BASE_URL,
        'models': models,
        'can_apply': can_apply,
    }


@router.post('/quickstart-apply')
async def quickstart_apply(request: Request) -> dict[str, Any]:
    """Одним кликом добавить рекомендуемый Ollama-стек: LLM + embedder + roles."""
    from fastapi import HTTPException
    from pydantic import ValidationError
    from services.console.config_io import patch_local, read_layered, read_local, validate_merged

    cfg_path = request.app.state.config_path
    current_local = read_local(cfg_path)
    merged_view = read_layered(cfg_path)

    # 1. LLM main (multimodal qwen3.5:9b)
    llm_item = {
        'name': 'main',
        'base_url': DEFAULT_OLLAMA_BASE_URL,
        'model': RECOMMENDED_OLLAMA_LLM_MODEL,
        'api_key': 'ollama',
        'capabilities': ['text', 'vision'],
        'enable_thinking': False,
        'context_window': 32768,
        'max_concurrent': 1,
    }
    llms = list(merged_view.get('llms') or [])
    llms = [item for item in llms if item.get('name') != 'main'] + [llm_item]

    # 2. Embedder (Ollama qwen3-embedding:4b).
    # tokenizer = HF-id для точного подсчёта токенов в чанкере. Если не задать,
    # cli/main.py fallback'ом возьмёт model 'qwen3-embedding:4b' — это НЕ HF id
    # (двоеточие, Ollama-нотация), AutoTokenizer.from_pretrained упадёт.
    embedder_item = {
        'base_url': DEFAULT_OLLAMA_BASE_URL,
        'model': RECOMMENDED_OLLAMA_EMBEDDER_MODEL,
        'tokenizer': 'Qwen/Qwen3-Embedding-4B',
        'api_key': 'ollama',
        'dim': RECOMMENDED_OLLAMA_EMBEDDER_DIM,
        'max_concurrent': 1,
    }

    # 3. Roles: llm=main, vision=main
    indexing = dict(current_local.get('indexing') or {})
    indexing['llm'] = 'main'
    indexing['vision'] = 'main'
    indexing['dense_embedder'] = embedder_item

    candidate_local = {**current_local, 'llms': llms, 'indexing': indexing}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    patch_local(cfg_path, {'llms': llms, 'indexing': indexing})
    return {'ok': True, 'llm': 'main', 'embedder': RECOMMENDED_OLLAMA_EMBEDDER_MODEL}


@router.get('/ollama-models')
async def ollama_models(base_url: str) -> dict[str, Any]:
    """Список installed-моделей с Ollama-сервера. Для UI-dropdown в пресетах.

    `base_url` — то же значение, которое юзер вводит в форме пресета (с `/v1` или без).
    Делаем `GET {host}/api/tags` (нативный Ollama API, не OpenAI-compat).
    Возвращает {ok, models: [...], error?}.
    """
    api_root = _ollama_api_root(base_url)
    if api_root is None:
        return {'ok': False, 'models': [], 'error': 'URL не похож на Ollama-сервер'}

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            r = await client.get(f'{api_root}/api/tags')
            r.raise_for_status()
            data = r.json()
            models = sorted(m['name'] for m in data.get('models', []))
            return {'ok': True, 'models': models, 'host': api_root}
    except Exception as e:
        return {'ok': False, 'models': [], 'error': f'{type(e).__name__}: {e}'}


@router.get('/gate')
async def setup_gate(request: Request) -> dict[str, Any]:
    """Setup-gate: можно ли запускать индексацию.

    Реализован прямо в console (не дёргаем indexer) — gate нужен в UI ДО
    того как юзер кликнул Start, и не зависит от indexer-availability.
    """
    cfg_path = request.app.state.config_path
    ok, blockers = is_setup_complete(cfg_path)
    return {'ok': ok, 'blockers': blockers}


@router.get('/checklist')
async def checklist(request: Request) -> dict[str, Any]:
    """Проверка окружения: Qdrant + Ollama-models из текущего конфига."""
    cfg_path = request.app.state.config_path
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        return {
            'config_ok': False,
            'config_error': str(e),
            'qdrant': None,
            'ollama': [],
        }

    # Собираем (base_url, model, role) для проверки на Ollama.
    expected = _collect_expected_models(cfg)

    # Группируем по base_url — один запрос /api/tags на хост, не на каждую модель.
    by_host = _group_by_ollama_host(expected)
    installed_by_host = await _query_ollama_installed(by_host.keys())

    ollama_status = _build_ollama_status(by_host, installed_by_host)

    qdrant_status = await _check_qdrant(cfg.qdrant)

    return {
        'config_ok': True,
        'qdrant': qdrant_status,
        'ollama': ollama_status,
    }


def _collect_expected_models(cfg) -> list[dict[str, str]]:
    """Все (base_url, model, role) которые нужны индексеру по текущему конфигу."""
    out = []
    for llm in cfg.llms:
        out.append({'base_url': llm.base_url, 'model': llm.model, 'role': f'llm/{llm.name}'})
    if cfg.indexing and cfg.indexing.dense_embedder and cfg.indexing.dense_embedder.base_url:
        emb = cfg.indexing.dense_embedder
        out.append({'base_url': emb.base_url, 'model': emb.model, 'role': 'dense_embedder'})
    return out


def _group_by_ollama_host(expected: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """{ollama_api_root: [items, ...]} только для тех URL что похожи на Ollama."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in expected:
        api_root = _ollama_api_root(item['base_url'])
        if api_root is None:
            continue
        grouped.setdefault(api_root, []).append(item)
    return grouped


def _ollama_api_root(base_url: str) -> str | None:
    """Превратить OpenAI-compat base_url в корень для Ollama-нативного API.

    Ollama раздаёт OpenAI-compat на `/v1`, нативный API — на корне (`/api/tags`).
    Возвращает None если URL не похож на Ollama.
    """
    if not any(hint in base_url for hint in OLLAMA_HOST_HINTS):
        return None
    parsed = urlparse(base_url)
    return f'{parsed.scheme}://{parsed.netloc}'


async def _query_ollama_installed(hosts) -> dict[str, set[str] | None]:
    """{host: {model_name, ...} | None}. None — host недоступен."""
    result: dict[str, set[str] | None] = {}
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
        for host in hosts:
            try:
                r = await client.get(f'{host}/api/tags')
                r.raise_for_status()
                data = r.json()
                # Ollama отдаёт {"models": [{"name": "qwen3.5:9b", ...}, ...]}
                result[host] = {m['name'] for m in data.get('models', [])}
            except Exception as e:
                logger.info('Ollama %s unreachable: %s', host, e)
                result[host] = None
    return result


def _build_ollama_status(
    by_host: dict[str, list[dict[str, str]]],
    installed_by_host: dict[str, set[str] | None],
) -> list[dict[str, Any]]:
    """Развернуть в плоский список per-model статус."""
    out = []
    for host, items in by_host.items():
        installed = installed_by_host.get(host)
        host_reachable = installed is not None
        for item in items:
            model = item['model']
            out.append({
                'host': host,
                'model': model,
                'role': item['role'],
                'host_reachable': host_reachable,
                'installed': bool(installed and model in installed) if host_reachable else None,
                'pull_cmd': f'ollama pull {model}',
            })
    return out


# ---------------------------------------------------------------------------
# POST /api/setup/confluence-page-paths
# Резолв названий + breadcrumb-путей для page IDs из chip-инпутов wizard'а.
# ---------------------------------------------------------------------------

class ConfluencePagePathsRequest(BaseModel):
    """Поля совпадают с form'ой Confluence-пресета.

    Если api_token и password оба пустые, но source_name задан — backend
    возьмёт секрет из текущего config (case Edit, где UI не отдаёт секрет).
    """
    url: str
    username: str
    api_token: str | None = None
    password: str | None = None
    source_name: str | None = None
    ids: list[str]


@router.post('/confluence-page-paths')
async def confluence_page_paths(
    req: ConfluencePagePathsRequest, request: Request,
) -> dict[str, dict[str, Any]]:
    """Резолв названий + breadcrumb-путей по списку Confluence page IDs.

    Возвращает {id: {title, path, url, error?}}. Один HTTP-вызов на ID
    (atlassian-python-api не имеет batch-API для ancestors). Параллелит
    через asyncio.gather с ограничением concurrency.
    """
    cfg_path = request.app.state.config_path

    # Items могут быть numeric IDs ИЛИ URLs (Confluence display/spaces).
    # Дедупликация по input-ключу (то что прислал юзер).
    clean_inputs: list[str] = []
    seen: set[str] = set()
    for raw in req.ids:
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        clean_inputs.append(s)

    if not clean_inputs:
        return {}

    # atlassian-python-api — sync. Через run_in_executor чтобы не блочить event loop.
    cf = _build_cf_client(
        req.url, req.username, req.api_token, req.password, req.source_name, cfg_path,
    )

    sem = asyncio.Semaphore(8)

    async def fetch_one(input_str: str) -> tuple[str, dict[str, Any]]:
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                page = await loop.run_in_executor(None, lambda: _resolve_page(cf, input_str))
                return input_str, _page_info(req.url, page, input_str)
            except Exception as e:
                logger.info('Failed to resolve confluence input %r: %s', input_str, e)
                return input_str, {
                    'id': input_str if input_str.isdigit() else '',
                    'title': input_str,
                    'path': '',
                    'error': f'{type(e).__name__}: {e}',
                }

    pairs = await asyncio.gather(*(fetch_one(inp) for inp in clean_inputs))
    return dict(pairs)


def _resolve_page(cf, input_str: str) -> dict:
    """Универсальный резолв: numeric ID, URL с pageId=, /display/SPACE/Title,
    /spaces/SPACE/pages/N/Title. Возвращает page-dict с ancestors+space.
    """
    from urllib.parse import parse_qs, unquote, urlparse

    if input_str.isdigit():
        return cf.get_page_by_id(input_str, expand='ancestors,space')

    u = urlparse(input_str)
    if not u.scheme:
        # Не URL и не числа — попытаться как title без space неоднозначно. Падаем.
        raise ValueError(f'Не URL и не numeric ID: {input_str!r}')

    # 1. Query string ?pageId=...
    qs = parse_qs(u.query)
    if 'pageId' in qs and qs['pageId'][0].isdigit():
        return cf.get_page_by_id(qs['pageId'][0], expand='ancestors,space')

    path_parts = [p for p in u.path.split('/') if p]

    # 2. /display/{space}/{title} (on-prem стиль)
    if len(path_parts) >= 3 and path_parts[0] == 'display':
        space = path_parts[1]
        title = unquote(path_parts[2].replace('+', ' '))
        return cf.get_page_by_title(space, title, expand='ancestors,space')

    # 3. /spaces/{space}/pages/{id}/{title-slug} (Cloud) или /wiki/spaces/...
    if 'pages' in path_parts:
        idx = path_parts.index('pages')
        if idx + 1 < len(path_parts) and path_parts[idx + 1].isdigit():
            return cf.get_page_by_id(path_parts[idx + 1], expand='ancestors,space')

    # 4. /wiki/display/{space}/{title} (на всякий)
    if len(path_parts) >= 4 and path_parts[0] == 'wiki' and path_parts[1] == 'display':
        space = path_parts[2]
        title = unquote(path_parts[3].replace('+', ' '))
        return cf.get_page_by_title(space, title, expand='ancestors,space')

    raise ValueError(f'Не удалось распарсить URL: {input_str!r}')


def _confluence_page_url(base_url: str, page: dict) -> str:
    """Собрать прямой URL страницы. atlassian-python-api отдаёт _links.webui — относительный."""
    rel = (page.get('_links') or {}).get('webui') or ''
    if not rel:
        # fallback: viewpage.action
        return f"{base_url.rstrip('/')}/pages/viewpage.action?pageId={page.get('id')}"
    if rel.startswith('http'):
        return rel
    return f"{base_url.rstrip('/')}{rel}"


def _resolve_confluence_secret(
    cfg_path, source_name: str | None, api_token: str | None, password: str | None,
) -> str:
    """Секрет из формы; если пусто (case Edit) — fall back на config по source_name."""
    secret = (api_token or '').strip() or (password or '').strip()
    if not secret and source_name:
        from services.console.config_io import read_layered
        merged = read_layered(cfg_path)
        existing = next(
            (s for s in (merged.get('sources') or [])
             if s.get('kind') == 'confluence' and s.get('name') == source_name),
            None,
        )
        if existing:
            secret = existing.get('api_token') or existing.get('password') or ''
    return secret


def _build_cf_client(
    url: str, username: str, api_token: str | None, password: str | None,
    source_name: str | None, cfg_path,
):
    """Сконструировать atlassian.Confluence клиент с резолвом секрета. 400 если нет."""
    secret = _resolve_confluence_secret(cfg_path, source_name, api_token, password)
    if not secret:
        raise HTTPException(
            status_code=400,
            detail='Заполните api_token (Cloud) или password (on-prem).',
        )
    from atlassian import Confluence
    is_cloud = bool(api_token and not password)
    return Confluence(
        url=url.rstrip('/'), username=username, password=secret,
        cloud=is_cloud, timeout=10,
    )


def _page_info(base_url: str, page: dict, fallback_title: str = '') -> dict[str, Any]:
    """Унифицированный dict страницы: id, title, path (breadcrumb), url, space (key)."""
    title = page.get('title') or fallback_title
    ancestors = page.get('ancestors') or []
    space_obj = page.get('space') or {}
    space_name = space_obj.get('name') or space_obj.get('key')
    parts: list[str] = []
    if space_name:
        parts.append(space_name)
    parts.extend(a.get('title') or a.get('id') for a in ancestors)
    parts.append(title)
    return {
        'id': str(page.get('id') or ''),
        'title': title,
        'path': ' / '.join(p for p in parts if p),
        'url': _confluence_page_url(base_url, page),
        'space': space_obj.get('key'),
    }


class ConfluenceSearchTitlesRequest(BaseModel):
    """Поиск страниц по префиксу названия в «знакомых» пространствах (live CQL).

    Auth-поля как у /confluence-page-paths (секрет — из формы или config по source_name).
    """
    url: str
    username: str
    api_token: str | None = None
    password: str | None = None
    source_name: str | None = None
    spaces: list[str] = []
    query: str
    limit: int = 10


# CQL-safe space key (стандартные ключи + персональные ~user, точки/дефисы).
_SPACE_KEY_RE = re.compile(r'^[A-Za-z0-9_~.\-]+$')


@router.post('/confluence-search-titles')
async def confluence_search_titles(
    req: ConfluenceSearchTitlesRequest, request: Request,
) -> dict[str, Any]:
    """Live-поиск страниц Confluence по началу названия внутри заданных spaces.

    Без spaces или пустого query → {results: []} (фронт показывает подсказку
    «введите ссылку/ID, чтобы открыть пространство»). Поиск только по «знакомым»
    пространствам — guardrail против перебора всего инстанса.
    """
    cfg_path = request.app.state.config_path
    keys = [k.strip() for k in (req.spaces or []) if k and _SPACE_KEY_RE.match(k.strip())]
    query = (req.query or '').strip()
    if not keys or not query:
        return {'results': []}

    cf = _build_cf_client(
        req.url, req.username, req.api_token, req.password, req.source_name, cfg_path,
    )

    space_list = ','.join(f'"{k}"' for k in keys)
    q = query.replace('\\', '\\\\').replace('"', '\\"')
    cql = f'type = page AND space in ({space_list}) AND title ~ "{q}*"'
    limit = max(1, min(req.limit or 10, 25))

    def _run() -> list[dict[str, Any]]:
        resp = cf.get('rest/api/search', params={
            'cql': cql,
            'limit': limit,
            'expand': 'content.space,content.ancestors',
        })
        out: list[dict[str, Any]] = []
        for it in (resp or {}).get('results', []):
            page = it.get('content') or {}
            # _links.webui приходит на уровне результата, не content — пробрасываем.
            if not page.get('_links') and it.get('_links'):
                page = {**page, '_links': it['_links']}
            if not page.get('id'):
                continue
            out.append(_page_info(req.url, page))
        return out

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.info('Confluence title search failed (cql=%r): %s', cql, e)
        raise HTTPException(status_code=400, detail=f'{type(e).__name__}: {e}') from e
    return {'results': results}


async def _check_qdrant(qdrant_cfg) -> dict[str, Any]:
    """Ping Qdrant через get_collections — он лёгкий и не требует прав."""
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port, timeout=PROBE_TIMEOUT)
    try:
        cols = await client.get_collections()
        return {
            'reachable': True,
            'collections': [c.name for c in cols.collections],
            'host': qdrant_cfg.host,
            'port': qdrant_cfg.port,
        }
    except Exception as e:
        return {
            'reachable': False,
            'error': f'{type(e).__name__}: {e}',
            'host': qdrant_cfg.host,
            'port': qdrant_cfg.port,
        }
    finally:
        await client.close()
