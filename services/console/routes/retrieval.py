"""Retrieval-конфиг (секция config.local.yml.retrieval).

Источник истины для pipeline (агентский RAG). OWUI Valves остаются как
override-механизм, но настройки по умолчанию приходят отсюда.

Pipeline подхватывает retrieval-правки hot-reload'ом: следит за mtime
config.yml + config.local.yml и пересобирает настройки/клиенты/тулы без рестарта
(см. morag_pipeline._maybe_reload / CLAUDE.md). Рестарт контейнера не нужен.
"""
from __future__ import annotations

from typing import Any

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from morag.config import (
    DEFAULT_ADMIN_INSTRUCTIONS,
    DEFAULT_ANSWER_STYLE,
    DEFAULT_CORPUS_DESCRIPTION,
    RetrievalConfig,
)
from morag.retrieval.prompt import build_prompt_sections
from morag.retrieval.tools import REGISTRY
from services.console.config_io import read_layered, read_local, validate_merged, write_local

router = APIRouter()


@router.get('/doc-lookup')
async def doc_lookup(id: str, request: Request) -> dict[str, Any]:
    """Достать title + url документа из Qdrant docs collection.

    Принимает на вход разные форматы:
      - Полный doc_id вида `confluence:virgo:1068775100` — резолв через uuid5
        (быстро, один запрос).
      - URL страницы (например, `https://virgo.redelephant.ru/pages/viewpage.action?pageId=...`)
        или просто numeric pageId — поиск по payload.url или по hash-суффиксу
        doc_id (медленнее, scroll + filter).

    Возвращает `{ok, id, title, url}` — поле `id` это resolved полный doc_id
    (Console пишет его в config). `{ok: false, error}` если не найден.
    """
    cfg_path = request.app.state.config_path
    merged = read_layered(cfg_path)
    qdrant_cfg = (merged.get('qdrant') or {})
    host = qdrant_cfg.get('host', 'qdrant')
    port = qdrant_cfg.get('port', 6333)
    docs_collection = qdrant_cfg.get('collection_docs', 'docs')

    import httpx
    import re as _re
    base = f'http://{host}:{port}/collections/{docs_collection}'
    inp = (id or '').strip()
    if not inp:
        return {'ok': False, 'id': id, 'error': 'пустой запрос'}

    async with httpx.AsyncClient(timeout=10) as c:
        # 1) Точный doc_id — пробуем uuid5 lookup напрямую (samый быстрый путь).
        if ':' in inp:
            point_id = _doc_uuid(inp)
            r = await c.get(f'{base}/points/{point_id}',
                            params={'with_payload': 'true', 'with_vector': 'false'})
            if r.status_code == 200:
                return _format_doc_lookup_response(inp, r.json())

        # 2) URL → ищем по payload.url (exact match).
        if inp.startswith(('http://', 'https://')):
            r = await c.post(f'{base}/points/scroll', json={
                'limit': 1, 'with_payload': True, 'with_vector': False,
                'filter': {'must': [{'key': 'url', 'match': {'value': inp}}]},
            })
            r.raise_for_status()
            points = (r.json().get('result') or {}).get('points') or []
            if points:
                payload = points[0].get('payload') or {}
                return _format_doc_lookup_response(payload.get('id') or inp, {'result': {'payload': payload}})

        # 3) URL с pageId или просто numeric — фильтр по суффиксу `:<pageId>` в id.
        page_id_match = _re.search(r'(?:pageId=|^)(\d+)$', inp)
        if page_id_match:
            page_id = page_id_match.group(1)
            r = await c.post(f'{base}/points/scroll', json={
                'limit': 5, 'with_payload': True, 'with_vector': False,
            })
            r.raise_for_status()
            # Нужен полноценный фильтр по суффиксу — Qdrant не умеет regex, поэтому
            # фильтруем приближённо через text match по `:<pageId>` и client-side
            # уточняем endswith.
            r2 = await c.post(f'{base}/points/scroll', json={
                'limit': 50, 'with_payload': True, 'with_vector': False,
                'filter': {'must': [{'key': 'id', 'match': {'text': page_id}}]},
            })
            if r2.status_code == 200:
                for p in (r2.json().get('result') or {}).get('points') or []:
                    pl = p.get('payload') or {}
                    if (pl.get('id') or '').endswith(f':{page_id}'):
                        return _format_doc_lookup_response(pl['id'], {'result': {'payload': pl}})

    return {'ok': False, 'id': inp, 'error': 'не найден в базе'}


def _doc_uuid(doc_id: str) -> str:
    """Qdrant point id = uuid5(_DOC_NAMESPACE, doc_id). NS из
    morag.storage.repository._DOC_NAMESPACE."""
    import uuid
    _DOC_NS = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
    return str(uuid.uuid5(_DOC_NS, doc_id))


def _doc_path_str(payload: dict) -> str:
    """payload.path — list[str] (обычно один элемент-breadcrumb). → строка."""
    path_val = payload.get('path')
    if isinstance(path_val, list):
        return path_val[0] if path_val else ''
    return path_val or ''


def _format_doc_lookup_response(doc_id: str, qdrant_response: dict) -> dict[str, Any]:
    payload = (qdrant_response.get('result') or {}).get('payload') or {}
    path = _doc_path_str(payload)
    title = payload.get('title') or path or doc_id
    return {
        'ok': True,
        'id': payload.get('id') or doc_id,
        'title': title,
        'url': payload.get('url') or '',
        'path': path,
        # Бейдж glossary-чипа: spaceKey в Qdrant-payload нет — показываем source_name.
        'source_name': payload.get('source_name') or '',
    }


@router.get('/doc-search')
async def doc_search(q: str, request: Request, limit: int = 10) -> dict[str, Any]:
    """Поиск проиндексированных документов по части названия (для автокомплита глоссария).

    Матчим title подстрокой (case-insensitive) на стороне Python — full-text индекс
    на title не гарантирован. Док в коллекции на порядки меньше чанков; скроллим до
    SCAN_CAP. При росте корпуса оптимизировать через payload full-text index + MatchText.
    Глоссарий ссылается только на уже проиндексированное → ищем по Qdrant docs, а не
    по живому Confluence (в отличие от поиска источников). known-spaces тут не нужны.
    """
    query = (q or '').strip().lower()
    if len(query) < 2:
        return {'results': []}

    cfg_path = request.app.state.config_path
    merged = read_layered(cfg_path)
    qdrant_cfg = (merged.get('qdrant') or {})
    host = qdrant_cfg.get('host', 'qdrant')
    port = qdrant_cfg.get('port', 6333)
    docs_collection = qdrant_cfg.get('collection_docs', 'docs')
    limit = max(1, min(limit or 10, 25))

    import httpx
    base = f'http://{host}:{port}/collections/{docs_collection}'
    results: list[dict[str, Any]] = []
    offset: Any = None
    scanned = 0
    scan_cap = 5000

    async with httpx.AsyncClient(timeout=10) as c:
        while scanned < scan_cap and len(results) < limit:
            body: dict[str, Any] = {
                'limit': 256,
                'with_vector': False,
                'with_payload': {'include': ['id', 'title', 'path', 'url', 'source_name']},
            }
            if offset is not None:
                body['offset'] = offset
            r = await c.post(f'{base}/points/scroll', json=body)
            if r.status_code != 200:
                break
            result = r.json().get('result') or {}
            points = result.get('points') or []
            for p in points:
                pl = p.get('payload') or {}
                title = pl.get('title') or ''
                if query in title.lower():
                    results.append({
                        'id': pl.get('id') or '',
                        'title': title,
                        'path': _doc_path_str(pl),
                        'url': pl.get('url') or '',
                        'badge': pl.get('source_name') or '',
                    })
                    if len(results) >= limit:
                        break
            scanned += len(points)
            offset = result.get('next_page_offset')
            if offset is None:
                break

    return {'results': results}


# ---------------------------------------------------------------------------
# Request schema (повторяет RetrievalConfig, без обязательности llm-полей —
# UI может присылать частичные правки + валидация дальше через validate_merged)
# ---------------------------------------------------------------------------

class RoleIn(BaseModel):
    llm: str = Field(min_length=1)
    enable_thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class FindSectionIn(BaseModel):
    doc_pool: int | None = None
    descent_threshold: float | None = None
    top_docs: int | None = None


class SearchIn(BaseModel):
    limit: int | None = None
    unique_docs_cap: int | None = None
    sections_limit: int | None = None
    max_iterations: int | None = None
    answer_max_tokens: int | None = None
    find_section: FindSectionIn | None = None


class FeaturesIn(BaseModel):
    enable_diversity_nudge: bool | None = None


class PromptsIn(BaseModel):
    admin_instructions: str = ''
    corpus_description: str = ''
    answer_style: str = ''
    completeness_check: bool = True


class GlossaryIn(BaseModel):
    enabled: bool = False
    doc_id: str = ''                # back-compat (single)
    doc_ids: list[str] = Field(default_factory=list)
    description: str = ''


class RetrievalIn(BaseModel):
    agent: RoleIn | None = None
    reranker: RoleIn | None = None
    search: SearchIn = SearchIn()
    features: FeaturesIn = FeaturesIn()
    prompts: PromptsIn = PromptsIn()
    # Единый список агентских тулов ({type, name, enabled, description, ...}).
    # Схему записей валидирует validate_merged → RetrievalToolConfig union.
    # Legacy-блок glossary оставлен для старых клиентов; при непустом tools
    # glossary в payload игнорируется (см. put_retrieval).
    tools: list[dict[str, Any]] = Field(default_factory=list)
    glossary: GlossaryIn = GlossaryIn()
    http_timeout: int | None = None


class ToolPreviewIn(BaseModel):
    """Конфиг ОДНОГО инстанса тула для расчёта «как видит агент»."""
    type: str
    name: str = ''
    description: str = ''
    required: bool | None = None
    trigger: str = ''
    fields: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post('/tool-preview')
async def tool_preview(req: ToolPreviewIn) -> dict[str, Any]:
    """Эффективные тексты тула из реестра типов (morag.retrieval.tools):
    `description` (в schema, видит агент) и `prompt` (секция system prompt).
    Без записи — для предпросмотра/редактирования в Console UI. Шаблоны живут
    в Python, поэтому считаем здесь, а не дублируем в JS."""
    spec = REGISTRY.get(req.type)
    if spec is None:
        return {'ok': False, 'error': f'неизвестный тип тула: {req.type}'}
    cfg = req.model_dump()
    try:
        # default_description — то же, но с пустым description: дефолт типа,
        # которым UI предзаполняет поле core-тула (и сравнивает на save).
        return {
            'ok': True,
            'core': spec.core,
            'default_name': spec.default_name,
            'description': spec.describe(cfg),
            'default_description': spec.describe({**cfg, 'description': ''}),
            'prompt': spec.prompt_section(cfg).strip(),
        }
    except Exception as e:  # noqa: BLE001 — отдаём ошибку в UI, не 500
        return {'ok': False, 'error': str(e)}


class PromptPreviewIn(BaseModel):
    """Редактируемые поля промпта (текущее состояние редактора) — для живого превью."""
    corpus_description: str = ''
    answer_style: str = ''
    completeness_check: bool = True
    admin_instructions: str = ''
    require_find_section: bool | None = None  # None = взять из сохранённого конфига


@router.post('/prompt-preview')
async def prompt_preview(req: PromptPreviewIn, request: Request) -> dict[str, Any]:
    """Структура собранного системного промпта (секции + подсветка настраиваемых
    частей) — ровно то, что pipeline шлёт агенту (morag.retrieval.prompt — единый
    источник). Редактируемые тексты берём из тела (живой предпросмотр по мере
    правки), структурную find_section-политику — из сохранённого конфига. KM —
    нота-плейсхолдер (реальная карта подставляется в pipeline в рантайме)."""
    cfg_path = request.app.state.config_path
    if req.require_find_section is not None:
        require_fs = req.require_find_section  # из несохранённого редактора (live-тумблер)
    else:
        require_fs = True
        try:
            rc = RetrievalConfig.model_validate(read_layered(cfg_path).get('retrieval') or {})
            fs = next((t for t in rc.tools if getattr(t, 'type', '') == 'find_section'), None)
            if fs is not None and getattr(fs, 'required', None) is not None:
                require_fs = bool(fs.required)
            elif getattr(rc.search, 'require_find_section', None) is not None:
                require_fs = bool(rc.search.require_find_section)
        except Exception:  # noqa: BLE001 — превью best-effort, дефолт REQUIRED
            pass
    sections = build_prompt_sections(
        corpus_description=req.corpus_description,
        require_find_section=require_fs,
        completeness_check=req.completeness_check,
        answer_style=req.answer_style,
        admin_instructions=req.admin_instructions or DEFAULT_ADMIN_INSTRUCTIONS,
        knowledge_map='[авто из корпуса — подставляется при запросе]',
        editor=True,
    )
    return {
        'ok': True,
        'sections': [
            {
                'label': s.label, 'kind': s.kind, 'field': s.field, 'text': s.text,
                'edit_kind': s.edit_kind, 'description': s.description, 'enabled': s.enabled,
            }
            for s in sections
        ],
    }


@router.get('/config')
async def get_retrieval(request: Request) -> dict[str, Any]:
    """Текущий retrieval-блок + полный effective-view с дефолтами Pydantic.

    Возвращает:
      - retrieval: то что в config.local.yml (primary+local merge), может быть None.
      - effective: тот же блок, но со всеми Pydantic-default'ами раскрытыми.
        Если retrieval=None и в llms[] есть хотя бы одна модель — effective
        собирается с первой LLM как agent+reranker (placeholder для UI).
      - llms: список LLM-имён для dropdown'ов.
    """
    cfg_path = request.app.state.config_path
    merged = read_layered(cfg_path)
    retrieval_raw = merged.get('retrieval')
    llms = [llm for llm in (merged.get('llms') or []) if llm.get('name')]

    # Build effective view: existing retrieval merged с Pydantic-defaults.
    # Если agent/reranker == None — подставляем placeholder с первой LLM из pool
    # (UI должен показать дефолтные значения temperature/max_tokens etc).
    placeholder = llms[0]['name'] if llms else ''
    try:
        cfg_obj = RetrievalConfig.model_validate(retrieval_raw or {})
    except Exception:
        # Невалидный baseline — отдаём как есть, UI покажет ошибку при сохранении.
        return {
            'retrieval': retrieval_raw,
            'effective': retrieval_raw or {},
            'llms': [{'name': llm.get('name'), 'model': llm.get('model'),
                      'capabilities': llm.get('capabilities') or ['text']}
                     for llm in llms],
        }

    # Заполнить placeholder для agent/reranker если они None.
    # Для placeholder LLM используем первую из pool — иначе валидатор min_length
    # упадёт. После dump перезапишем '' если пула нет.
    from morag.config import RetrievalAgentConfig, RetrievalRerankerConfig
    if cfg_obj.agent is None:
        cfg_obj.agent = RetrievalAgentConfig(llm=placeholder or 'placeholder')
    if cfg_obj.reranker is None:
        cfg_obj.reranker = RetrievalRerankerConfig(llm=placeholder or 'placeholder')
    effective = cfg_obj.model_dump()
    # Если LLM в pool нет — затираем placeholder в выводе (UI dropdown покажет «— выберите —»)
    if not placeholder:
        effective['agent']['llm'] = ''
        effective['reranker']['llm'] = ''

    return {
        'retrieval': retrieval_raw,
        'effective': effective,
        # дефолтная «роль» агента: console преднаполняет инпут «Описание корпуса»,
        # если corpus_description не задан, и сохраняет только при отличии от дефолта.
        'default_corpus_description': DEFAULT_CORPUS_DESCRIPTION,
        # дефолтные «Правила ответа»: тот же приём prefill+save-if-changed для answer_style.
        'default_answer_style': DEFAULT_ANSWER_STYLE,
        # дефолтные «Инструкции администратора» — тот же приём prefill+save-if-changed.
        'default_admin_instructions': DEFAULT_ADMIN_INSTRUCTIONS,
        'llms': [{'name': llm.get('name'), 'model': llm.get('model'),
                  'capabilities': llm.get('capabilities') or ['text']}
                 for llm in llms],
    }


@router.put('/config')
async def put_retrieval(req: RetrievalIn, request: Request) -> dict[str, Any]:
    """Записать retrieval-секцию в config.local.yml. Валидирует через Pydantic.

    Diff-write: сравниваем payload с effective-baseline (primary config.yml +
    Pydantic-defaults, БЕЗ существующего config.local.yml). Совпадающие поля
    отбрасываем — в local попадает только то, что юзер реально изменил
    относительно baseline. Если дельта пуста — удаляем ключ retrieval из local.

    Возвращает {'ok': True, 'restart_required': False} — pipeline подхватывает
    retrieval-правки hot-reload'ом по mtime конфига (см. morag_pipeline._maybe_reload),
    рестарт контейнера не нужен.
    """
    cfg_path = request.app.state.config_path
    incoming = _strip_none(req.model_dump())
    # Новый UI шлёт единый список tools — legacy glossary-блок тогда не пишем
    # (Pydantic-миграция дублей не создаст, но конфиг с обеими формами
    # двусмысленен для человека).
    if incoming.get('tools'):
        incoming.pop('glossary', None)

    # 1. Валидация: candidate-local = существующий local + наш retrieval-патч.
    #    Без существующего local validate_merged не увидит llms (которые юзер
    #    добавил через UI и которые лежат в config.local.yml) — ссылка на agent.llm
    #    упадёт даже если она валидна в реальной merged-картинке.
    current_local = read_local(cfg_path)
    candidate_local = {**current_local, 'retrieval': incoming}
    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    # 2. Считаем effective-baseline (primary + Pydantic defaults), БЕЗ local.
    baseline_effective = _baseline_retrieval_effective(cfg_path)

    # 3. diff: убираем из incoming поля, совпадающие с baseline_effective
    delta = _diff_against_baseline(incoming, baseline_effective)

    # 4. Записываем local: либо delta, либо удаляем секцию если пусто
    if delta:
        current_local['retrieval'] = delta
    elif 'retrieval' in current_local:
        del current_local['retrieval']
    write_local(cfg_path, current_local)

    return {'ok': True, 'restart_required': False}


def _baseline_retrieval_effective(cfg_path: str | Path) -> dict[str, Any]:
    """Effective retrieval из ТОЛЬКО baseline config.yml (без overlay).

    Pydantic-валидация подтягивает дефолты для незаполненных полей. Если
    baseline вообще не имеет секции retrieval — возвращаем {} (всё считается
    дельтой).
    """
    cfg_path = Path(cfg_path)
    with open(cfg_path, encoding='utf-8') as f:
        primary = yaml.safe_load(f) or {}
    raw = primary.get('retrieval')
    if not raw:
        return {}
    try:
        return RetrievalConfig.model_validate(raw).model_dump()
    except Exception:
        # Невалидная baseline — не блокируем save, считаем что дельта = всё.
        return {}


def _diff_against_baseline(incoming: Any, baseline: Any) -> Any:
    """Вернуть подмножество `incoming` где значения отличаются от `baseline`.

    Рекурсивно сравнивает dict'ы поэлементно. Для не-dict значений (scalar,
    list) — если equal, dropping; иначе оставляем целиком (списки не diff'аем
    поэлементно — слишком много краевых случаев).
    """
    if isinstance(incoming, dict) and isinstance(baseline, dict):
        result = {}
        for k, v in incoming.items():
            if k not in baseline:
                # baseline не знает этого поля — оставляем целиком
                result[k] = v
                continue
            sub_diff = _diff_against_baseline(v, baseline[k])
            # Для dict-различий: пустой dict значит «всё совпало» → не пишем
            if isinstance(sub_diff, dict) and not sub_diff:
                continue
            # Для не-dict: возвращённое значение либо равно incoming, либо
            # _SENTINEL_EQUAL означает «совпадает с baseline» → пропускаем
            if sub_diff is _SENTINEL_EQUAL:
                continue
            result[k] = sub_diff
        return result
    # Не-dict: scalar / list / None
    if incoming == baseline:
        return _SENTINEL_EQUAL
    return incoming


# Sentinel для рекурсивного _diff: означает «совпадает с baseline, пропустить»
_SENTINEL_EQUAL = object()


def _strip_none(data: Any) -> Any:
    """Рекурсивно удалить ключи со значением None ИЛИ пустыми dict'ами.

    Для Optional-полей UI отправляет None если юзер не заполнил — мы такие
    поля просто не пишем в YAML, чтобы дефолты Pydantic применились на
    стороне pipeline (а не записывать null'ы в config).

    Пустые dict'ы (все поля None) тоже дропаются: например, `find_section: {}`
    после strip означает «юзер ничего не задал в этой подсекции» — лучше
    не писать ключ совсем, чтобы Pipeline применил Pydantic-defaults.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if v is None:
                continue
            cleaned = _strip_none(v)
            # Пустой dict после очистки — пропускаем
            if isinstance(cleaned, dict) and not cleaned:
                continue
            result[k] = cleaned
        return result
    if isinstance(data, list):
        return [_strip_none(item) for item in data]
    return data
