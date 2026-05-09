"""Retrieval-конфиг (секция config.local.yml.retrieval).

Источник истины для pipeline (агентский RAG). Pipeline читает config.yml при
старте контейнера; OWUI Valves остаются как override-механизм, но настройки
по умолчанию приходят отсюда.

Изменения требуют `docker compose restart pipelines` — pipeline не перечитывает
конфиг в runtime (см. CLAUDE.md). Console UI должен показывать соответствующий
баннер после сохранения.
"""
from __future__ import annotations

from typing import Any

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from morag.config import RetrievalConfig
from services.console.config_io import read_layered, read_local, validate_merged, write_local

router = APIRouter()


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
    citation_max_chars: int | None = None
    answer_max_tokens: int | None = None
    find_section: FindSectionIn | None = None


class FeaturesIn(BaseModel):
    enable_diversity_nudge: bool | None = None


class PromptsIn(BaseModel):
    admin_instructions: str = ''


class RetrievalIn(BaseModel):
    agent: RoleIn | None = None
    reranker: RoleIn | None = None
    search: SearchIn = SearchIn()
    features: FeaturesIn = FeaturesIn()
    prompts: PromptsIn = PromptsIn()
    http_timeout: int | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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

    Возвращает {'ok': True, 'restart_required': True}.
    """
    cfg_path = request.app.state.config_path
    incoming = _strip_none(req.model_dump())

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

    return {'ok': True, 'restart_required': True}


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
