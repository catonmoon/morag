"""GET /api/presets и POST /api/presets/apply — wizard-пресеты.

apply route добавляет item в `llms[]` или `sources[]` в config.local.yml,
с replace-by-(name) для idempotency. Не deep-merge'ит top-level (lists в
overlay перезаписываются целиком — пишем новый полный список).
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from services.console.config_io import patch_local, read_layered, read_local, validate_merged
from services.console.presets import (
    LLM_PRESETS,
    SOURCE_PRESETS,
    apply_preset,
    serialize_preset,
)

router = APIRouter()


class ApplyPresetRequest(BaseModel):
    target: Literal['llm', 'source']
    preset_id: str
    form: dict[str, Any]


@router.get('')
async def list_presets() -> dict[str, list[dict[str, Any]]]:
    return {
        'llm': [serialize_preset(p) for p in LLM_PRESETS],
        'source': [serialize_preset(p) for p in SOURCE_PRESETS],
    }


@router.post('/apply')
async def apply(req: ApplyPresetRequest, request: Request) -> dict[str, Any]:
    """Добавить новый item в llms[] или sources[] в config.local.yml.

    Replace-by-name semantics: если item с таким name (для llms) или
    (kind, name) (для sources) уже существует — заменяется. Иначе — append.
    """
    cfg_path = request.app.state.config_path
    try:
        item = apply_preset(req.target, req.preset_id, req.form)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Bad form data: {e}') from e

    # Читаем merged-вид (primary + local) для апсерта — потому что lists
    # в deep_merge перезаписываются целиком: если запишем в local только
    # новый item, primary-shipped llms/sources исчезнут из merged-вида и
    # сломают indexing.llm/.vision references.
    current_local = read_local(cfg_path)
    merged_view = read_layered(cfg_path)
    list_field = 'llms' if req.target == 'llm' else 'sources'
    current_list = list(merged_view.get(list_field, []))

    if req.target == 'llm':
        merged_list = _upsert_by_name(current_list, item, key='name')
    else:  # source
        merged_list = _upsert_by_name(current_list, item, key=('kind', 'name'))

    candidate_local = {**current_local, list_field: merged_list}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    # Lists в deep_merge перезаписываются целиком — патчим прямо весь список
    patch_local(cfg_path, {list_field: merged_list})
    return {'ok': True, 'added': item, 'list_size': len(merged_list)}


def _upsert_by_name(items: list[dict], new_item: dict, key) -> list[dict]:
    """Replace item с тем же `key` или append если не найден.

    `key` может быть строкой (для llms.name) или tuple (для sources.kind+name).
    """
    if isinstance(key, str):
        def get_key(x): return x.get(key)
    else:
        def get_key(x): return tuple(x.get(k) for k in key)

    new_key = get_key(new_item)
    result = []
    replaced = False
    for existing in items:
        if get_key(existing) == new_key:
            result.append(new_item)
            replaced = True
        else:
            result.append(existing)
    if not replaced:
        result.append(new_item)
    return result


# ---------------------------------------------------------------------------
# POST /api/presets/roles — set indexing.llm + indexing.vision
# ---------------------------------------------------------------------------

class SetRolesRequest(BaseModel):
    llm: str            # имя из llms-pool для default-роли (text)
    vision: str         # имя из llms-pool для vision-роли


@router.post('/roles')
async def set_roles(req: SetRolesRequest, request: Request) -> dict[str, Any]:
    """Установить indexing.llm (default text) и indexing.vision."""
    cfg_path = request.app.state.config_path
    current_local = read_local(cfg_path)

    indexing = dict(current_local.get('indexing') or {})
    indexing['llm'] = req.llm
    indexing['vision'] = req.vision
    candidate_local = {**current_local, 'indexing': indexing}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    patch_local(cfg_path, {'indexing': {'llm': req.llm, 'vision': req.vision}})
    return {'ok': True, 'llm': req.llm, 'vision': req.vision}
