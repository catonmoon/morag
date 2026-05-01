"""GET /api/presets и POST /api/presets/apply — wizard-пресеты провайдеров."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from services.console.config_io import patch_local, validate_merged
from services.console.presets import (
    DENSE_EMBEDDER_PRESETS,
    LLM_PRESETS,
    apply_preset,
    serialize_preset,
)

router = APIRouter()


class ApplyPresetRequest(BaseModel):
    target: Literal['llm', 'llm_vision', 'dense_embedder', 'sparse_embedder']
    preset_id: str
    form: dict[str, Any]


@router.get('')
async def list_presets() -> dict[str, list[dict[str, Any]]]:
    return {
        'llm': [serialize_preset(p) for p in LLM_PRESETS],
        'dense_embedder': [serialize_preset(p) for p in DENSE_EMBEDDER_PRESETS],
    }


@router.post('/apply')
async def apply(req: ApplyPresetRequest, request: Request) -> dict[str, Any]:
    cfg_path = request.app.state.config_path
    try:
        snippet = apply_preset(req.target, req.preset_id, req.form)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # Например, отсутствует обязательное поле формы
        raise HTTPException(status_code=400, detail=f'Bad form data: {e}') from e

    # Валидируем что merged конфиг будет корректен
    from morag.config import _deep_merge

    from services.console.config_io import read_local
    candidate_local = _deep_merge(read_local(cfg_path), snippet)
    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e

    new_local = patch_local(cfg_path, snippet)
    return {'ok': True, 'local': new_local}
