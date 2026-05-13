"""Cron schedule management — read/write `indexing.schedule` + hot-reload."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from services.console.config_io import patch_local, read_layered, validate_merged
from services.console.indexer_client import IndexerError

router = APIRouter()


class ScheduleRequest(BaseModel):
    cron: str | None  # None = выключить cron; иначе — crontab expression


@router.get('')
async def get_schedule(request: Request) -> dict[str, Any]:
    """Текущее расписание из merged config."""
    cfg_path = request.app.state.config_path
    merged = read_layered(cfg_path)
    cron = (merged.get('indexing') or {}).get('schedule')
    return {'cron': cron, 'enabled': bool(cron)}


@router.put('')
async def set_schedule(req: ScheduleRequest, request: Request) -> dict[str, Any]:
    """Обновить indexing.schedule в config.local.yml + hot-reload в indexer'е."""
    cfg_path = request.app.state.config_path

    # Валидация cron-выражения происходит в indexer'е (там есть apscheduler;
    # console-образ намеренно без [indexing] extras). Если выражение невалидное —
    # indexer вернёт ошибку через reload_schedule ниже, мы её прокидываем юзеру.
    cron = req.cron.strip() if req.cron else None

    # patch local + полная Pydantic-валидация merged
    from services.console.config_io import read_local
    current_local = read_local(cfg_path)
    indexing = dict(current_local.get('indexing') or {})
    if cron:
        indexing['schedule'] = cron
    else:
        indexing.pop('schedule', None)
    candidate_local = {**current_local, 'indexing': indexing}

    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    # patch_local deep-merge'ит, схедул прокинется
    patch = {'indexing': {'schedule': cron}} if cron else {'indexing': {'schedule': None}}
    patch_local(cfg_path, patch)

    # Hot-reload в indexer'е (best-effort: если indexer down, запомним и
    # перечитает на старте сам)
    try:
        result = await request.app.state.indexer.reload_schedule()
        return {'cron': cron, 'enabled': bool(cron), 'reload': result}
    except IndexerError as e:
        return {'cron': cron, 'enabled': bool(cron), 'reload_error': str(e)}
