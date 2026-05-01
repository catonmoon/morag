"""Управление индексацией: status / start / stop / kill.

Тонкая обёртка над IndexerClient — endpoints просто проксируют запросы к
control-plane'у в morag-indexer.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.console.indexer_client import (
    DEFAULT_STOP_GRACE_SECONDS,
    AlreadyRunning,
    IndexerError,
)

router = APIRouter()


class StartRequest(BaseModel):
    reset: bool = False


class StopRequest(BaseModel):
    grace_seconds: int = DEFAULT_STOP_GRACE_SECONDS


class IndexStatusResponse(BaseModel):
    is_running: bool
    run: dict[str, Any] | None
    progress: dict[str, Any] | None


class StopResponse(BaseModel):
    result: Literal['graceful', 'killed', 'not_running']


@router.get('/status', response_model=IndexStatusResponse)
async def get_status(request: Request) -> IndexStatusResponse:
    try:
        s = await request.app.state.indexer.status()
    except IndexerError as e:
        # Indexer недоступен — возвращаем «idle» с предупреждением вместо 500
        return IndexStatusResponse(
            is_running=False, run=None,
            progress={'state': 'unreachable', 'phase': '', 'processed': 0, 'total': 0,
                      'started_at': None, 'updated_at': None,
                      'error': str(e), 'current_doc_id': None},
        )
    return IndexStatusResponse(**s)


@router.post('/start')
async def start_index(req: StartRequest, request: Request) -> dict[str, Any]:
    try:
        return await request.app.state.indexer.start_index(reset=req.reset)
    except AlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except IndexerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post('/stop', response_model=StopResponse)
async def stop_index(req: StopRequest, request: Request) -> StopResponse:
    try:
        r = await request.app.state.indexer.stop(grace_seconds=req.grace_seconds)
    except IndexerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return StopResponse(result=r['result'])


@router.post('/kill', response_model=StopResponse)
async def kill_index(request: Request) -> StopResponse:
    try:
        r = await request.app.state.indexer.kill()
    except IndexerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return StopResponse(result=r['result'])
