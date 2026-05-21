"""Тесты IndexerClient через ASGI-mock control-plane.

Поднимаем фейковый FastAPI приложение которое имитирует control-plane endpoints,
прокидываем его в IndexerClient через ASGITransport. Реальной сети нет.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport
from pydantic import BaseModel

from services.console.indexer_client import (
    AlreadyRunning,
    IndexerClient,
    IndexerError,
    SetupIncomplete,
)


class _StartReq(BaseModel):
    reset: bool = False


class _StopReq(BaseModel):
    grace_seconds: int | None = None


def make_fake_app(state: dict) -> FastAPI:
    """state содержит заранее подготовленные ответы и контроль над ошибками."""
    app = FastAPI()

    @app.get('/control/status')
    async def status():
        if state.get('status_error'):
            raise HTTPException(status_code=500, detail='boom')
        return state.get('status_payload', {'is_running': False, 'run': None, 'progress': None})

    @app.post('/control/start')
    async def start(req: _StartReq):
        if state.get('start_409'):
            raise HTTPException(status_code=409, detail='Task already running')
        if state.get('start_412'):
            raise HTTPException(status_code=412, detail={'blockers': state['start_412']})
        state['last_start_reset'] = req.reset
        return {'started_at': 'T', 'kind': 'index', 'reset': req.reset}

    @app.get('/control/setup-status')
    async def setup_status():
        return state.get('setup_status_payload', {'ok': True, 'blockers': []})

    @app.post('/control/reload-schedule')
    async def reload_schedule():
        return state.get('reload_payload', {'schedule': None})

    @app.post('/control/stop')
    async def stop(req: _StopReq):
        state['last_stop_grace'] = req.grace_seconds
        return {'result': 'graceful'}

    @app.post('/control/kill')
    async def kill():
        return {'result': 'killed'}

    @app.post('/control/rebuild-km')
    async def rebuild():
        if state.get('rebuild_409'):
            raise HTTPException(status_code=409, detail='busy')
        return {'started_at': 'T', 'kind': 'rebuild_km'}

    return app


def make_client(state: dict) -> IndexerClient:
    return IndexerClient(
        'http://fake',
        transport=ASGITransport(app=make_fake_app(state)),
    )


class TestStatus:

    async def test_returns_payload(self):
        state = {'status_payload': {'is_running': True, 'run': None, 'progress': None}}
        c = make_client(state)
        s = await c.status()
        assert s['is_running'] is True

    async def test_raises_indexer_error_on_5xx(self):
        c = make_client({'status_error': True})
        with pytest.raises(IndexerError):
            await c.status()


class TestStart:

    async def test_passes_reset_flag(self):
        state = {}
        c = make_client(state)
        result = await c.start_index(reset=True)
        assert result == {'started_at': 'T', 'kind': 'index', 'reset': True}
        assert state['last_start_reset'] is True

    async def test_default_reset_false(self):
        state = {}
        c = make_client(state)
        await c.start_index()
        assert state['last_start_reset'] is False

    async def test_409_raises_already_running(self):
        c = make_client({'start_409': True})
        with pytest.raises(AlreadyRunning):
            await c.start_index()


class TestStop:

    async def test_passes_grace_seconds(self):
        state = {}
        c = make_client(state)
        await c.stop(grace_seconds=42)
        assert state['last_stop_grace'] == 42

    async def test_default_grace_seconds_omitted(self):
        """По умолчанию grace_seconds в body не отправляется — indexer применит
        config.indexing.stop_grace_seconds (или None — ждать без таймаута)."""
        state = {}
        c = make_client(state)
        await c.stop()
        assert state['last_stop_grace'] is None


class TestKill:

    async def test_returns_result(self):
        c = make_client({})
        r = await c.kill()
        assert r == {'result': 'killed'}


class TestRebuildKM:

    async def test_returns_run_info(self):
        c = make_client({})
        r = await c.start_rebuild_km()
        assert r['kind'] == 'rebuild_km'

    async def test_409(self):
        c = make_client({'rebuild_409': True})
        with pytest.raises(AlreadyRunning):
            await c.start_rebuild_km()


class TestSetupGate:

    async def test_412_raises_setup_incomplete_with_blockers(self):
        blockers = ['add at least one source', 'configure LLM']
        c = make_client({'start_412': blockers})
        with pytest.raises(SetupIncomplete) as exc:
            await c.start_index()
        assert exc.value.blockers == blockers

    async def test_setup_status(self):
        c = make_client({'setup_status_payload': {'ok': False, 'blockers': ['x']}})
        s = await c.setup_status()
        assert s['ok'] is False
        assert s['blockers'] == ['x']


class TestReloadSchedule:

    async def test_returns_active_schedule(self):
        c = make_client({'reload_payload': {'schedule': '0 */6 * * *'}})
        r = await c.reload_schedule()
        assert r['schedule'] == '0 */6 * * *'

    async def test_disabled_returns_none(self):
        c = make_client({'reload_payload': {'schedule': None}})
        r = await c.reload_schedule()
        assert r['schedule'] is None
