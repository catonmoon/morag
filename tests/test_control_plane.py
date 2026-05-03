"""Тесты для IndexerControlPlane.

Используем фейковую `run_index` coroutine (асинхронный sleep с уважением к
cancel_event) — без реальной индексации, без mocks библиотечного уровня.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from morag.control_plane import (
    AlreadyRunning,
    IndexerControlPlane,
)


async def _fake_index(
    duration: float = 5.0,
    cancel_event: asyncio.Event | None = None,
    status_reporter=None,
    reset: bool = False,
):
    """Имитирует cmd_index: пишет в reporter, спит, проверяет cancel_event."""
    if status_reporter is not None:
        status_reporter.start_phase('fake_indexing', 1)
    deadline = asyncio.get_event_loop().time() + duration
    while asyncio.get_event_loop().time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            if status_reporter is not None:
                status_reporter.finish('cancelled')
            return
        await asyncio.sleep(0.05)
    if status_reporter is not None:
        status_reporter.document_done('only-doc')
        status_reporter.finish('completed')


_VALID_CONFIG = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': '/x'}],
    'llms': [
        {'name': 'main', 'base_url': 'http://x/v1', 'model': 'm', 'api_key': 'k',
         'capabilities': ['text', 'vision']},
    ],
    'indexing': {
        'llm': 'main', 'vision': 'main',
        'dense_embedder': {'model': 'e', 'base_url': 'http://x/v1', 'dim': 8},
    },
}


def make_cp(tmp_path: Path, *, duration: float = 5.0,
            setup_complete: bool = True) -> IndexerControlPlane:
    """Создаёт control-plane с временным config'ом.

    `setup_complete=True` → primary + local (gate проходит)
    `setup_complete=False` → только primary (gate fail: local missing)
    """
    import yaml as _yaml
    cfg = tmp_path / 'config.yml'
    cfg.write_text(_yaml.safe_dump(_VALID_CONFIG))
    if setup_complete:
        # Реальный контент — пустой/только-комментарии файл считается «не настроен»
        (tmp_path / 'config.local.yml').write_text('qdrant:\n  host: customhost\n')
    return IndexerControlPlane(
        config_path=str(cfg),
        status_file_path=tmp_path / 'state.json',
        run_index=lambda **kw: _fake_index(duration=duration, **kw),
        run_rebuild_km=lambda **kw: _fake_index(duration=duration, **kw),
    )


class TestStart:

    async def test_start_makes_task_running(self, tmp_path):
        cp = make_cp(tmp_path, duration=5.0)
        info = await cp.start_index()
        try:
            assert info.kind == 'index'
            assert cp.is_running()
            assert cp.status()['is_running'] is True
        finally:
            await cp.kill()

    async def test_start_twice_raises(self, tmp_path):
        cp = make_cp(tmp_path, duration=5.0)
        await cp.start_index()
        try:
            with pytest.raises(AlreadyRunning):
                await cp.start_index()
        finally:
            await cp.kill()

    async def test_start_after_completion_works(self, tmp_path):
        cp = make_cp(tmp_path, duration=0.1)
        await cp.start_index()
        await asyncio.sleep(0.3)
        assert not cp.is_running()
        # Можно стартовать заново
        info = await cp.start_index()
        assert info.kind == 'index'
        await cp.kill()

    async def test_start_rebuild_km(self, tmp_path):
        cp = make_cp(tmp_path, duration=5.0)
        info = await cp.start_rebuild_km()
        try:
            assert info.kind == 'rebuild_km'
        finally:
            await cp.kill()


class TestStop:

    async def test_stop_graceful(self, tmp_path):
        cp = make_cp(tmp_path, duration=10.0)
        await cp.start_index()
        await asyncio.sleep(0.1)  # дать start_phase успеть
        result = await cp.stop(grace_seconds=2)
        assert result == 'graceful'
        assert not cp.is_running()

    async def test_stop_when_idle(self, tmp_path):
        cp = make_cp(tmp_path)
        result = await cp.stop()
        assert result == 'not_running'

    async def test_stop_writes_cancelled_to_state_file(self, tmp_path):
        import json
        cp = make_cp(tmp_path, duration=10.0)
        await cp.start_index()
        await asyncio.sleep(0.1)
        await cp.stop(grace_seconds=2)
        data = json.loads((tmp_path / 'state.json').read_text())
        assert data['state'] == 'cancelled'


class TestKill:

    async def test_kill_running_task(self, tmp_path):
        cp = make_cp(tmp_path, duration=10.0)
        await cp.start_index()
        result = await cp.kill()
        assert result == 'killed'

    async def test_kill_when_idle(self, tmp_path):
        cp = make_cp(tmp_path)
        assert (await cp.kill()) == 'not_running'


class TestStatus:

    async def test_status_idle(self, tmp_path):
        cp = make_cp(tmp_path)
        s = cp.status()
        assert s['is_running'] is False
        assert s['run'] is None
        assert s['progress'] is None

    async def test_status_running_includes_run(self, tmp_path):
        cp = make_cp(tmp_path, duration=5.0)
        info = await cp.start_index(reset=True)
        try:
            s = cp.status()
            assert s['is_running'] is True
            assert s['run']['kind'] == 'index'
            assert s['run']['reset'] is True
            assert s['run']['started_at'] == info.started_at
        finally:
            await cp.kill()

    async def test_status_progress_from_state_file(self, tmp_path):
        cp = make_cp(tmp_path, duration=5.0)
        await cp.start_index()
        try:
            # дать время записать start_phase
            for _ in range(20):
                s = cp.status()
                if s['progress'] and s['progress'].get('state') == 'running':
                    return
                await asyncio.sleep(0.05)
            pytest.fail(f'progress never showed running, last={s}')
        finally:
            await cp.kill()


class TestSetupGate:
    """start_index/rebuild_km блокируются если setup-gate не пройден."""

    async def test_start_index_raises_setup_incomplete(self, tmp_path):
        from morag.setup_gate import SetupIncomplete
        cp = make_cp(tmp_path, setup_complete=False)
        with pytest.raises(SetupIncomplete) as exc:
            await cp.start_index()
        assert exc.value.blockers
        assert 'Setup' in exc.value.blockers[0]
        assert cp.is_running() is False

    async def test_start_rebuild_km_raises_setup_incomplete(self, tmp_path):
        from morag.setup_gate import SetupIncomplete
        cp = make_cp(tmp_path, setup_complete=False)
        with pytest.raises(SetupIncomplete):
            await cp.start_rebuild_km()

    async def test_setup_status_reports_state(self, tmp_path):
        a = tmp_path / 'a'
        b = tmp_path / 'b'
        a.mkdir()
        b.mkdir()
        cp_blocked = make_cp(a, setup_complete=False)
        cp_ok = make_cp(b, setup_complete=True)
        assert cp_blocked.setup_status()['ok'] is False
        assert cp_blocked.setup_status()['blockers']
        assert cp_ok.setup_status()['ok'] is True
        assert cp_ok.setup_status()['blockers'] == []
