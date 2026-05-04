"""Тесты GET/PUT /api/schedule — управление cron из UI."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from services.console.app import create_app


PRIMARY_CONFIG = {
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


@pytest.fixture
def workspace(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(PRIMARY_CONFIG))
    old = os.environ.get('MORAG_CONFIG_PATH')
    os.environ['MORAG_CONFIG_PATH'] = str(cfg)
    yield {'cfg': cfg, 'tmp': tmp_path}
    if old is None:
        os.environ.pop('MORAG_CONFIG_PATH', None)
    else:
        os.environ['MORAG_CONFIG_PATH'] = old


@pytest.fixture
async def client(workspace):
    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.indexer = AsyncMock()
        app.state.indexer.reload_schedule.return_value = {'schedule': '0 */6 * * *'}
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://t') as c:
            yield c


class TestSchedule:

    async def test_get_default_disabled(self, client):
        r = await client.get('/api/schedule')
        assert r.status_code == 200
        d = r.json()
        assert d['enabled'] is False
        assert d['cron'] is None

    async def test_put_enables_schedule(self, client, workspace):
        r = await client.put('/api/schedule', json={'cron': '0 */6 * * *'})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['cron'] == '0 */6 * * *'
        assert d['enabled'] is True

        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        assert local['indexing']['schedule'] == '0 */6 * * *'

    async def test_put_disables_with_null(self, client, workspace):
        # Сначала включаем
        await client.put('/api/schedule', json={'cron': '0 */6 * * *'})
        # Потом выключаем
        r = await client.put('/api/schedule', json={'cron': None})
        assert r.status_code == 200
        assert r.json()['enabled'] is False

        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        # schedule должен быть None в local (или вовсе отсутствовать)
        assert local.get('indexing', {}).get('schedule') in (None,)

    async def test_put_invalid_cron_returns_400(self, client):
        r = await client.put('/api/schedule', json={'cron': 'not-a-cron'})
        assert r.status_code == 400
        assert 'Invalid cron' in r.text

    async def test_put_calls_indexer_reload(self, client):
        await client.put('/api/schedule', json={'cron': '0 */6 * * *'})
        client._transport.app.state.indexer.reload_schedule.assert_awaited_once()

    async def test_put_handles_indexer_unreachable(self, client):
        from services.console.indexer_client import IndexerError
        client._transport.app.state.indexer.reload_schedule.side_effect = IndexerError('down')
        r = await client.put('/api/schedule', json={'cron': '0 */6 * * *'})
        # Сохранение прошло, но reload_error в ответе
        assert r.status_code == 200
        assert 'reload_error' in r.json()
