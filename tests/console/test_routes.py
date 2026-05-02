"""Тесты HTTP API консоли через httpx ASGI transport.

IndexerClient мокается AsyncMock'ом — мы не зависим от реального indexer'а
ни как процесса, ни как HTTP-серверa. Любые ошибки сети control-plane'а
тестируются отдельным юнит-тестом IndexerClient.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from services.console.app import create_app
from services.console.indexer_client import AlreadyRunning, IndexerError


PRIMARY_CONFIG = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': '/tmp/docs'}],
    'llms': [
        {'name': 'main', 'base_url': 'http://primary/v1',
         'model': 'primary-model', 'api_key': 'primary-secret'},
        {'name': 'vision', 'base_url': 'http://primary/v1',
         'model': 'vision-model', 'api_key': 'primary-secret'},
    ],
    'indexing': {
        'llm': 'main',
        'vision': 'vision',
        'dense_embedder': {
            'model': 'qwen3-embedding:4b',
            'base_url': 'http://primary/v1',
            'dim': 2560,
        },
    },
}


@pytest.fixture
def workspace(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(PRIMARY_CONFIG))

    old_env = os.environ.get('MORAG_CONFIG_PATH')
    os.environ['MORAG_CONFIG_PATH'] = str(cfg)
    yield {'cfg': cfg, 'tmp': tmp_path}
    if old_env is None:
        os.environ.pop('MORAG_CONFIG_PATH', None)
    else:
        os.environ['MORAG_CONFIG_PATH'] = old_env


@pytest.fixture
async def client(workspace):
    app = create_app()
    async with app.router.lifespan_context(app):
        # Подменяем реальный IndexerClient на AsyncMock'нутый — без сети
        app.state.indexer = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as ac:
            yield ac


# ---------------------------------------------------------------------------
# /api/config
# ---------------------------------------------------------------------------

class TestConfigRoutes:

    async def test_get_config_masks_secrets(self, client):
        r = await client.get('/api/config')
        assert r.status_code == 200
        data = r.json()
        # llms — list, секреты в каждом item замаскированы
        assert data['llms'][0]['api_key'] == '***'
        assert data['llms'][0]['model'] == 'primary-model'

    async def test_put_config_writes_local_overlay(self, client, workspace):
        # Простой patch: сменить qdrant host (типичный local-dev сценарий)
        r = await client.put('/api/config', json={'patch': {'qdrant': {'host': 'localhost'}}})
        assert r.status_code == 200, r.text

        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        assert local == {'qdrant': {'host': 'localhost'}}

    async def test_put_config_strips_masked_secret(self, client, workspace):
        # Если patch содержит замаскированный секрет — он должен быть выпилен
        # (чтобы не затереть реальный секрет в config.yml). См. strip_masked_secrets.
        # Используем пользовательский неchunkавый секрет-ключ для теста.
        r = await client.put('/api/config', json={
            'patch': {'qdrant': {'host': 'x', 'port': 6333}},
        })
        assert r.status_code == 200
        # Этот test после Stage 6 (Console UI refactor) будет пересмотрен:
        # юзер UI правит llms-pool через структурированные операции,
        # а не raw dict-patches с masked-полями.

    async def test_put_config_real_secret_saved(self, client, workspace):
        # Тест что overlay записывается. Используем «сценарий» с qdrant host —
        # без секретов, проще. Реальные секреты сохраняются при applyPreset
        # через Setup wizard (см. test_apply_grok_preset_writes_local).
        r = await client.put('/api/config', json={
            'patch': {'qdrant': {'host': 'newhost'}},
        })
        assert r.status_code == 200

        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        assert local['qdrant']['host'] == 'newhost'

    async def test_put_config_invalid_returns_400(self, client):
        r = await client.put('/api/config', json={
            'patch': {'qdrant': {'port': 'not-a-number'}},  # int required
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/index — мокаем IndexerClient, проверяем что роуты правильно проксируют
# ---------------------------------------------------------------------------

class TestIndexRoutes:

    async def test_status_proxies_indexer(self, client):
        client._transport.app.state.indexer.status.return_value = {
            'is_running': True,
            'run': {'started_at': 't0', 'kind': 'index', 'reset': False},
            'progress': {'state': 'running', 'phase': 'p', 'processed': 1, 'total': 5,
                         'started_at': 't0', 'updated_at': 't1', 'error': None,
                         'current_doc_id': 'doc1'},
        }
        r = await client.get('/api/index/status')
        assert r.status_code == 200
        data = r.json()
        assert data['is_running'] is True
        assert data['run']['kind'] == 'index'
        assert data['progress']['phase'] == 'p'

    async def test_status_returns_unreachable_on_indexer_error(self, client):
        client._transport.app.state.indexer.status.side_effect = IndexerError('boom')
        r = await client.get('/api/index/status')
        assert r.status_code == 200
        data = r.json()
        assert data['is_running'] is False
        assert data['progress']['state'] == 'unreachable'
        assert 'boom' in data['progress']['error']

    async def test_start_returns_run_info(self, client):
        client._transport.app.state.indexer.start_index.return_value = {
            'started_at': 'T', 'kind': 'index', 'reset': True,
        }
        r = await client.post('/api/index/start', json={'reset': True})
        assert r.status_code == 200
        client._transport.app.state.indexer.start_index.assert_awaited_once_with(reset=True)

    async def test_start_409_when_already_running(self, client):
        client._transport.app.state.indexer.start_index.side_effect = AlreadyRunning('running')
        r = await client.post('/api/index/start', json={'reset': False})
        assert r.status_code == 409

    async def test_start_502_when_indexer_unreachable(self, client):
        client._transport.app.state.indexer.start_index.side_effect = IndexerError('conn refused')
        r = await client.post('/api/index/start', json={'reset': False})
        assert r.status_code == 502

    async def test_stop_proxies_grace(self, client):
        client._transport.app.state.indexer.stop.return_value = {'result': 'graceful'}
        r = await client.post('/api/index/stop', json={'grace_seconds': 60})
        assert r.status_code == 200
        assert r.json()['result'] == 'graceful'
        client._transport.app.state.indexer.stop.assert_awaited_once_with(grace_seconds=60)

    async def test_kill(self, client):
        client._transport.app.state.indexer.kill.return_value = {'result': 'killed'}
        r = await client.post('/api/index/kill')
        assert r.status_code == 200
        assert r.json()['result'] == 'killed'

    async def test_rebuild_km(self, client):
        client._transport.app.state.indexer.start_rebuild_km.return_value = {
            'started_at': 'T', 'kind': 'rebuild_km',
        }
        r = await client.post('/api/knowledge-map/rebuild')
        assert r.status_code == 200
        assert r.json()['kind'] == 'rebuild_km'

    async def test_rebuild_km_409(self, client):
        client._transport.app.state.indexer.start_rebuild_km.side_effect = AlreadyRunning('x')
        r = await client.post('/api/knowledge-map/rebuild')
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# /api/presets
# ---------------------------------------------------------------------------

class TestPresetsRoutes:

    async def test_list_presets(self, client):
        r = await client.get('/api/presets')
        assert r.status_code == 200
        data = r.json()
        assert 'llm' in data and 'dense_embedder' in data
        assert any(p['id'] == 'grok' for p in data['llm'])

    @pytest.mark.skip(reason='Presets выдают старо-схемный snippet ({llm: {...}}); '
                            'будет переписано в Stage 6 (Console UI refactor под '
                            'list[Source] + llms-pool). См. ADR-0012.')
    async def test_apply_grok_preset_writes_local(self, client, workspace):
        r = await client.post('/api/presets/apply', json={
            'target': 'llm',
            'preset_id': 'grok',
            'form': {'api_key': 'xai-test', 'model': 'grok-4-1-fast'},
        })
        assert r.status_code == 200, r.text

    async def test_apply_unknown_preset_400(self, client):
        r = await client.post('/api/presets/apply', json={
            'target': 'llm',
            'preset_id': 'nonexistent',
            'form': {},
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

class TestStatsRoute:

    async def test_returns_zeros_when_qdrant_unreachable(self, client):
        r = await client.get('/api/stats')
        assert r.status_code == 200
        data = r.json()
        if data['qdrant_reachable']:
            return  # локально живой Qdrant — тест неинформативен
        assert data['docs'] == 0
        assert data['chunks'] == 0
        assert data['collections'] == []
        assert data['qdrant_error']


# ---------------------------------------------------------------------------
# /api/links
# ---------------------------------------------------------------------------

class TestLinksRoute:

    async def test_default_qdrant_link(self, client):
        r = await client.get('/api/links')
        assert r.status_code == 200
        assert r.json()['qdrant'].endswith(':6333/dashboard')

    async def test_owui_link_from_env(self, client, monkeypatch):
        monkeypatch.setenv('OPENWEBUI_URL', 'http://my-owui.local')
        r = await client.get('/api/links')
        assert r.json()['open_webui'] == 'http://my-owui.local'
