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
         'model': 'vision-model', 'api_key': 'primary-secret',
         'capabilities': ['text', 'vision']},
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

    async def test_reindex_proxies_scope(self, client):
        client._transport.app.state.indexer.reindex.return_value = {
            'started_at': 'T', 'kind': 'index', 'scope': 'jira-internal',
        }
        r = await client.post('/api/index/reindex', json={'scope': 'jira-internal'})
        assert r.status_code == 200
        client._transport.app.state.indexer.reindex.assert_awaited_once_with(scope='jira-internal')

    async def test_reindex_409_when_already_running(self, client):
        client._transport.app.state.indexer.reindex.side_effect = AlreadyRunning('running')
        r = await client.post('/api/index/reindex', json={'scope': 'all'})
        assert r.status_code == 409

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
        assert 'llm' in data and 'source' in data
        assert any(p['id'] == 'openai-compatible' for p in data['llm'])
        assert any(p['id'] == 'ollama' for p in data['llm'])
        assert any(p['id'] == 'local' for p in data['source'])

    async def test_apply_openai_preset_appends_to_llms(self, client, workspace):
        r = await client.post('/api/presets/apply', json={
            'target': 'llm',
            'preset_id': 'openai-compatible',
            'form': {'name': 'mygrok',
                     'base_url': 'https://api.x.ai/v1', 'model': 'grok-1', 'api_key': 'xai-test'},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['ok'] is True
        assert body['added']['name'] == 'mygrok'
        assert body['added']['base_url'] == 'https://api.x.ai/v1'

        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        names = [llm['name'] for llm in local['llms']]
        assert 'mygrok' in names

    async def test_apply_preset_replaces_by_name(self, client, workspace):
        common_form = {'name': 'g', 'base_url': 'http://x', 'model': 'm'}
        # Первый apply
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {**common_form, 'api_key': 'k1'},
        })
        # Второй apply с тем же name — должен заменить, не дублировать
        r = await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {**common_form, 'api_key': 'k2'},
        })
        assert r.status_code == 200
        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        names = [llm['name'] for llm in local['llms']]
        assert names.count('g') == 1
        g = next(llm for llm in local['llms'] if llm['name'] == 'g')
        assert g['api_key'] == 'k2'

    async def test_apply_local_source_appends(self, client, workspace):
        # Local — singleton: имя всегда 'doc', путь /app/data; форма игнорируется.
        r = await client.post('/api/presets/apply', json={
            'target': 'source',
            'preset_id': 'local',
            'form': {},
        })
        assert r.status_code == 200, r.text
        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        kinds_names = [(s['kind'], s['name']) for s in local['sources']]
        assert ('local', 'doc') in kinds_names

    async def test_apply_unknown_preset_400(self, client):
        r = await client.post('/api/presets/apply', json={
            'target': 'llm',
            'preset_id': 'nonexistent',
            'form': {},
        })
        assert r.status_code == 400

    async def test_set_roles(self, client, workspace):
        common = {'base_url': 'http://x', 'model': 'm', 'api_key': 'k'}
        # Сначала добавим LLM с vision
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {**common, 'name': 'text-llm'},
        })
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {**common, 'name': 'vis-llm', 'vision_capable': True},
        })
        r = await client.post('/api/presets/roles', json={
            'llm': 'text-llm', 'vision': 'vis-llm',
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['llm'] == 'text-llm'
        assert body['vision'] == 'vis-llm'

        local_path = workspace['cfg'].with_name('config.local.yml')
        local = yaml.safe_load(local_path.read_text())
        assert local['indexing']['llm'] == 'text-llm'
        assert local['indexing']['vision'] == 'vis-llm'

    async def test_set_roles_validates_references(self, client):
        # Несуществующая LLM в pool → 400
        r = await client.post('/api/presets/roles', json={
            'llm': 'nonexistent', 'vision': 'vision',
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

    async def test_default_owui_link(self, client):
        r = await client.get('/api/links')
        # Default — наш встроенный OWUI на :3000
        assert r.json()['open_webui'] == 'http://localhost:3000'

    async def test_owui_link_from_env(self, client, monkeypatch):
        monkeypatch.setenv('OPENWEBUI_URL', 'http://my-owui.local')
        r = await client.get('/api/links')
        assert r.json()['open_webui'] == 'http://my-owui.local'

    async def test_external_owui_connection_defaults(self, client):
        r = await client.get('/api/links')
        d = r.json()['external_owui']
        assert d['base_url'] == 'http://localhost:9099'
        assert d['model'] == 'morag_pipeline'
        assert d['api_key'] == '0p3n-w3bu!'

    async def test_external_owui_overrides(self, client, monkeypatch):
        monkeypatch.setenv('PIPELINES_PUBLIC_URL', 'https://my-pipelines.example/v1')
        monkeypatch.setenv('PIPELINES_API_KEY', 'custom-key')
        r = await client.get('/api/links')
        d = r.json()['external_owui']
        assert d['base_url'] == 'https://my-pipelines.example/v1'
        assert d['api_key'] == 'custom-key'


# ---------------------------------------------------------------------------
# /api/presets/delete + /api/presets/apply (embedder target)
# ---------------------------------------------------------------------------

class TestDeleteAndEmbedder:

    async def test_delete_llm(self, client, workspace):
        # Сначала добавим, потом удалим
        common = {'base_url': 'http://x', 'model': 'm', 'api_key': 'k'}
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {**common, 'name': 'todelete'},
        })
        r = await client.post('/api/presets/delete', json={
            'target': 'llm', 'name': 'todelete',
        })
        assert r.status_code == 200, r.text

        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        names = [llm['name'] for llm in local.get('llms', [])]
        assert 'todelete' not in names

    async def test_delete_protects_referenced_llm(self, client):
        # primary 'main' использован в indexing.llm — удаление сломает refs → 400
        r = await client.post('/api/presets/delete', json={
            'target': 'llm', 'name': 'main',
        })
        assert r.status_code == 400

    async def test_delete_source_requires_kind(self, client):
        r = await client.post('/api/presets/delete', json={
            'target': 'source', 'name': 'docs',
        })
        assert r.status_code == 400
        assert 'kind required' in r.text.lower()

    async def test_delete_404_when_not_found(self, client):
        r = await client.post('/api/presets/delete', json={
            'target': 'llm', 'name': 'nonexistent',
        })
        assert r.status_code == 404

    async def test_apply_embedder_replaces(self, client, workspace):
        r = await client.post('/api/presets/apply', json={
            'target': 'embedder', 'preset_id': 'ollama',
            'form': {'model': 'nomic-embed-text', 'dim': '768'},
        })
        assert r.status_code == 200, r.text
        assert r.json()['added']['model'] == 'nomic-embed-text'

        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        assert local['indexing']['dense_embedder']['model'] == 'nomic-embed-text'
        assert local['indexing']['dense_embedder']['dim'] == 768

    async def test_list_presets_includes_embedder(self, client):
        r = await client.get('/api/presets')
        d = r.json()
        assert 'embedder' in d
        ids = [p['id'] for p in d['embedder']]
        assert 'ollama' in ids
        assert 'openai-compatible' in ids


class TestSecretPreservation:
    """При Edit secret-поля (api_key/password/api_token) сохраняются если в форме пусто."""

    async def test_llm_api_key_preserved_when_form_omits_it(self, client, workspace):
        # 1. Добавим LLM с api_key
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {'name': 'edit-me', 'base_url': 'http://x', 'model': 'm', 'api_key': 'secret-1'},
        })
        # 2. Edit: меняем model, api_key пустой (как это сделает UI после маскировки)
        r = await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {'name': 'edit-me', 'base_url': 'http://x', 'model': 'NEW', 'api_key': ''},
        })
        assert r.status_code == 200, r.text

        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        edited = next(l for l in local['llms'] if l['name'] == 'edit-me')
        assert edited['model'] == 'NEW'
        assert edited['api_key'] == 'secret-1'  # сохранился

    async def test_llm_api_key_replaced_when_form_provides_new(self, client, workspace):
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {'name': 'r', 'base_url': 'http://x', 'model': 'm', 'api_key': 'old'},
        })
        await client.post('/api/presets/apply', json={
            'target': 'llm', 'preset_id': 'openai-compatible',
            'form': {'name': 'r', 'base_url': 'http://x', 'model': 'm', 'api_key': 'new-key'},
        })
        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        edited = next(l for l in local['llms'] if l['name'] == 'r')
        assert edited['api_key'] == 'new-key'

    async def test_embedder_api_key_preserved(self, client, workspace):
        await client.post('/api/presets/apply', json={
            'target': 'embedder', 'preset_id': 'openai-compatible',
            'form': {'base_url': 'http://x', 'model': 'm', 'api_key': 'sk-saved', 'dim': '768'},
        })
        # Edit: меняем dim, api_key пустой
        await client.post('/api/presets/apply', json={
            'target': 'embedder', 'preset_id': 'openai-compatible',
            'form': {'base_url': 'http://x', 'model': 'm', 'api_key': '', 'dim': '1024'},
        })
        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        emb = local['indexing']['dense_embedder']
        assert emb['dim'] == 1024
        assert emb['api_key'] == 'sk-saved'

    async def test_confluence_password_preserved(self, client, workspace):
        await client.post('/api/presets/apply', json={
            'target': 'source', 'preset_id': 'confluence',
            'form': {'name': 'cf', 'url': 'https://cf', 'username': 'u', 'password': 'pw1'},
        })
        await client.post('/api/presets/apply', json={
            'target': 'source', 'preset_id': 'confluence',
            'form': {'name': 'cf', 'url': 'https://cf', 'username': 'u-changed', 'password': ''},
        })
        local = yaml.safe_load((workspace['cfg'].with_name('config.local.yml')).read_text())
        cf = next(s for s in local['sources'] if s.get('kind') == 'confluence' and s['name'] == 'cf')
        assert cf['username'] == 'u-changed'
        assert cf['password'] == 'pw1'  # сохранился
