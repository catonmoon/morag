"""Тесты GET /api/setup/checklist — onboarding-проверка окружения."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from services.console.app import create_app
from services.console.routes import setup as setup_routes


PRIMARY_CONFIG = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': '/tmp/docs'}],
    'llms': [
        {'name': 'main', 'base_url': 'http://host.docker.internal:11434/v1',
         'model': 'qwen3.5:9b', 'api_key': 'ollama',
         'capabilities': ['text', 'vision']},
    ],
    'indexing': {
        'llm': 'main', 'vision': 'main',
        'dense_embedder': {
            'model': 'qwen3-embedding:4b',
            'base_url': 'http://host.docker.internal:11434/v1',
            'dim': 2560,
        },
    },
    'qdrant': {'host': 'localhost', 'port': 16333},  # порт точно не занят
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
        app.state.indexer = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url='http://test') as ac:
            yield ac


class TestChecklist:

    async def test_ollama_installed_and_missing(self, client):
        # Патчим _query_ollama_installed напрямую (см. модуль setup_routes),
        # чтобы не трогать реальный httpx.AsyncClient (его юзает и тестовый клиент).
        async def fake_query(hosts):
            return {h: {'qwen3.5:9b', 'other:1b'} for h in hosts}

        with patch.object(setup_routes, '_query_ollama_installed', new=fake_query):
            r = await client.get('/api/setup/checklist')

        assert r.status_code == 200
        data = r.json()
        assert data['config_ok'] is True

        ollama = data['ollama']
        assert len(ollama) == 2

        llm = next(m for m in ollama if m['role'].startswith('llm/'))
        assert llm['model'] == 'qwen3.5:9b'
        assert llm['installed'] is True
        assert llm['host_reachable'] is True
        assert llm['pull_cmd'] == 'ollama pull qwen3.5:9b'

        emb = next(m for m in ollama if m['role'] == 'dense_embedder')
        assert emb['model'] == 'qwen3-embedding:4b'
        assert emb['installed'] is False
        assert emb['pull_cmd'] == 'ollama pull qwen3-embedding:4b'

    async def test_ollama_unreachable(self, client):
        async def fake_query(hosts):
            return {h: None for h in hosts}  # None = host недоступен

        with patch.object(setup_routes, '_query_ollama_installed', new=fake_query):
            r = await client.get('/api/setup/checklist')

        data = r.json()
        for m in data['ollama']:
            assert m['host_reachable'] is False
            assert m['installed'] is None

    async def test_qdrant_unreachable(self, client):
        async def fake_query(hosts):
            return {h: set() for h in hosts}

        with patch.object(setup_routes, '_query_ollama_installed', new=fake_query):
            r = await client.get('/api/setup/checklist')

        data = r.json()
        assert data['qdrant']['port'] == 16333
        assert data['qdrant']['reachable'] is False
        assert 'error' in data['qdrant']


class TestUnitFunctions:

    def test_ollama_api_root_for_known_hosts(self):
        assert (setup_routes._ollama_api_root('http://host.docker.internal:11434/v1')
                == 'http://host.docker.internal:11434')
        assert (setup_routes._ollama_api_root('http://localhost:11434/v1')
                == 'http://localhost:11434')

    def test_ollama_api_root_returns_none_for_non_ollama(self):
        assert setup_routes._ollama_api_root('https://api.x.ai/v1') is None
        assert setup_routes._ollama_api_root('https://openrouter.ai/api/v1') is None
        assert setup_routes._ollama_api_root('http://my-vllm.internal/v1') is None


class TestOllamaModels:

    async def test_returns_error_for_non_ollama_url(self, client):
        r = await client.get('/api/setup/ollama-models?base_url=https://api.openai.com/v1')
        d = r.json()
        assert d['ok'] is False
        assert 'не похож' in d['error'].lower()
        assert d['models'] == []

    async def test_returns_error_when_ollama_unreachable(self, client):
        # Реальный запрос на 1 — точно нет ollama. Должен вернуть ok=False с error.
        r = await client.get('/api/setup/ollama-models?base_url=http://127.0.0.1:1/v1')
        d = r.json()
        assert d['ok'] is False
        assert d['error']
        assert d['models'] == []
