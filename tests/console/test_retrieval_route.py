"""Тесты GET/PUT /api/retrieval/config."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from services.console.app import create_app


PRIMARY_CONFIG = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': '/tmp/docs'}],
    'llms': [
        {'name': 'main', 'base_url': 'http://x/v1',
         'model': 'qwen3.5:9b', 'api_key': 'k', 'capabilities': ['text', 'vision']},
        {'name': 'cheap', 'base_url': 'http://x/v1',
         'model': 'haiku', 'api_key': 'k'},
    ],
    'indexing': {
        'llm': 'main', 'vision': 'main',
        'dense_embedder': {'model': 'qwen3-embedding:4b',
                           'base_url': 'http://x/v1', 'dim': 2560},
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
        app.state.indexer = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url='http://test') as ac:
            yield ac


class TestGetRetrieval:

    async def test_returns_none_when_not_configured(self, client):
        r = await client.get('/api/retrieval/config')
        assert r.status_code == 200
        data = r.json()
        assert data['retrieval'] is None
        # effective содержит дефолты + первая LLM как placeholder для agent/reranker
        assert data['effective']['agent']['llm'] == 'main'
        assert data['effective']['reranker']['llm'] == 'main'
        assert data['effective']['search']['limit'] == 100        # Pydantic default
        assert data['effective']['search']['find_section']['doc_pool'] == 20
        assert data['effective']['features']['enable_diversity_nudge'] is True
        # llms-pool возвращается всегда — для UI dropdown'ов
        names = [llm['name'] for llm in data['llms']]
        assert names == ['main', 'cheap']

    async def test_returns_existing_section_with_defaults_filled(self, client, workspace):
        local = workspace['tmp'] / 'config.local.yml'
        local.write_text(yaml.safe_dump({
            'retrieval': {
                'agent': {'llm': 'main', 'enable_thinking': False},
                'reranker': {'llm': 'cheap'},
                'search': {'limit': 80},
            },
        }))
        r = await client.get('/api/retrieval/config')
        data = r.json()
        # raw retrieval — как в файле
        assert data['retrieval']['agent']['llm'] == 'main'
        assert data['retrieval']['agent']['enable_thinking'] is False
        # effective — с дефолтами Pydantic
        eff = data['effective']
        assert eff['agent']['llm'] == 'main'
        assert eff['agent']['enable_thinking'] is False
        assert eff['search']['limit'] == 80                       # из конфига
        assert eff['search']['unique_docs_cap'] == 10             # default
        assert eff['search']['find_section']['descent_threshold'] == 0.5  # default


class TestPutRetrieval:

    async def test_writes_minimal_valid(self, client, workspace):
        body = {
            'agent': {'llm': 'main'},
            'reranker': {'llm': 'cheap'},
        }
        r = await client.put('/api/retrieval/config', json=body)
        assert r.status_code == 200
        data = r.json()
        assert data['ok'] is True
        assert data['restart_required'] is False   # hot-reload: рестарт не нужен

        # Проверим что файл записан правильно
        local = workspace['tmp'] / 'config.local.yml'
        assert local.exists()
        loaded = yaml.safe_load(local.read_text())
        assert loaded['retrieval']['agent']['llm'] == 'main'
        assert loaded['retrieval']['reranker']['llm'] == 'cheap'

    async def test_rejects_unknown_llm_reference(self, client):
        body = {
            'agent': {'llm': 'doesnotexist'},
            'reranker': {'llm': 'main'},
        }
        r = await client.put('/api/retrieval/config', json=body)
        assert r.status_code == 400

    async def test_strips_none_fields(self, client, workspace):
        # UI шлёт null'ы для незаполненных полей — они не должны попасть в YAML.
        body = {
            'agent': {'llm': 'main', 'enable_thinking': None,
                      'temperature': None, 'max_tokens': None},
            'reranker': {'llm': 'cheap', 'enable_thinking': None,
                         'max_tokens': None},
            'search': {
                'limit': 60, 'unique_docs_cap': None,
                'find_section': {'doc_pool': None, 'descent_threshold': None,
                                 'top_docs': None},
            },
            'http_timeout': None,
        }
        r = await client.put('/api/retrieval/config', json=body)
        assert r.status_code == 200

        local = workspace['tmp'] / 'config.local.yml'
        loaded = yaml.safe_load(local.read_text())
        # null-поля не записаны
        agent = loaded['retrieval']['agent']
        assert 'enable_thinking' not in agent
        assert 'temperature' not in agent
        # Заполненное — записано
        assert loaded['retrieval']['search']['limit'] == 60
        # Пустой find_section после strip_none — отсутствует
        assert 'find_section' not in loaded['retrieval']['search']

    async def test_section_overrides_preserved(self, client, workspace):
        # WYSIWYG: посекционные оверрайды промпта пишутся как section_overrides.
        body = {
            'agent': {'llm': 'main'},
            'reranker': {'llm': 'main'},
            'prompts': {'section_overrides': {'admin': '\n\n## Хедер\nТестовая инструкция'}},
        }
        r = await client.put('/api/retrieval/config', json=body)
        assert r.status_code == 200

        local = workspace['tmp'] / 'config.local.yml'
        loaded = yaml.safe_load(local.read_text())
        ov = loaded['retrieval']['prompts']['section_overrides']
        assert ov['admin'] == '\n\n## Хедер\nТестовая инструкция'


class TestDiffWrite:
    """Diff-write: значения совпадающие с baseline + Pydantic-defaults
    не записываются в config.local.yml.
    """

    @pytest.fixture
    def workspace_with_baseline(self, tmp_path: Path):
        # Baseline уже содержит часть retrieval — search/features.
        baseline = {
            **PRIMARY_CONFIG,
            'retrieval': {
                'search': {'limit': 50, 'unique_docs_cap': 10},
                'features': {'enable_diversity_nudge': True},
                'http_timeout': 300,
            },
        }
        cfg = tmp_path / 'config.yml'
        cfg.write_text(yaml.safe_dump(baseline))
        old_env = os.environ.get('MORAG_CONFIG_PATH')
        os.environ['MORAG_CONFIG_PATH'] = str(cfg)
        yield {'cfg': cfg, 'tmp': tmp_path}
        if old_env is None:
            os.environ.pop('MORAG_CONFIG_PATH', None)
        else:
            os.environ['MORAG_CONFIG_PATH'] = old_env

    @pytest.fixture
    async def client_baseline(self, workspace_with_baseline):
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.indexer = AsyncMock()
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url='http://test') as ac:
                yield ac

    async def test_only_agent_reranker_written_when_search_matches_baseline(
        self, client_baseline, workspace_with_baseline,
    ):
        body = {
            'agent': {'llm': 'main'},
            'reranker': {'llm': 'cheap'},
            # Эти поля совпадают с baseline → не должны попасть в local
            'search': {'limit': 50, 'unique_docs_cap': 10},
            'features': {'enable_diversity_nudge': True},
            'http_timeout': 300,
        }
        r = await client_baseline.put('/api/retrieval/config', json=body)
        assert r.status_code == 200

        local = workspace_with_baseline['tmp'] / 'config.local.yml'
        loaded = yaml.safe_load(local.read_text())
        retr = loaded['retrieval']
        # Только agent + reranker — остальное совпало с baseline и не записано
        assert retr == {
            'agent': {'llm': 'main'},
            'reranker': {'llm': 'cheap'},
        }

    async def test_only_changed_fields_written(
        self, client_baseline, workspace_with_baseline,
    ):
        body = {
            'agent': {'llm': 'main'},
            'reranker': {'llm': 'main'},
            'search': {'limit': 100, 'unique_docs_cap': 10},  # limit изменён, cap = baseline
            'features': {'enable_diversity_nudge': False},     # изменён
            'http_timeout': 300,                                # baseline
        }
        r = await client_baseline.put('/api/retrieval/config', json=body)
        assert r.status_code == 200

        local = workspace_with_baseline['tmp'] / 'config.local.yml'
        loaded = yaml.safe_load(local.read_text())
        retr = loaded['retrieval']
        # search содержит ТОЛЬКО изменённое поле limit
        assert retr['search'] == {'limit': 100}
        # features — изменённое
        assert retr['features'] == {'enable_diversity_nudge': False}
        # http_timeout совпал — отсутствует
        assert 'http_timeout' not in retr

    async def test_empty_delta_removes_retrieval_key(
        self, client_baseline, workspace_with_baseline,
    ):
        # Сначала запишем что-то ненужное
        local = workspace_with_baseline['tmp'] / 'config.local.yml'
        local.write_text(yaml.safe_dump({
            'retrieval': {'search': {'limit': 999}},
            'qdrant': {'host': 'somehost'},   # не должен пострадать
        }))

        # Теперь PUT со значениями совпадающими с baseline
        body = {
            'search': {'limit': 50},      # = baseline
            'features': {'enable_diversity_nudge': True},
        }
        r = await client_baseline.put('/api/retrieval/config', json=body)
        assert r.status_code == 200

        loaded = yaml.safe_load(local.read_text())
        assert 'retrieval' not in loaded
        # qdrant overlay сохранился
        assert loaded['qdrant']['host'] == 'somehost'
