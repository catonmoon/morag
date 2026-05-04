"""Тесты services/console/presets.py — LLM + source presets под new schema."""
import pytest

from services.console.presets import (
    LLM_PRESETS,
    SOURCE_PRESETS,
    apply_preset,
    find_preset,
    serialize_preset,
)


class TestFindPreset:
    def test_finds_existing_llm(self):
        p = find_preset('llm', 'openai-compatible')
        assert p.id == 'openai-compatible'
        assert p.target == 'llm'

    def test_finds_existing_source(self):
        p = find_preset('source', 'local')
        assert p.id == 'local'
        assert p.target == 'source'

    def test_raises_for_unknown(self):
        with pytest.raises(KeyError):
            find_preset('llm', 'nonexistent')


class TestApplyLLMPresets:
    def test_openai_minimal(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'mygrok',
            'base_url': 'https://api.x.ai/v1',
            'model': 'grok-4-1-fast-non-reasoning',
            'api_key': 'xai-test',
        })
        assert snippet['name'] == 'mygrok'
        assert snippet['base_url'] == 'https://api.x.ai/v1'
        assert snippet['model'] == 'grok-4-1-fast-non-reasoning'
        assert snippet['api_key'] == 'xai-test'
        assert snippet['capabilities'] == ['text']
        # max_concurrent всегда отдаём — даже если юзер не задал
        assert snippet['max_concurrent'] == 4

    def test_vision_capable_flag(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'g', 'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
            'vision_capable': True,
        })
        assert snippet['capabilities'] == ['text', 'vision']

    def test_vision_capable_string_form(self):
        # Из HTML checkbox приходит 'on' или 'true'
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'g', 'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
            'vision_capable': 'on',
        })
        assert 'vision' in snippet['capabilities']

    def test_default_name_when_empty(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': '', 'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
        })
        assert snippet['name'] == 'main'  # default name для openai-пресета

    def test_openai_custom_concurrent(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'or', 'base_url': 'https://openrouter.ai/api/v1',
            'model': 'anthropic/claude', 'api_key': 'sk-or',
            'max_concurrent': '8',
        })
        assert snippet['max_concurrent'] == 8

    def test_ollama(self):
        snippet = apply_preset('llm', 'ollama', {
            'name': 'local-qwen', 'model': 'qwen3:4b',
        })
        assert snippet['api_key'] == 'ollama'           # hardcoded для Ollama
        assert snippet['enable_thinking'] is False
        assert snippet['max_concurrent'] == 1           # дефолт для Ollama
        assert snippet['base_url'].endswith(':11434/v1')


class TestApplySourcePresets:
    def test_local(self):
        s = apply_preset('source', 'local', {'name': 'docs', 'path': 'data/'})
        assert s == {'kind': 'local', 'name': 'docs', 'path': 'data/'}

    def test_local_default_name(self):
        s = apply_preset('source', 'local', {'name': '', 'path': 'data/'})
        assert s['name'] == 'docs'

    def test_confluence_cloud_minimal(self):
        s = apply_preset('source', 'confluence', {
            'name': 'corp', 'url': 'https://corp/', 'username': 'u', 'api_token': 't',
        })
        assert s['kind'] == 'confluence'
        assert s['api_token'] == 't'
        assert 'password' not in s

    def test_confluence_with_spaces(self):
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'spaces': 'DOCS, ENG, ML',
        })
        assert s['spaces'] == ['DOCS', 'ENG', 'ML']

    def test_confluence_with_attachments(self):
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'attachments_enabled': True,
        })
        assert s['attachments'] == {'enabled': True}

    def test_confluence_onprem(self):
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'password': 'p',
        })
        assert s['password'] == 'p'
        assert 'api_token' not in s

    def test_confluence_api_token_wins_over_password(self):
        # Если юзер по ошибке заполнил оба поля — приоритет api_token (Cloud-режим)
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u',
            'api_token': 't', 'password': 'p',
        })
        assert s['api_token'] == 't'
        assert 'password' not in s

    def test_jira(self):
        s = apply_preset('source', 'jira', {
            'name': 'internal', 'url': 'https://j', 'username': 'u', 'password': 'p',
        })
        assert s == {
            'kind': 'jira', 'name': 'internal',
            'url': 'https://j', 'username': 'u', 'password': 'p',
        }


class TestSerializePreset:
    def test_all_serializable(self):
        for p in LLM_PRESETS + SOURCE_PRESETS:
            data = serialize_preset(p)
            assert 'id' in data
            assert 'fields' in data
            assert isinstance(data['fields'], list)
            for f in data['fields']:
                assert 'name' in f and 'label' in f


class TestPresetsValidateUnderNewSchema:
    """Snippets выданные пресетами должны добавляться в Pydantic Config без ошибок."""

    def test_openai_llm_validates(self):
        from morag.config import Config
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'main',
            'base_url': 'https://api.x.ai/v1',
            'model': 'grok-4-1-fast-non-reasoning',
            'api_key': 'xai-test',
        })
        # Минимальный конфиг с этим LLM должен пройти валидацию
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [snippet, {'name': 'v', 'base_url': 'x', 'model': 'm', 'api_key': 'k',
                               'capabilities': ['text', 'vision']}],
            'indexing': {
                'llm': 'main', 'vision': 'v',
                'dense_embedder': {'model': 'm'},
            },
        })
        assert cfg.llm_by_name('main').base_url == 'https://api.x.ai/v1'

    def test_local_source_validates(self):
        from morag.config import Config
        snippet = apply_preset('source', 'local', {'name': 'mydocs', 'path': 'data/'})
        cfg = Config.model_validate({
            'sources': [snippet],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert cfg.sources[0].name == 'mydocs'


class TestApplyEmbedderPresets:
    def test_ollama_minimal(self):
        snippet = apply_preset('embedder', 'ollama', {
            'model': 'qwen3-embedding:4b',
            'dim': '2560',
        })
        assert snippet['model'] == 'qwen3-embedding:4b'
        assert snippet['api_key'] == 'ollama'
        assert snippet['dim'] == 2560
        assert snippet['max_concurrent'] == 1
        assert snippet['base_url'].endswith(':11434/v1')

    def test_openai_compatible(self):
        snippet = apply_preset('embedder', 'openai-compatible', {
            'base_url': 'https://api.openai.com/v1',
            'model': 'text-embedding-3-small',
            'api_key': 'sk-test',
            'dim': '1536',
        })
        assert snippet == {
            'base_url': 'https://api.openai.com/v1',
            'model': 'text-embedding-3-small',
            'api_key': 'sk-test',
            'dim': 1536,
            'max_concurrent': 4,
        }

    def test_validates_under_schema(self):
        from morag.config import Config
        snippet = apply_preset('embedder', 'ollama', {
            'model': 'qwen3-embedding:4b', 'dim': '2560',
        })
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': '/x'}],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k',
                      'capabilities': ['text', 'vision']}],
            'indexing': {
                'llm': 'm', 'vision': 'm',
                'dense_embedder': snippet,
            },
        })
        assert cfg.indexing.dense_embedder.dim == 2560
        assert cfg.indexing.dense_embedder.api_key == 'ollama'
