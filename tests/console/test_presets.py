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
        p = find_preset('llm', 'grok')
        assert p.id == 'grok'
        assert p.target == 'llm'

    def test_finds_existing_source(self):
        p = find_preset('source', 'local')
        assert p.id == 'local'
        assert p.target == 'source'

    def test_raises_for_unknown(self):
        with pytest.raises(KeyError):
            find_preset('llm', 'nonexistent')


class TestApplyLLMPresets:
    def test_grok_minimal(self):
        snippet = apply_preset('llm', 'grok', {
            'name': 'mygrok', 'api_key': 'xai-test',
        })
        assert snippet['name'] == 'mygrok'
        assert snippet['base_url'] == 'https://api.x.ai/v1'
        assert snippet['model'] == 'grok-4-1-fast-non-reasoning'
        assert snippet['api_key'] == 'xai-test'
        assert snippet['capabilities'] == ['text']

    def test_vision_capable_flag(self):
        snippet = apply_preset('llm', 'grok', {
            'name': 'g', 'api_key': 'k', 'vision_capable': True,
        })
        assert snippet['capabilities'] == ['text', 'vision']

    def test_vision_capable_string_form(self):
        # Из HTML checkbox приходит 'on' или 'true'
        snippet = apply_preset('llm', 'grok', {
            'name': 'g', 'api_key': 'k', 'vision_capable': 'on',
        })
        assert 'vision' in snippet['capabilities']

    def test_default_name_when_empty(self):
        snippet = apply_preset('llm', 'grok', {'name': '', 'api_key': 'k'})
        assert snippet['name'] == 'grok'

    def test_ollama(self):
        snippet = apply_preset('llm', 'ollama', {
            'name': 'local-qwen', 'model': 'qwen3:4b',
        })
        assert snippet['api_key'] == 'ollama'  # hardcoded для Ollama
        assert snippet['enable_thinking'] is False

    def test_openrouter(self):
        snippet = apply_preset('llm', 'openrouter', {
            'name': 'or', 'model': 'anthropic/claude', 'api_key': 'sk-or-test',
        })
        assert snippet['base_url'] == 'https://openrouter.ai/api/v1'

    def test_custom_full(self):
        snippet = apply_preset('llm', 'custom', {
            'name': 'my-vllm', 'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
            'context_window': '16384',
        })
        assert snippet['context_window'] == 16384


class TestApplySourcePresets:
    def test_local(self):
        s = apply_preset('source', 'local', {'name': 'docs', 'path': 'data/'})
        assert s == {'kind': 'local', 'name': 'docs', 'path': 'data/'}

    def test_local_default_name(self):
        s = apply_preset('source', 'local', {'name': '', 'path': 'data/'})
        assert s['name'] == 'docs'

    def test_confluence_cloud_minimal(self):
        s = apply_preset('source', 'confluence-cloud', {
            'name': 'corp', 'url': 'https://corp/', 'username': 'u', 'api_token': 't',
        })
        assert s['kind'] == 'confluence'
        assert s['api_token'] == 't'
        assert 'password' not in s

    def test_confluence_cloud_with_spaces(self):
        s = apply_preset('source', 'confluence-cloud', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'spaces': 'DOCS, ENG, ML',
        })
        assert s['spaces'] == ['DOCS', 'ENG', 'ML']

    def test_confluence_cloud_with_attachments(self):
        s = apply_preset('source', 'confluence-cloud', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'attachments_enabled': True,
        })
        assert s['attachments'] == {'enabled': True}

    def test_confluence_onprem(self):
        s = apply_preset('source', 'confluence-onprem', {
            'name': 'c', 'url': 'x', 'username': 'u', 'password': 'p',
        })
        assert s['password'] == 'p'
        assert 'api_token' not in s

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

    def test_grok_llm_validates(self):
        from morag.config import Config
        snippet = apply_preset('llm', 'grok', {'name': 'main', 'api_key': 'xai-test'})
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
