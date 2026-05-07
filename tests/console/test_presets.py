"""Тесты services/console/presets.py — LLM + source presets под new schema."""
import pytest

from services.console.presets import (
    LLM_PRESETS,
    SOURCE_PRESETS,
    apply_preset,
    find_preset,
    serialize_preset,
    suggest_llm_name,
    suggest_source_name,
    unique_name,
)


class TestNameAutoGeneration:

    def test_local_singleton(self):
        assert suggest_source_name('local', {}) == 'doc'

    def test_confluence_subdomain(self):
        assert suggest_source_name('confluence', {'url': 'https://corp.atlassian.net'}) == 'corp'

    def test_confluence_skips_generic(self):
        assert suggest_source_name(
            'confluence', {'url': 'https://confluence.acme.com'},
        ) == 'acme'

    def test_jira_subdomain(self):
        assert suggest_source_name('jira', {'url': 'https://jira.bigco.com'}) == 'bigco'

    def test_no_url_falls_back_to_kind(self):
        assert suggest_source_name('confluence', {}) == 'confluence'

    def test_llm_known_provider(self):
        assert suggest_llm_name({'base_url': 'https://api.x.ai/v1'}) == 'grok'
        assert suggest_llm_name({'base_url': 'https://openrouter.ai/api/v1'}) == 'openrouter'
        assert suggest_llm_name({'base_url': 'https://api.openai.com/v1'}) == 'openai'

    def test_llm_ollama_localhost(self):
        assert suggest_llm_name(
            {'base_url': 'http://host.docker.internal:11434/v1'},
        ) == 'ollama'

    def test_llm_unknown_provider_subdomain(self):
        # Кастомный vLLM: vllm.kth.pro → 'kth' (skip 'vllm' generic? нет, vllm НЕ generic)
        # Тут должен взять первый non-generic — vllm → берётся, т.к. не в _GENERIC_HOST_PARTS
        # Но _domainFirstLabel пропускает только {www, api, confluence, jira, wiki, docs}
        assert suggest_llm_name({'base_url': 'https://vllm.kth.pro'}) == 'vllm'

    def test_unique_name_no_collision(self):
        assert unique_name('grok', set()) == 'grok'
        assert unique_name('grok', {'other'}) == 'grok'

    def test_unique_name_with_collision(self):
        assert unique_name('grok', {'grok'}) == 'grok-2'
        assert unique_name('grok', {'grok', 'grok-2'}) == 'grok-3'

    def test_unique_name_empty_base(self):
        assert unique_name('', set()) == 'main'


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

    def test_name_auto_generated_from_url(self):
        # Auto-suggest имя по subdomain URL (или провайдеру). 'http://x' → 'x'.
        snippet = apply_preset('llm', 'openai-compatible', {
            'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
        })
        assert snippet['name'] == 'x'

    def test_name_grok_from_xai(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'base_url': 'https://api.x.ai/v1', 'model': 'grok-4',
        })
        assert snippet['name'] == 'grok'

    def test_name_openrouter(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'base_url': 'https://openrouter.ai/api/v1', 'model': 'anthropic/claude',
        })
        assert snippet['name'] == 'openrouter'

    def test_name_ollama(self):
        snippet = apply_preset('llm', 'ollama', {'model': 'qwen3:4b'})
        assert snippet['name'] == 'ollama'

    def test_openai_custom_concurrent(self):
        snippet = apply_preset('llm', 'openai-compatible', {
            'name': 'or', 'base_url': 'https://openrouter.ai/api/v1',
            'model': 'anthropic/claude', 'api_key': 'sk-or',
            'max_concurrent': '8',
        })
        assert snippet['max_concurrent'] == 8

    def test_ollama(self):
        snippet = apply_preset('llm', 'ollama', {'model': 'qwen3:4b'})
        assert snippet['api_key'] == 'ollama'           # hardcoded для Ollama
        assert snippet['enable_thinking'] is False
        assert snippet['max_concurrent'] == 1           # дефолт для Ollama
        assert snippet['base_url'].endswith(':11434/v1')


class TestApplySourcePresets:
    def test_local(self):
        # singleton: name всегда 'doc', path всегда /app/data, форма пустая
        s = apply_preset('source', 'local', {})
        assert s == {'kind': 'local', 'name': 'doc', 'path': '/app/data'}

    def test_local_ignores_form_fields(self):
        # любые попытки задать name через форму игнорируются
        s = apply_preset('source', 'local', {'name': 'something-else'})
        assert s['name'] == 'doc'

    def test_confluence_cloud_minimal(self):
        s = apply_preset('source', 'confluence', {
            'url': 'https://corp.atlassian.net/', 'username': 'u', 'api_token': 't',
        })
        assert s['kind'] == 'confluence'
        assert s['name'] == 'corp'              # auto: subdomain → name
        assert s['api_token'] == 't'
        assert 'password' not in s

    def test_confluence_name_skips_generic_prefix(self):
        # 'confluence.acme.com' → пропускаем generic 'confluence', берём 'acme'
        s = apply_preset('source', 'confluence', {
            'url': 'https://confluence.acme.com', 'username': 'u', 'password': 'p',
        })
        assert s['name'] == 'acme'

    def test_jira_name_from_subdomain(self):
        s = apply_preset('source', 'jira', {
            'url': 'https://jira.internal.bigco.com', 'username': 'u', 'password': 'p',
        })
        assert s['name'] == 'internal'         # skip 'jira', take next non-generic

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

    def test_confluence_ancestor_chips(self):
        # UI-формат: list[{id, comment}]. Должен попасть в YAML с inline-комментами.
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'ancestor_ids': [
                {'id': '1234', 'comment': 'DOCS / Архитектура'},
                {'id': '5678', 'comment': 'DOCS / Требования'},
            ],
            'skip_ancestor_ids': [
                {'id': '9999', 'comment': 'DOCS / Архив'},
            ],
        })
        # IDs as strings
        assert list(s['ancestor_ids']) == ['1234', '5678']
        assert list(s['skip_ancestor_ids']) == ['9999']
        # round-trip через ruamel сохраняет комменты — проверим в test_config_io

    def test_confluence_ancestor_legacy_csv(self):
        # Старый формат (CSV-строка) тоже работает — для ручной правки
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'ancestor_ids': '1234, 5678 9999',
        })
        assert list(s['ancestor_ids']) == ['1234', '5678', '9999']

    def test_confluence_no_ancestors_omitted(self):
        # Пустой ancestor_ids → ключ вообще не пишется
        s = apply_preset('source', 'confluence', {
            'name': 'c', 'url': 'x', 'username': 'u', 'api_token': 't',
            'ancestor_ids': [],
        })
        assert 'ancestor_ids' not in s
        assert 'skip_ancestor_ids' not in s

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
        snippet = apply_preset('source', 'local', {})
        cfg = Config.model_validate({
            'sources': [snippet],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert cfg.sources[0].name == 'doc'      # singleton hardcoded


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
