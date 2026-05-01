"""Тесты для services/console/presets.py."""
import pytest

from services.console.presets import (
    DENSE_EMBEDDER_PRESETS,
    LLM_PRESETS,
    apply_preset,
    find_preset,
    serialize_preset,
)


class TestFindPreset:
    def test_finds_existing(self):
        p = find_preset('llm', 'grok')
        assert p.id == 'grok'
        assert p.target == 'llm'

    def test_raises_for_unknown(self):
        with pytest.raises(KeyError):
            find_preset('llm', 'nonexistent')


class TestApplyPresetLLM:

    def test_grok(self):
        snippet = apply_preset('llm', 'grok', {'api_key': 'xai-test', 'model': 'grok-4-1-fast'})
        assert snippet == {
            'llm': {
                'base_url': 'https://api.x.ai/v1',
                'model': 'grok-4-1-fast',
                'api_key': 'xai-test',
                'context_window': 256000,
                'max_concurrent': 8,
            },
        }

    def test_grok_default_model(self):
        snippet = apply_preset('llm', 'grok', {'api_key': 'xai-test'})
        assert snippet['llm']['model'] == 'grok-4-1-fast-non-reasoning'

    def test_openrouter(self):
        snippet = apply_preset('llm', 'openrouter', {
            'model': 'anthropic/claude-haiku',
            'api_key': 'sk-or-test',
        })
        assert snippet['llm']['base_url'] == 'https://openrouter.ai/api/v1'
        assert snippet['llm']['model'] == 'anthropic/claude-haiku'

    def test_ollama_uses_dummy_api_key(self):
        """Ollama не требует ключа, но OpenAI-compat SDK требует непустое значение."""
        snippet = apply_preset('llm', 'ollama', {'model': 'qwen3:4b'})
        assert snippet['llm']['api_key'] == 'ollama'
        assert snippet['llm']['enable_thinking'] is False

    def test_custom_full(self):
        snippet = apply_preset('llm', 'custom', {
            'base_url': 'http://my.vllm/v1',
            'model': 'my-model',
            'api_key': 'k',
            'context_window': '16384',  # строка из формы — должна сконвертиться
            'max_concurrent': '2',
        })
        assert snippet['llm']['context_window'] == 16384
        assert snippet['llm']['max_concurrent'] == 2

    def test_custom_optional_max_concurrent(self):
        snippet = apply_preset('llm', 'custom', {
            'base_url': 'http://x', 'model': 'm', 'api_key': 'k',
        })
        assert 'max_concurrent' not in snippet['llm']


class TestApplyPresetEmbedder:

    def test_ollama_qwen3_defaults(self):
        snippet = apply_preset('dense_embedder', 'ollama-qwen3', {})
        embed = snippet['indexing']['dense_embedder']
        assert embed['model'] == 'qwen3-embedding:4b'
        assert embed['dim'] == 2560
        assert embed['tokenizer'] == 'tiktoken'
        assert 'Instruct:' in embed['query_template']

    def test_custom_minimal(self):
        snippet = apply_preset('dense_embedder', 'custom', {
            'base_url': 'http://x/v1',
            'model': 'm',
            'dim': '768',
        })
        embed = snippet['indexing']['dense_embedder']
        assert embed['dim'] == 768
        assert embed['tokenizer'] == 'tiktoken'


class TestSerializePreset:
    def test_serializable(self):
        for p in LLM_PRESETS + DENSE_EMBEDDER_PRESETS:
            data = serialize_preset(p)
            assert 'id' in data
            assert 'fields' in data
            assert isinstance(data['fields'], list)
            for f in data['fields']:
                assert 'name' in f and 'label' in f


class TestPresetsValid:
    """Sanity-check: все пресеты собираются в Pydantic-валидный snippet (где возможно)."""

    def test_grok_snippet_validates(self):
        from morag.config import LLMConfig
        snippet = apply_preset('llm', 'grok', {'api_key': 'xai-test'})
        cfg = LLMConfig(**snippet['llm'])
        assert cfg.base_url.startswith('https://')

    def test_ollama_embedder_validates(self):
        from morag.config import DenseEmbedderConfig
        snippet = apply_preset('dense_embedder', 'ollama-qwen3', {})
        cfg = DenseEmbedderConfig(**snippet['indexing']['dense_embedder'])
        assert cfg.dim == 2560
