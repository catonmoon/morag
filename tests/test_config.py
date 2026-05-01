"""Тесты загрузки конфига и overlay из config.local.yml."""
from pathlib import Path

import yaml

from morag.config import _deep_merge, load_config


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:

    def test_overlay_overrides_scalar(self):
        result = _deep_merge({'a': 1}, {'a': 2})
        assert result == {'a': 2}

    def test_overlay_adds_new_key(self):
        result = _deep_merge({'a': 1}, {'b': 2})
        assert result == {'a': 1, 'b': 2}

    def test_nested_dict_recursive_merge(self):
        result = _deep_merge(
            {'llm': {'model': 'old', 'timeout': 60}},
            {'llm': {'model': 'new'}},
        )
        assert result == {'llm': {'model': 'new', 'timeout': 60}}

    def test_lists_replaced_not_extended(self):
        """ancestor_ids: [1,2,3] в overlay ДОЛЖЕН заменять, а не расширять."""
        result = _deep_merge({'ids': [1, 2, 3]}, {'ids': [4, 5]})
        assert result == {'ids': [4, 5]}

    def test_overlay_dict_replaces_scalar(self):
        result = _deep_merge({'x': 1}, {'x': {'nested': True}})
        assert result == {'x': {'nested': True}}

    def test_overlay_scalar_replaces_dict(self):
        result = _deep_merge({'x': {'nested': True}}, {'x': 'string'})
        assert result == {'x': 'string'}

    def test_does_not_mutate_inputs(self):
        base = {'a': {'b': 1}}
        overlay = {'a': {'b': 2}}
        result = _deep_merge(base, overlay)
        assert base == {'a': {'b': 1}}
        assert overlay == {'a': {'b': 2}}
        assert result == {'a': {'b': 2}}

    def test_deep_nesting(self):
        base = {'a': {'b': {'c': {'d': 1, 'e': 2}}}}
        overlay = {'a': {'b': {'c': {'d': 99}}}}
        result = _deep_merge(base, overlay)
        assert result == {'a': {'b': {'c': {'d': 99, 'e': 2}}}}


# ---------------------------------------------------------------------------
# load_config с overlay
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    'sources': {'local_documents': {'path': '/tmp/docs'}},
    'llm': {
        'base_url': 'http://primary.example/v1',
        'model': 'primary-model',
        'api_key': 'primary-key',
    },
    'indexing': {
        'dense_embedder': {
            'model': 'qwen3-embedding:4b',
            'base_url': 'http://primary.example/v1',
            'dim': 2560,
        },
    },
}


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True))


class TestLoadConfigOverlay:

    def test_loads_without_overlay(self, tmp_path: Path):
        cfg_path = tmp_path / 'config.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)

        cfg = load_config(cfg_path)
        assert cfg.llm.base_url == 'http://primary.example/v1'
        assert cfg.llm.model == 'primary-model'

    def test_overlay_overrides_llm_model(self, tmp_path: Path):
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        write_yaml(local_path, {
            'llm': {
                'model': 'overridden-model',
                'api_key': 'secret-from-local',
            },
        })

        cfg = load_config(cfg_path)
        assert cfg.llm.model == 'overridden-model'
        assert cfg.llm.api_key == 'secret-from-local'
        # base_url не упомянут в overlay → остаётся primary
        assert cfg.llm.base_url == 'http://primary.example/v1'

    def test_overlay_can_replace_lists(self, tmp_path: Path):
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'

        primary = dict(MINIMAL_CONFIG)
        primary['sources'] = {
            'local_documents': {'path': '/tmp/docs'},
            'confluence': {
                'url': 'https://x',
                'username': 'u',
                'ancestor_ids': ['1', '2', '3'],
            },
        }
        write_yaml(cfg_path, primary)
        write_yaml(local_path, {
            'sources': {'confluence': {'ancestor_ids': ['99']}},
        })

        cfg = load_config(cfg_path)
        assert cfg.sources.confluence.ancestor_ids == ['99']
        # username не тронут
        assert cfg.sources.confluence.username == 'u'

    def test_missing_overlay_is_fine(self, tmp_path: Path):
        """Отсутствие config.local.yml — нормальная ситуация, не падать."""
        cfg_path = tmp_path / 'config.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)

        cfg = load_config(cfg_path)
        assert cfg.llm.api_key == 'primary-key'

    def test_empty_overlay_yaml_does_nothing(self, tmp_path: Path):
        """Пустой config.local.yml (yaml.safe_load → None) не должен ломать загрузку."""
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        local_path.write_text('')

        cfg = load_config(cfg_path)
        assert cfg.llm.api_key == 'primary-key'

    def test_overlay_can_add_optional_section(self, tmp_path: Path):
        """Overlay добавляет llm_vision которого не было в основном конфиге."""
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        write_yaml(local_path, {
            'llm_vision': {
                'base_url': 'http://vision.example/v1',
                'model': 'qwen2.5-vl',
                'api_key': 'vision-key',
            },
        })

        cfg = load_config(cfg_path)
        assert cfg.llm_vision is not None
        assert cfg.llm_vision.model == 'qwen2.5-vl'
