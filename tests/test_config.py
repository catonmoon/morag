"""Тесты загрузки конфига и overlay из config.local.yml."""
from pathlib import Path

import pytest
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
    'sources': [
        {'kind': 'local', 'name': 'docs', 'path': '/tmp/docs'},
    ],
    'llms': [
        {
            'name': 'main',
            'base_url': 'http://primary.example/v1',
            'model': 'primary-model',
            'api_key': 'primary-key',
        },
        {
            'name': 'vision',
            'base_url': 'http://primary.example/v1',
            'model': 'vision-model',
            'api_key': 'primary-key',
            'capabilities': ['text', 'vision'],
        },
    ],
    'indexing': {
        'llm': 'main',
        'vision': 'vision',
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
        assert cfg.llm_by_name('main').base_url == 'http://primary.example/v1'
        assert cfg.llm_by_name('main').model == 'primary-model'

    def test_overlay_overrides_qdrant_host(self, tmp_path: Path):
        """Типичный use-case — переопределить host для local development."""
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        write_yaml(local_path, {
            'qdrant': {'host': 'localhost'},
        })

        cfg = load_config(cfg_path)
        assert cfg.qdrant.host == 'localhost'
        # port не упомянут — остался default
        assert cfg.qdrant.port == 6333

    def test_overlay_can_replace_lists(self, tmp_path: Path):
        """Списки в overlay перезаписываются целиком (не extend)."""
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        # overlay полностью заменяет llms (например, юзер хочет другую LLM)
        write_yaml(local_path, {
            'llms': [
                {'name': 'main', 'base_url': 'http://new/v1', 'model': 'new-m', 'api_key': 'k'},
                {'name': 'vision', 'base_url': 'http://new/v1', 'model': 'new-v', 'api_key': 'k',
                 'capabilities': ['text', 'vision']},
            ],
        })

        cfg = load_config(cfg_path)
        assert cfg.llm_by_name('main').base_url == 'http://new/v1'

    def test_missing_overlay_is_fine(self, tmp_path: Path):
        cfg_path = tmp_path / 'config.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)

        cfg = load_config(cfg_path)
        assert cfg.llm_by_name('main').api_key == 'primary-key'

    def test_empty_overlay_yaml_does_nothing(self, tmp_path: Path):
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        local_path.write_text('')

        cfg = load_config(cfg_path)
        assert cfg.llm_by_name('main').api_key == 'primary-key'

    def test_overlay_can_add_pdf_section(self, tmp_path: Path):
        """Overlay добавляет pdf-секцию которой не было в primary."""
        cfg_path = tmp_path / 'config.yml'
        local_path = tmp_path / 'config.local.yml'
        write_yaml(cfg_path, MINIMAL_CONFIG)
        write_yaml(local_path, {
            'pdf': {'mode': 'vision', 'dpi': 200},
        })

        cfg = load_config(cfg_path)
        assert cfg.pdf is not None
        assert cfg.pdf.dpi == 200


# ---------------------------------------------------------------------------
# New schema: Source discriminated union, LLM pool, role mapping
# ---------------------------------------------------------------------------

class TestSourcesDiscriminatedUnion:
    """Sources как list[Source] с дискриминатором kind."""

    def test_local_source(self):
        from morag.config import Config, LocalSourceConfig
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'docs', 'path': 'data/'}],
            'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert isinstance(cfg.sources[0], LocalSourceConfig)
        assert cfg.sources[0].name == 'docs'
        assert cfg.sources[0].kind == 'local'
        assert cfg.sources[0].enabled is True

    def test_multiple_confluence_instances(self):
        from morag.config import Config, ConfluenceSourceConfig
        cfg = Config.model_validate({
            'sources': [
                {'kind': 'confluence', 'name': 'corp', 'url': 'https://corp/',
                 'username': 'u', 'api_token': 't'},
                {'kind': 'confluence', 'name': 'vendor', 'url': 'https://vendor/',
                 'username': 'u', 'api_token': 't'},
            ],
            'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        confs = [s for s in cfg.sources if isinstance(s, ConfluenceSourceConfig)]
        assert len(confs) == 2
        assert {s.name for s in confs} == {'corp', 'vendor'}

    def test_multiple_jira_instances(self):
        from morag.config import Config, JiraSourceConfig
        cfg = Config.model_validate({
            'sources': [
                {'kind': 'jira', 'name': 'internal', 'url': 'https://j1/',
                 'username': 'u', 'password': 'p'},
                {'kind': 'jira', 'name': 'vendor', 'url': 'https://j2/',
                 'username': 'u', 'password': 'p'},
            ],
            'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        jiras = [s for s in cfg.sources if isinstance(s, JiraSourceConfig)]
        assert len(jiras) == 2

    def test_jira_requires_password_no_api_token(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as ctx:
            Config.model_validate({
                'sources': [
                    {'kind': 'jira', 'name': 'j', 'url': 'x', 'username': 'u',
                     'api_token': 't'},  # api_token не существует в JiraSourceConfig
                ],
                'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
            })
        # api_token не определён в JiraSourceConfig (только on-prem) → либо ignored либо ошибка
        # (зависит от Pydantic strict mode; здесь password обязателен)
        assert 'password' in str(ctx.value).lower() or 'api_token' in str(ctx.value).lower()

    def test_confluence_requires_secret(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='password.*api_token'):
            Config.model_validate({
                'sources': [
                    {'kind': 'confluence', 'name': 'c', 'url': 'x', 'username': 'u'},
                    # ни password, ни api_token
                ],
                'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
            })

    def test_duplicate_source_kind_name_rejected(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='Duplicate source'):
            Config.model_validate({
                'sources': [
                    {'kind': 'local', 'name': 'docs', 'path': 'a'},
                    {'kind': 'local', 'name': 'docs', 'path': 'b'},
                ],
                'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
            })

    def test_same_name_different_kind_ok(self):
        """name='main' разрешено для одновременно local и confluence."""
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [
                {'kind': 'local', 'name': 'main', 'path': 'a'},
                {'kind': 'confluence', 'name': 'main', 'url': 'x', 'username': 'u',
                 'api_token': 't'},
            ],
            'llms': [{'name': 'l', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert len(cfg.sources) == 2

    def test_source_name_lowercase_validation(self):
        """name должен быть lowercase, без пробелов и спецсимволов."""
        from morag.config import Config
        from pydantic import ValidationError
        for bad_name in ['Docs', 'docs main', 'docs/', 'docs!']:
            with pytest.raises(ValidationError):
                Config.model_validate({
                    'sources': [{'kind': 'local', 'name': bad_name, 'path': 'a'}],
                    'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
                })

    def test_disabled_source(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [
                {'kind': 'local', 'name': 'docs', 'path': 'a', 'enabled': False},
                {'kind': 'local', 'name': 'docs2', 'path': 'b'},  # default enabled
            ],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert cfg.sources[0].enabled is False
        assert cfg.sources[1].enabled is True
        # sources_by_kind отдаёт только enabled
        enabled_locals = cfg.sources_by_kind('local')
        assert [s.name for s in enabled_locals] == ['docs2']

    def test_min_one_source_required(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Config.model_validate({
                'sources': [],
                'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
            })


class TestLLMPool:
    """LLMs как именованный пул."""

    def test_multiple_llms(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [
                {'name': 'main', 'base_url': 'x', 'model': 'm1', 'api_key': 'k'},
                {'name': 'smart', 'base_url': 'y', 'model': 'm2', 'api_key': 'k'},
                {'name': 'cheap', 'base_url': 'z', 'model': 'm3', 'api_key': 'k'},
            ],
        })
        assert {llm.name for llm in cfg.llms} == {'main', 'smart', 'cheap'}
        assert cfg.llm_by_name('smart').model == 'm2'

    def test_duplicate_llm_name_rejected(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='Duplicate llm name'):
            Config.model_validate({
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [
                    {'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'},
                    {'name': 'main', 'base_url': 'y', 'model': 'm2', 'api_key': 'k'},
                ],
            })

    def test_min_one_llm_required(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Config.model_validate({
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [],
            })


class TestLLMRoleMapping:
    """Маппинг роли → имя LLM, поддержка short и full форм."""

    def test_short_form_string(self):
        from morag.config import LLMRoleMapping
        m = LLMRoleMapping.model_validate('main')
        assert m.default == 'main'
        assert m.overrides == {}
        assert m.name_for('any_role') == 'main'

    def test_full_form_dict(self):
        from morag.config import LLMRoleMapping
        m = LLMRoleMapping.model_validate({
            'default': 'main',
            'overrides': {'doc_summary': 'smart', 'knowledge_map': 'smart'},
        })
        assert m.default == 'main'
        assert m.name_for('doc_summary') == 'smart'
        assert m.name_for('knowledge_map') == 'smart'
        assert m.name_for('context_generation') == 'main'  # fallback на default

    def test_indexing_llm_short_form(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [
                {'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'},
                {'name': 'vision', 'base_url': 'x', 'model': 'v', 'api_key': 'k',
                 'capabilities': ['text', 'vision']},
            ],
            'indexing': {
                'llm': 'main',  # короткая форма
                'vision': 'vision',
                'dense_embedder': {'model': 'm'},
            },
        })
        assert cfg.indexing.llm.default == 'main'
        assert cfg.indexing.llm.overrides == {}

    def test_unknown_llm_reference_rejected(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='not found in llms pool'):
            Config.model_validate({
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
                'indexing': {
                    'llm': 'nonexistent',
                    'vision': 'main',
                    'dense_embedder': {'model': 'm'},
                },
            })

    def test_unknown_override_reference_rejected(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='not found in llms pool'):
            Config.model_validate({
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
                'indexing': {
                    'llm': {'default': 'main', 'overrides': {'doc_summary': 'ghost'}},
                    'vision': 'main',
                    'dense_embedder': {'model': 'm'},
                },
            })


class TestLLMCapabilities:
    """LLMInstance.capabilities + Config-level валидация что vision-роль
    указывает на LLM с capability 'vision'."""

    def test_default_capabilities_is_text_only(self):
        from morag.config import LLMInstance
        llm = LLMInstance(name='m', base_url='x', model='m', api_key='k')
        assert llm.capabilities == ['text']

    def test_explicit_multimodal(self):
        from morag.config import LLMInstance
        llm = LLMInstance(
            name='m', base_url='x', model='m', api_key='k',
            capabilities=['text', 'vision'],
        )
        assert 'vision' in llm.capabilities

    def test_empty_capabilities_rejected(self):
        from morag.config import LLMInstance
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMInstance(
                name='m', base_url='x', model='m', api_key='k',
                capabilities=[],
            )

    def test_unknown_capability_rejected(self):
        from morag.config import LLMInstance
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMInstance(
                name='m', base_url='x', model='m', api_key='k',
                capabilities=['text', 'audio'],  # 'audio' не литерал
            )

    def test_indexing_vision_must_have_vision_capability(self):
        from morag.config import Config
        from pydantic import ValidationError
        # main — text-only (default), указан как vision-роль → должен упасть
        with pytest.raises(ValidationError, match="не объявляет capability 'vision'"):
            Config.model_validate({
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [
                    {'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'},
                ],
                'indexing': {
                    'llm': 'main',
                    'vision': 'main',  # ← не имеет capability vision
                    'dense_embedder': {'model': 'm'},
                },
            })

    def test_multimodal_llm_can_serve_both_roles(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [
                {'name': 'qwen', 'base_url': 'x', 'model': 'qwen-vl', 'api_key': 'k',
                 'capabilities': ['text', 'vision']},
            ],
            'indexing': {
                'llm': 'qwen',     # text-role
                'vision': 'qwen',  # vision-role — тот же LLM
                'dense_embedder': {'model': 'm'},
            },
        })
        # один LLM в пуле, обе роли указывают на него
        assert cfg.indexing.llm.default == 'qwen'
        assert cfg.indexing.vision == 'qwen'

    def test_dedicated_vision_llm(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [
                {'name': 'grok', 'base_url': 'x', 'model': 'grok-4', 'api_key': 'k'},
                {'name': 'qwen-vl', 'base_url': 'y', 'model': 'qwen2.5-vl', 'api_key': 'k',
                 'capabilities': ['vision']},  # only vision
            ],
            'indexing': {
                'llm': 'grok',
                'vision': 'qwen-vl',
                'dense_embedder': {'model': 'm'},
            },
        })
        assert cfg.indexing.vision == 'qwen-vl'


class TestSchemaVersion:

    def test_schema_version_defaults_to_1(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert cfg.schema_version == 1

    def test_explicit_schema_version_1(self):
        from morag.config import Config
        cfg = Config.model_validate({
            'schema_version': 1,
            'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
            'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
        })
        assert cfg.schema_version == 1

    def test_unknown_schema_version_rejected(self):
        from morag.config import Config
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Config.model_validate({
                'schema_version': 2,
                'sources': [{'kind': 'local', 'name': 'd', 'path': 'a'}],
                'llms': [{'name': 'm', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
            })
