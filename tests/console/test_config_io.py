"""Тесты для services/console/config_io.py."""
from pathlib import Path

import yaml

from services.console.config_io import (
    SECRET_MASK,
    make_commented_id_list,
    mask_secrets,
    patch_local,
    read_layered,
    read_local,
    strip_masked_secrets,
    validate_merged,
    write_local,
)


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True))


PRIMARY = {
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


# ---------------------------------------------------------------------------
# mask_secrets
# ---------------------------------------------------------------------------

class TestMaskSecrets:

    def test_masks_api_key(self):
        result = mask_secrets({'api_key': 'sk-abc123'})
        assert result == {'api_key': SECRET_MASK}

    def test_masks_password_and_token(self):
        result = mask_secrets({'password': 'p', 'api_token': 't', 'token': 'x'})
        assert result == {'password': SECRET_MASK, 'api_token': SECRET_MASK, 'token': SECRET_MASK}

    def test_does_not_mask_empty_or_none(self):
        """Пустые секреты не маскируются — UI должен видеть что значение не задано."""
        result = mask_secrets({'api_key': '', 'password': None})
        assert result == {'api_key': '', 'password': None}

    def test_recurses_into_nested_dicts(self):
        result = mask_secrets({'llm': {'api_key': 'secret', 'model': 'm'}})
        assert result == {'llm': {'api_key': SECRET_MASK, 'model': 'm'}}

    def test_recurses_into_lists(self):
        result = mask_secrets({'items': [{'api_key': 's1'}, {'api_key': 's2'}]})
        assert result == {'items': [{'api_key': SECRET_MASK}, {'api_key': SECRET_MASK}]}

    def test_does_not_mutate_input(self):
        data = {'api_key': 'secret'}
        mask_secrets(data)
        assert data == {'api_key': 'secret'}

    def test_passthrough_for_non_secret_fields(self):
        result = mask_secrets({'model': 'gpt-4', 'temperature': 0.5})
        assert result == {'model': 'gpt-4', 'temperature': 0.5}


# ---------------------------------------------------------------------------
# strip_masked_secrets
# ---------------------------------------------------------------------------

class TestStripMaskedSecrets:

    def test_strips_masked_api_key(self):
        """Если UI вернул '***' — это плейсхолдер, не настоящее значение, удалить."""
        result = strip_masked_secrets({'llm': {'api_key': SECRET_MASK, 'model': 'm'}})
        assert result == {'llm': {'model': 'm'}}

    def test_keeps_real_secret(self):
        """Если UI прислал реальный новый секрет — сохранить."""
        result = strip_masked_secrets({'llm': {'api_key': 'new-real-key'}})
        assert result == {'llm': {'api_key': 'new-real-key'}}

    def test_keeps_empty_string_secret(self):
        """Пустая строка ≠ маска — пользователь, возможно, хочет очистить ключ."""
        result = strip_masked_secrets({'llm': {'api_key': ''}})
        assert result == {'llm': {'api_key': ''}}


# ---------------------------------------------------------------------------
# read_layered / read_local / write_local / patch_local
# ---------------------------------------------------------------------------

class TestReadWrite:

    def test_read_layered_without_local(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        result = read_layered(cfg)
        assert result == PRIMARY

    def test_read_layered_with_local_overrides(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        local = tmp_path / 'config.local.yml'
        write_yaml(cfg, PRIMARY)
        # Overlay меняет qdrant.host (типичный сценарий), остальное наследуется
        write_yaml(local, {'qdrant': {'host': 'overridden'}})

        result = read_layered(cfg)
        assert result['qdrant']['host'] == 'overridden'
        # llms из primary остались (overlay не трогал)
        assert result['llms'][0]['model'] == 'primary-model'

    def test_read_local_returns_empty_when_missing(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        assert read_local(cfg) == {}

    def test_write_local_creates_file(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        write_local(cfg, {'llm': {'model': 'overridden'}})

        local = tmp_path / 'config.local.yml'
        assert local.exists()
        assert yaml.safe_load(local.read_text()) == {'llm': {'model': 'overridden'}}

    def test_write_local_overwrites_completely(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        local = tmp_path / 'config.local.yml'
        write_yaml(cfg, PRIMARY)
        write_local(cfg, {'a': 1, 'b': 2})

        write_local(cfg, {'c': 3})
        assert yaml.safe_load(local.read_text()) == {'c': 3}

    def test_patch_local_merges_with_existing(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        write_local(cfg, {'llm': {'model': 'm1', 'api_key': 'k1'}})

        patch_local(cfg, {'llm': {'model': 'm2'}})
        result = read_local(cfg)
        assert result == {'llm': {'model': 'm2', 'api_key': 'k1'}}

    def test_write_local_atomic_no_tmp_file_left(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        write_local(cfg, {'x': 1})

        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert 'config.local.yml.tmp' not in siblings


# ---------------------------------------------------------------------------
# validate_merged
# ---------------------------------------------------------------------------

class TestValidateMerged:

    def test_valid_overlay_passes(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        # Overlay меняет qdrant.host (типичный local-dev сценарий)
        config_obj = validate_merged(cfg, {'qdrant': {'host': 'localhost'}})
        assert config_obj.qdrant.host == 'localhost'

    def test_invalid_overlay_raises(self, tmp_path: Path):
        from pydantic import ValidationError
        import pytest
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)
        # qdrant.port должен быть int; передадим строку
        with pytest.raises(ValidationError):
            validate_merged(cfg, {'qdrant': {'port': 'not-a-number'}})


# ---------------------------------------------------------------------------
# ruamel.yaml inline-комменты для chip-полей
# ---------------------------------------------------------------------------

class TestCommentedIdList:

    def test_writes_inline_comments(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)

        seq = make_commented_id_list([
            {'id': '1234', 'comment': 'DOCS / Архитектура'},
            {'id': '5678', 'comment': 'DOCS / Требования'},
        ])
        write_local(cfg, {'sources': [{
            'kind': 'confluence', 'name': 'corp', 'ancestor_ids': seq,
        }]})

        text = (tmp_path / 'config.local.yml').read_text()
        # Каждый ID на своей строке, в той же строке — комментарий
        assert '"1234"' in text or "'1234'" in text or '1234' in text
        assert '# DOCS / Архитектура' in text
        assert '# DOCS / Требования' in text

    def test_empty_comment_no_hash(self, tmp_path: Path):
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)

        seq = make_commented_id_list([
            {'id': '1234', 'comment': ''},
            {'id': '5678'},                # без поля comment
        ])
        write_local(cfg, {'sources': [{'kind': 'c', 'name': 'x', 'ancestor_ids': seq}]})

        text = (tmp_path / 'config.local.yml').read_text()
        # Когда комментариев нет — символ # вообще не должен фигурировать в этой секции
        # (мы не добавляем eol-комменты для пустых)
        assert '#' not in text

    def test_roundtrip_keeps_comments(self, tmp_path: Path):
        # write → read через ruamel должен сохранить комменты как метаданные.
        # Здесь проверяем что они хотя бы переживают чтение и не падают.
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)

        seq = make_commented_id_list([{'id': '1234', 'comment': 'Архитектура'}])
        write_local(cfg, {'sources': [{
            'kind': 'confluence', 'name': 'c', 'ancestor_ids': seq,
        }]})

        local = read_local(cfg)
        assert list(local['sources'][0]['ancestor_ids']) == ['1234']

    def test_patch_local_preserves_chip_comments(self, tmp_path: Path):
        # Если мы сделали apply через patch_local — комменты остались в файле
        cfg = tmp_path / 'config.yml'
        write_yaml(cfg, PRIMARY)

        seq = make_commented_id_list([{'id': '777', 'comment': 'Раздел A'}])
        patch_local(cfg, {'sources': [{
            'kind': 'confluence', 'name': 'c', 'ancestor_ids': seq,
        }]})
        # Второй patch — не трогаем sources, добавляем qdrant overlay
        patch_local(cfg, {'qdrant': {'host': 'newhost'}})

        text = (tmp_path / 'config.local.yml').read_text()
        assert '# Раздел A' in text
        assert 'newhost' in text
