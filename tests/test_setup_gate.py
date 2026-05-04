"""Тесты setup_gate — слабая проверка готовности конфига к индексации."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from morag.setup_gate import SetupIncomplete, is_setup_complete, require_setup_complete


VALID_CONFIG = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': '/x'}],
    'llms': [
        {'name': 'main', 'base_url': 'http://x/v1', 'model': 'm', 'api_key': 'k',
         'capabilities': ['text', 'vision']},
    ],
    'indexing': {
        'llm': 'main', 'vision': 'main',
        'dense_embedder': {'model': 'e', 'base_url': 'http://x/v1', 'dim': 8},
    },
}


def test_blocked_when_local_yml_missing(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    # local.yml не создан
    ok, blockers = is_setup_complete(cfg)
    assert ok is False
    assert any('Setup' in b for b in blockers)


def test_blocked_when_local_yml_empty(tmp_path: Path):
    # Empty file часто создаётся `touch` для docker bind mount — это «не настроен»
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    (tmp_path / 'config.local.yml').write_text('')
    ok, blockers = is_setup_complete(cfg)
    assert ok is False
    assert any('Setup' in b for b in blockers)


def test_blocked_when_local_yml_only_comments(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    (tmp_path / 'config.local.yml').write_text('# overlay placeholder\n# nothing yet\n')
    ok, blockers = is_setup_complete(cfg)
    assert ok is False
    assert any('Setup' in b for b in blockers)


def test_passes_when_local_yml_has_content(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    (tmp_path / 'config.local.yml').write_text('qdrant:\n  host: customhost\n')
    ok, blockers = is_setup_complete(cfg)
    assert ok is True
    assert blockers == []


def test_blocked_when_config_invalid(tmp_path: Path):
    # Local exists с реальным контентом, но primary невалиден (нет llms)
    cfg = tmp_path / 'config.yml'
    bad = {**VALID_CONFIG, 'llms': []}
    cfg.write_text(yaml.safe_dump(bad))
    (tmp_path / 'config.local.yml').write_text('qdrant:\n  host: x\n')
    ok, blockers = is_setup_complete(cfg)
    assert ok is False
    assert any('Некорректная конфигурация' in b for b in blockers)


def test_require_raises_with_blockers(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    with pytest.raises(SetupIncomplete) as exc:
        require_setup_complete(cfg)
    assert exc.value.blockers
    assert 'Setup' in exc.value.blockers[0]


def test_require_passes_silently(tmp_path: Path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text(yaml.safe_dump(VALID_CONFIG))
    (tmp_path / 'config.local.yml').write_text('qdrant:\n  host: x\n')
    require_setup_complete(cfg)  # не должно бросать
