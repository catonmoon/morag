"""Секция `indexing.annotations` (ADR-0027): нет — None; есть — парсится с умолчаниями."""
from morag.config import AnnotationsConfig, Config

_BASE = {
    'sources': [{'kind': 'local', 'name': 'docs', 'path': 'data/'}],
    'llms': [{'name': 'main', 'base_url': 'x', 'model': 'm', 'api_key': 'k'}],
}


def test_absent_by_default():
    cfg = Config.model_validate({**_BASE, 'indexing': {}})
    assert cfg.indexing.annotations is None


def test_section_parses_with_defaults():
    cfg = Config.model_validate({
        **_BASE,
        'indexing': {'annotations': {'boundaries': {'enabled': True, 'window_sec': 8}}},
    })
    ann = cfg.indexing.annotations
    assert isinstance(ann, AnnotationsConfig)
    assert ann.suffix == '.annotations.json'
    assert ann.boundaries.enabled is True
    assert ann.boundaries.window_sec == 8.0
    assert ann.boundaries.min_tokens == 120
    assert ann.field is None
