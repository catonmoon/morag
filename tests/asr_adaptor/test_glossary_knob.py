"""Потолок ответа глоссария — ручка окружения; умолчание прежнее (8000)."""
import importlib
import sys


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('ASR_GLOSSARY_MAX_TOKENS', raising=False)
    else:
        monkeypatch.setenv('ASR_GLOSSARY_MAX_TOKENS', value)
    sys.modules.pop('stages.glossary', None)
    return importlib.import_module('stages.glossary')


def test_default_ceiling_is_unchanged(monkeypatch):
    assert _reload(monkeypatch, None)._GLOSSARY_MAX_TOKENS == 8000


def test_ceiling_comes_from_the_environment(monkeypatch):
    assert _reload(monkeypatch, '4000')._GLOSSARY_MAX_TOKENS == 4000
    _reload(monkeypatch, None)
