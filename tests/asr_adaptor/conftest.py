"""Тесты стадий services/asr-adaptor.

Каталог сервиса кладётся в `sys.path`: в имени дефис, пакетом его не импортировать (в отличие от
`services.console`), поэтому стадии берутся как верхнеуровневые модули `stages.*`. Покрываем
чистую арифметику — ни сети, ни аудио-бэкендов, ни торча.
"""
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[2] / 'services' / 'asr-adaptor'
if str(SERVICE) not in sys.path:
    sys.path.insert(0, str(SERVICE))

import pytest  # noqa: E402


@pytest.fixture
def silence(tmp_path: Path) -> Path:
    """Тишина на 90 с: покрытие считается по заголовку wav (см. fakes.py)."""
    from fakes import make_silence
    return make_silence(tmp_path / 'source.wav')


@pytest.fixture
def rich(monkeypatch, silence):
    """Богатый бэкенд-заглушка поверх `pipeline.*` — для графа и конвейера разом."""
    from fakes import install_rich
    return install_rich(monkeypatch, silence)
