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
