"""Чтение, маскировка и запись конфигов morag для Console API.

Layering:
    config.yml         — primary, под git, дефолты + публичные настройки
    config.local.yml   — overlay, gitignored, секреты + user-overrides от UI

Console читает merged-вид (primary deep-merged с local) для отображения,
маскирует секреты перед отдачей наружу, и пишет правки только в local.

Используем ruamel.yaml round-trip parser — он сохраняет inline-комменты
при чтении/записи. Например, для Confluence ancestor_ids backend записывает
ID + breadcrumb-путь как комментарий, чтобы юзер мог глазами понять
содержимое overlay'а.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from morag.config import Config, _deep_merge

# Один глобальный YAML-парсер. typ='rt' — round-trip с сохранением комментов.
# allow_unicode не нужен (ruamel пишет UTF-8 по умолчанию).
_yaml = YAML(typ='rt')
_yaml.preserve_quotes = True
_yaml.width = 4096                  # не лочить длинные строки
_yaml.indent(mapping=2, sequence=4, offset=2)

# Поля, значение которых маскируется в API-выдаче.
# Пробег рекурсивный по всем dict'ам — match по точному имени ключа.
SECRET_KEYS = frozenset({'api_key', 'password', 'api_token', 'token'})

# Маска, отдаваемая в UI вместо реального секрета.
SECRET_MASK = '***'


def read_layered(primary_path: str | Path) -> dict[str, Any]:
    """Прочитать primary + local и вернуть результат deep-merge.

    Возвращает RAW dict (без Pydantic-валидации) — для отображения в UI и
    для последующей записи. Структура та же, что Config.model_validate ожидает.
    """
    primary_path = Path(primary_path)
    with open(primary_path, encoding='utf-8') as f:
        primary = _yaml.load(f) or {}

    local_path = _local_path(primary_path)
    if local_path.exists():
        with open(local_path, encoding='utf-8') as f:
            local = _yaml.load(f) or {}
        return _deep_merge(primary, local)

    return primary


def read_local(primary_path: str | Path) -> dict[str, Any]:
    """Прочитать только overlay (config.local.yml). {} если файла нет."""
    local_path = _local_path(Path(primary_path))
    if not local_path.exists():
        return {}
    with open(local_path, encoding='utf-8') as f:
        return _yaml.load(f) or {}


def write_local(primary_path: str | Path, local_data: dict[str, Any]) -> None:
    """Перезаписать config.local.yml.

    БЕЗ atomic rename: в docker-compose файл монтируется как bind-mount
    (одиночный файл), и os.replace на таком mount'е падает с EBUSY — нельзя
    подменить inode bind-mounted файла из контейнера. Console — единственный
    писатель в local.yml, конкурентных записей нет, поэтому non-atomic безопасно.
    Худший случай: console упал посреди записи → файл получился частичный,
    юзер чинит руками. Маловероятно.
    """
    local_path = _local_path(Path(primary_path))
    local_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    _yaml.dump(local_data, buf)
    local_path.write_text(buf.getvalue())


def patch_local(primary_path: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge patch в существующий config.local.yml, записать, вернуть новое содержимое.

    patch — частичные правки от UI (например, {'llm': {'model': 'grok-4-1-fast'}}).
    Существующие поля overlay'а сохраняются, новые добавляются, конфликты — patch выигрывает.
    """
    current = read_local(primary_path)
    merged = _deep_merge(current, patch)
    write_local(primary_path, merged)
    return merged


def validate_merged(primary_path: str | Path, candidate_local: dict[str, Any]) -> Config:
    """Проверить что primary + candidate_local даёт валидный Config.

    Бросает pydantic.ValidationError при несоответствии — caller (FastAPI) превращает в HTTP 400.
    """
    primary_path = Path(primary_path)
    with open(primary_path, encoding='utf-8') as f:
        primary = _yaml.load(f) or {}
    merged = _deep_merge(primary, candidate_local)
    return Config.model_validate(merged)


def make_commented_id_list(
    items: list[dict[str, str]],
) -> CommentedSeq:
    """Список ID-чипов с inline-комментами для записи в YAML.

    Вход: [{'id': '1234567', 'comment': 'Space / Раздел / Страница'}, ...].
    `comment` опционален — без него ID пишется без комментария.
    Использовать в _build_confluence для ancestor_ids/skip_ancestor_ids.

    Результат — CommentedSeq, который ruamel.yaml сериализует как:
        - "1234567"   # Space / Раздел / Страница
    """
    seq = CommentedSeq()
    for idx, item in enumerate(items):
        seq.append(str(item['id']))
        comment = (item.get('comment') or '').strip()
        if comment:
            # column=0 → ruamel сам подберёт отступ; \n не нужен
            seq.yaml_add_eol_comment(comment, key=idx, column=0)
    return seq


def mask_secrets(data: Any) -> Any:
    """Рекурсивно заменить значения секретных полей на '***'.

    Не мутирует вход — возвращает новый dict/list. Пустые/None секреты не маскируются
    (чтобы UI видел что ключ ещё не задан).
    """
    if isinstance(data, dict):
        return {
            k: SECRET_MASK if (k in SECRET_KEYS and v) else mask_secrets(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [mask_secrets(item) for item in data]
    return data


def strip_masked_secrets(patch: Any) -> Any:
    """Удалить из patch'а поля со значением SECRET_MASK.

    UI отдаёт обратно полный конфиг включая masked-поля. Сохранять '***' в local.yml
    нельзя — это перетрёт реальные секреты. Поэтому удаляем такие поля из patch'а до записи.
    """
    if isinstance(patch, dict):
        result = {}
        for k, v in patch.items():
            if k in SECRET_KEYS and v == SECRET_MASK:
                continue
            result[k] = strip_masked_secrets(v)
        return result
    if isinstance(patch, list):
        return [strip_masked_secrets(item) for item in patch]
    return patch


def _local_path(primary_path: Path) -> Path:
    return primary_path.with_name('config.local.yml')


__all__ = [
    'SECRET_KEYS',
    'SECRET_MASK',
    'make_commented_id_list',
    'mask_secrets',
    'patch_local',
    'read_layered',
    'read_local',
    'strip_masked_secrets',
    'validate_merged',
    'write_local',
    'ValidationError',
]
