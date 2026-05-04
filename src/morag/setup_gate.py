"""Setup-gate — проверка что юзер закончил initial config через UI.

Цель: не запускать индексацию (manual или cron) пока config выглядит как
свежераспакованный example. Иначе первая индексация валится с
непонятными ошибками на placeholder-секретах.

«Закончил initial config» = есть `config.local.yml` (юзер тронул через
Console UI хотя бы раз) И merged-config валиден через Pydantic.

Gate проверяется в edge-points: на cron tick + на /control/start клик.
Не watch'им state — текущий running task живёт со своим уже-загруженным
конфигом независимо от gate.
"""
from __future__ import annotations

from pathlib import Path


class SetupIncomplete(RuntimeError):
    """Setup-gate не пройден — индексацию запускать нельзя.

    `blockers` — list[str], каждая строка — отдельная причина для UI.
    """

    def __init__(self, blockers: list[str]) -> None:
        super().__init__('; '.join(blockers))
        self.blockers = blockers


def is_setup_complete(config_path: str | Path) -> tuple[bool, list[str]]:
    """Returns (ok, blockers). Если ok=True — blockers пустой."""
    cfg_path = Path(config_path)
    blockers: list[str] = []

    local_path = cfg_path.with_name('config.local.yml')
    if not local_path.exists() or _is_empty_yaml(local_path):
        # Пустой файл считается «не настроен» — для docker bind mount файл часто
        # создаётся пустым (touch) до запуска, иначе docker создаст директорию.
        blockers.append(
            'Конфигурация не настроена — откройте раздел Setup '
            'и добавьте источник или LLM, чтобы продолжить.'
        )
        return False, blockers

    # Конфиг грузится через Pydantic (не сломан после правок)
    from morag.config import load_config
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        blockers.append(f'Некорректная конфигурация: {type(e).__name__}: {e}')
        return False, blockers

    # Проверки готовности к индексации (Pydantic-схема ослаблена, но индексер
    # требует все эти компоненты для работы)
    if not cfg.sources:
        blockers.append('Нет ни одного источника — добавьте через Setup → Источники.')
    if not cfg.llms:
        blockers.append('Нет ни одной LLM — добавьте через Setup → LLM.')
    if cfg.indexing is None:
        blockers.append('Секция indexing отсутствует.')
    else:
        if cfg.indexing.dense_embedder is None:
            blockers.append('Embedder не настроен — Setup → Embedder.')
        if cfg.indexing.llm is None or cfg.indexing.vision is None:
            blockers.append('Роли LLM/vision не назначены — Setup → Роли.')

    return (not blockers), blockers


def require_setup_complete(config_path: str | Path) -> None:
    """Convenience: бросает SetupIncomplete если gate не пройден."""
    ok, blockers = is_setup_complete(config_path)
    if not ok:
        raise SetupIncomplete(blockers)


def _is_empty_yaml(path: Path) -> bool:
    """True если файл пустой / только комментарии / yaml.safe_load → None или {}."""
    try:
        import yaml
        text = path.read_text()
        if not text.strip():
            return True
        data = yaml.safe_load(text)
        return data is None or data == {}
    except Exception:
        # Некорректный YAML — Pydantic поймает ниже как «config invalid»
        return False
