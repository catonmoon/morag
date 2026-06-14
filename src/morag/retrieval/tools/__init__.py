"""Реестр типов агентских тулов (config-driven, см. retrieval.tools в config.yml).

Каждый ТИП тула описан одним `ToolSpec` в своём модуле пакета:
  - `core.py`    — системные search / find_section / get_doc (нельзя выключить,
                   description тюнится конфигом; у find_section — политика required);
  - `lookup.py`  — точечное обращение к справочным страницам (бывш. glossary);
  - `catalog.py` — структурный каталог корпуса (метаданные всех документов).

ИНСТАНСЫ тулов объявляются в конфиге списком `retrieval.tools` — у одного типа
может быть несколько инстансов с разными `name`/`description`/параметрами
(например, два lookup: глоссарий и страницы политик).

Как добавить новый тип тула:
  1. модуль в этом пакете с `SPEC = ToolSpec(...)` (+ handler-метод в Pipeline,
     если нужна логика с доступом к qdrant/searcher);
  2. регистрация в `REGISTRY` ниже;
  3. Pydantic-модель конфига в `morag.config` (union `RetrievalToolConfig`).
Seam: внешние HTTP/MCP-тулы — отдельный тип, чей handler проксирует вызов
на endpoint из конфига (транспорт пишется один раз).

Контракт handler'а: `(pipeline, cfg: dict, args: dict) -> tuple[str, list[dict]]`
— (текст для LLM, чанки для цитат). `cfg` — model_dump инстанса из конфига,
поэтому handler инстанс-aware (у каждого lookup — СВОИ doc_ids).
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple


class ToolSpec(NamedTuple):
    """Описание ТИПА тула — всё, что нужно pipeline'у, в одном месте."""
    type: str                # discriminator (= type в конфиге)
    default_name: str        # имя функции для агента, если cfg.name пуст
    core: bool               # системный: нельзя выключить/удалить
    parameters: dict         # JSON-schema параметров функции (OpenAI tools)
    describe: Callable       # (cfg: dict) -> str — description в schema; интерпретирует cfg['description']
    prompt_section: Callable # (cfg: dict) -> str — секция system-prompt ('' = нет)
    handler: Callable | None # (pipeline, cfg, args) -> (text, chunks); None = core-dispatch
    status: Callable | None  # (cfg, args, resolve_title) -> str; None = generic
    icon: str = '🛠️'


def _build_registry() -> dict[str, ToolSpec]:
    from morag.retrieval.tools import catalog, core, lookup
    specs = [*core.SPECS, lookup.SPEC, catalog.SPEC]
    return {s.type: s for s in specs}


REGISTRY: dict[str, ToolSpec] = _build_registry()


def build_tool_schema(spec: ToolSpec, cfg: dict[str, Any]) -> dict:
    """OpenAI function-schema инстанса: имя из cfg (или дефолт типа), description
    из cfg (или дефолт-билдер типа), параметры — шаблон типа."""
    return {
        'type': 'function',
        'function': {
            'name': cfg.get('name') or spec.default_name,
            'description': spec.describe(cfg),
            'parameters': spec.parameters,
        },
    }
