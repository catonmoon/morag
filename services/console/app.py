"""FastAPI приложение Console.

Bind на 127.0.0.1:8000 по умолчанию (см. compose). Никакой auth — рассчитано
на single-tenant локальный деплой через docker-compose.

Console — тонкий слой UI/API. Управление индексацией делегируется
control-plane'у в morag-indexer'е через HTTP (см. IndexerClient).

Конфигурация через env:
    MORAG_CONFIG_PATH   путь к config.yml (default: ./config.yml)
    MORAG_INDEXER_URL   URL control-plane'а (default: http://morag-indexer:9090)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.console.indexer_client import IndexerClient
from services.console.routes import config as config_routes
from services.console.routes import index as index_routes
from services.console.routes import knowledge_map as km_routes
from services.console.routes import presets as presets_routes
from services.console.routes import schedule as schedule_routes
from services.console.routes import setup as setup_routes
from services.console.routes import stats as stats_routes

logger = logging.getLogger(__name__)


def _resolve_config_path() -> Path:
    return Path(os.environ.get('MORAG_CONFIG_PATH', 'config.yml')).resolve()


def _resolve_indexer_url() -> str:
    return os.environ.get('MORAG_INDEXER_URL', 'http://morag-indexer:9090')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Положить в app.state config_path и IndexerClient (HTTP-клиент к indexer'у)."""
    config_path = _resolve_config_path()
    indexer_url = _resolve_indexer_url()

    logger.info('Console starting: config=%s indexer=%s', config_path, indexer_url)

    app.state.config_path = config_path
    app.state.indexer = IndexerClient(indexer_url)

    yield


def create_app() -> FastAPI:
    app = FastAPI(title='morag console', version='0.1.0', lifespan=lifespan)

    # CORS — нужно если UI host'ится отдельно (dev-режим). В production UI и API
    # на одном origin, CORS бесполезен но и не мешает.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(config_routes.router, prefix='/api/config', tags=['config'])
    app.include_router(index_routes.router, prefix='/api/index', tags=['index'])
    app.include_router(stats_routes.router, prefix='/api', tags=['stats'])
    app.include_router(km_routes.router, prefix='/api/knowledge-map', tags=['knowledge-map'])
    app.include_router(presets_routes.router, prefix='/api/presets', tags=['presets'])
    app.include_router(setup_routes.router, prefix='/api/setup', tags=['setup'])
    app.include_router(schedule_routes.router, prefix='/api/schedule', tags=['schedule'])

    # Статика UI (Этап 4). Каталог может быть пустым — mount всё равно работает.
    static_dir = Path(__file__).parent / 'static'
    if static_dir.exists():
        app.mount('/', StaticFiles(directory=static_dir, html=True), name='static')

    return app


app = create_app()
