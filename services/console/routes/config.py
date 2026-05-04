"""GET/PUT /api/config + POST /api/config/test."""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from services.console.config_io import (
    mask_secrets,
    patch_local,
    read_layered,
    strip_masked_secrets,
    validate_merged,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ConfigPatchRequest(BaseModel):
    patch: dict[str, Any]


class ConfigTestRequest(BaseModel):
    target: Literal['llm', 'dense_embedder', 'sparse_embedder', 'qdrant']
    name: str | None = None  # для target='llm': имя инстанса из llms-pool


class ConfigTestResponse(BaseModel):
    ok: bool
    detail: str | None = None


@router.get('')
async def get_config(request: Request) -> dict[str, Any]:
    """Вернуть merged config с замаскированными секретами."""
    cfg_path = request.app.state.config_path
    merged = read_layered(cfg_path)
    return mask_secrets(merged)


@router.put('')
async def put_config(req: ConfigPatchRequest, request: Request) -> dict[str, Any]:
    """Применить patch к config.local.yml.

    Поля со значением '***' (плейсхолдер masked-секрета) удаляются из patch'а —
    иначе они затёрли бы реальные секреты в local.yml.
    """
    cfg_path = request.app.state.config_path
    clean_patch = strip_masked_secrets(req.patch)

    # Сначала валидируем потенциально-новый local поверх primary
    from morag.config import _deep_merge

    from services.console.config_io import read_local
    candidate_local = _deep_merge(read_local(cfg_path), clean_patch)
    try:
        validate_merged(cfg_path, candidate_local)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors(include_url=False, include_input=False, include_context=False),
        ) from e

    # Валидно — пишем
    new_local = patch_local(cfg_path, clean_patch)
    return mask_secrets(new_local)


@router.post('/test', response_model=ConfigTestResponse)
async def test_config(req: ConfigTestRequest, request: Request) -> ConfigTestResponse:
    """Smoke-test провайдера на текущем merged-конфиге.

    LLM/embedder: тривиальный реальный вызов (1 token / 1 короткий текст).
    Qdrant: get_collections.
    """
    cfg_path = request.app.state.config_path
    from morag.config import load_config

    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        return ConfigTestResponse(ok=False, detail=f'Config invalid: {e}')

    try:
        if req.target == 'qdrant':
            from qdrant_client import AsyncQdrantClient
            client = AsyncQdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=10)
            try:
                cols = await client.get_collections()
                return ConfigTestResponse(ok=True, detail=f'{len(cols.collections)} collections')
            finally:
                await client.close()

        if req.target == 'llm':
            # Имя из pool. Если не указано — используем default из indexing.llm.
            name = req.name
            if name is None:
                if cfg.indexing is None or cfg.indexing.llm is None:
                    return ConfigTestResponse(ok=False, detail='No name; indexing.llm not configured')
                name = cfg.indexing.llm.default
            try:
                llm_inst = cfg.llm_by_name(name)
            except KeyError:
                return ConfigTestResponse(ok=False, detail=f'LLM {name!r} not in pool')
            return await _test_llm(llm_inst)

        if req.target == 'dense_embedder':
            if cfg.indexing is None or cfg.indexing.dense_embedder is None or not cfg.indexing.dense_embedder.base_url:
                return ConfigTestResponse(ok=False, detail='dense_embedder.base_url not configured')
            return await _test_dense_embedder(cfg.indexing.dense_embedder)

        if req.target == 'sparse_embedder':
            if cfg.indexing is None or not cfg.indexing.sparse_embedder.base_url:
                return ConfigTestResponse(ok=False, detail='sparse_embedder.base_url not configured')
            return await _test_sparse_embedder(cfg.indexing.sparse_embedder)

    except Exception as e:
        logger.exception('Test connection failed for %s', req.target)
        return ConfigTestResponse(ok=False, detail=f'{type(e).__name__}: {e}')

    return ConfigTestResponse(ok=False, detail='Unknown target')


async def _test_llm(llm_cfg) -> ConfigTestResponse:
    import time
    from morag.llm.client import GenerationParams, LLMClient
    client = LLMClient(
        base_url=llm_cfg.base_url,
        model=llm_cfg.model,
        api_key=llm_cfg.api_key,
        timeout=15,
        max_retries=0,
        enable_thinking=False,
    )
    t0 = time.monotonic()
    # max_tokens — kwarg complete(), не поле GenerationParams
    answer = await client.complete(
        messages=[{'role': 'user', 'content': 'ping'}],
        params=GenerationParams(temperature=0),
        max_tokens=1,
    )
    ms = int((time.monotonic() - t0) * 1000)
    snippet = answer.strip().replace('\n', ' ')
    if len(snippet) > 40:
        snippet = snippet[:40] + '…'
    return ConfigTestResponse(
        ok=True,
        detail=f'модель ответила за {ms} мс — «ping» → «{snippet}»',
    )


async def _test_dense_embedder(cfg) -> ConfigTestResponse:
    import time
    from morag.indexing.embedder import HttpEmbedder
    embedder = HttpEmbedder(
        cfg.base_url, cfg.model, cfg.dim,
        api_key=cfg.api_key or 'ollama',
        document_template=cfg.document_template,
        query_template=cfg.query_template,
        timeout=15,
        max_retries=0,
    )
    t0 = time.monotonic()
    vec = await embedder.embed_batch(['ping'])
    ms = int((time.monotonic() - t0) * 1000)
    return ConfigTestResponse(
        ok=True,
        detail=f'эмбеддер вернул вектор размерности {len(vec[0])} за {ms} мс',
    )


async def _test_sparse_embedder(cfg) -> ConfigTestResponse:
    from morag.indexing.embedder import HttpGteSparseEmbedder
    embedder = HttpGteSparseEmbedder(cfg.base_url, timeout=15)
    indices, values = await embedder.embed_query('ping')
    return ConfigTestResponse(ok=True, detail=f'{len(indices)} non-zero dims')
