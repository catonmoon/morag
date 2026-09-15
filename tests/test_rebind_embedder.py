"""`rebind-embedder`: тот же эмбеддер по другому адресу — переписать отпечаток, не векторы.

Отпечаток включает `base_url` намеренно (ADR-0012), и переезд индекса на сервер, где та же модель
отвечает по другому адресу, иначе означал бы полную переиндексацию. Здесь проверяется, что
команда трогает ТОЛЬКО документы со старым отпечатком, только payload, и что dry-run не пишет.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cli.main import cmd_rebind_embedder
from morag.indexing.embedder_fingerprint import compute_embedder_fingerprint

NEW_URL = 'https://gateway.example/api'
OLD_FP = compute_embedder_fingerprint('Qwen/Qwen3-Embedding-4B', 2560, 'http://host.docker.internal:11434/v1')
NEW_FP = compute_embedder_fingerprint('Qwen/Qwen3-Embedding-4B', 2560, NEW_URL)


def _config():
    return SimpleNamespace(
        qdrant=SimpleNamespace(host='q', port=6333, collection_docs='docs'),
        indexing=SimpleNamespace(dense_embedder=SimpleNamespace(
            model='Qwen/Qwen3-Embedding-4B', dim=2560, base_url=NEW_URL)),
    )


def _points():
    return [
        SimpleNamespace(id='p1', payload={'embedder_fingerprint': OLD_FP}),
        SimpleNamespace(id='p2', payload={'embedder_fingerprint': OLD_FP}),
        SimpleNamespace(id='p3', payload={'embedder_fingerprint': NEW_FP}),   # уже перепривязан
        SimpleNamespace(id='p4', payload={'embedder_fingerprint': 'other'}),  # чужой эмбеддер — не трогаем
    ]


@pytest.mark.asyncio
async def test_rebinds_only_documents_with_the_old_fingerprint():
    client = SimpleNamespace(
        scroll=AsyncMock(return_value=(_points(), None)),
        set_payload=AsyncMock(),
        close=AsyncMock(),
    )
    with patch('cli.main.load_config', return_value=_config()), \
         patch('cli.main.AsyncQdrantClient', return_value=client):
        await cmd_rebind_embedder('cfg.yml', OLD_FP)
    client.set_payload.assert_awaited_once()
    kwargs = client.set_payload.await_args.kwargs
    assert kwargs['collection_name'] == 'docs'
    assert kwargs['payload'] == {'embedder_fingerprint': NEW_FP}
    assert kwargs['points'] == ['p1', 'p2'], 'перепривязываются только документы со старым отпечатком'


@pytest.mark.asyncio
async def test_dry_run_writes_nothing():
    client = SimpleNamespace(
        scroll=AsyncMock(return_value=(_points(), None)),
        set_payload=AsyncMock(),
        close=AsyncMock(),
    )
    with patch('cli.main.load_config', return_value=_config()), \
         patch('cli.main.AsyncQdrantClient', return_value=client):
        await cmd_rebind_embedder('cfg.yml', OLD_FP, dry_run=True)
    client.set_payload.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_fingerprint_is_a_noop_without_touching_qdrant():
    client_factory = AsyncMock()
    with patch('cli.main.load_config', return_value=_config()), \
         patch('cli.main.AsyncQdrantClient', client_factory):
        await cmd_rebind_embedder('cfg.yml', NEW_FP)
    client_factory.assert_not_called()
