"""IndexerClient — async httpx-обёртка над HTTP control-plane'ом morag-indexer'а.

Заменяет ProcessManager: вместо спавна subprocess'ов делает HTTP-вызовы
к /control/start /stop /kill /status /rebuild-km.

URL берётся из env MORAG_INDEXER_URL, default http://morag-indexer:9090
(имя сервиса в docker-compose).
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_STOP_GRACE_SECONDS = 180


class IndexerError(RuntimeError):
    """Ошибка связи с indexer-control-plane'ом."""


class AlreadyRunning(RuntimeError):
    """Indexer вернул 409 — задача уже запущена."""


class SetupIncomplete(RuntimeError):
    """Indexer вернул 412 — конфиг ещё не настроен через UI.

    `blockers` — list[str] от control-plane'а, для отображения юзеру.
    """

    def __init__(self, blockers: list[str]) -> None:
        super().__init__('; '.join(blockers))
        self.blockers = blockers


class IndexerClient:
    """Тонкий HTTP-клиент. Не держит состояния — только URL."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip('/')
        # stop может ждать долго — отдельный timeout не нужен, control-plane
        # сам ограничит через grace_seconds.
        self._timeout = httpx.Timeout(timeout, read=300.0)
        # transport — для тестов: можно подсунуть ASGITransport(fake_app)
        # вместо реальных сетевых вызовов.
        self._transport = transport

    async def status(self) -> dict[str, Any]:
        return await self._get('/control/status')

    async def setup_status(self) -> dict[str, Any]:
        return await self._get('/control/setup-status')

    async def start_index(self, reset: bool = False) -> dict[str, Any]:
        return await self._post('/control/start', {'reset': reset})

    async def start_rebuild_km(self) -> dict[str, Any]:
        return await self._post('/control/rebuild-km', {})

    async def reindex(self, scope: str = 'all') -> dict[str, Any]:
        """Плавный реиндекс scope ('all' или имя источника), ADR-0014."""
        return await self._post('/control/reindex', {'scope': scope})

    async def stop(self, grace_seconds: int = DEFAULT_STOP_GRACE_SECONDS) -> dict[str, Any]:
        return await self._post('/control/stop', {'grace_seconds': grace_seconds})

    async def kill(self) -> dict[str, Any]:
        return await self._post('/control/kill', {})

    async def reload_schedule(self) -> dict[str, Any]:
        """Hot-reload cron из обновлённого config.local.yml. Возвращает active expression."""
        return await self._post('/control/reload-schedule', {})

    # ---- internals ----

    async def _get(self, path: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
            try:
                r = await c.get(self._base_url + path)
            except httpx.HTTPError as e:
                raise IndexerError(f'GET {path}: {type(e).__name__}: {e}') from e
        self._raise_for_status(r, path)
        return r.json()

    async def _post(self, path: str, body: dict) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
            try:
                r = await c.post(self._base_url + path, json=body)
            except httpx.HTTPError as e:
                raise IndexerError(f'POST {path}: {type(e).__name__}: {e}') from e
        self._raise_for_status(r, path)
        return r.json()

    @staticmethod
    def _raise_for_status(r: httpx.Response, path: str) -> None:
        if r.status_code == 409:
            raise AlreadyRunning(r.text)
        if r.status_code == 412:
            try:
                detail = r.json().get('detail', {})
                blockers = detail.get('blockers') if isinstance(detail, dict) else None
            except Exception:
                blockers = None
            raise SetupIncomplete(blockers or [r.text[:200]])
        if r.status_code >= 400:
            raise IndexerError(f'{path}: HTTP {r.status_code} — {r.text[:200]}')
