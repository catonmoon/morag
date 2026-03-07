from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


_DELAY = 1.0    # начальная задержка между попытками (секунды)
_BACKOFF = 2.0  # множитель: 1s → 2s → 4s → ...


@dataclass
class RetryPolicy:
    """Политика повторных попыток с экспоненциальной задержкой.

    max_retries=0 означает один вызов без повторов.
    Задержка и backoff фиксированы: начальная 1s, множитель 2.0.
    """

    max_retries: int = 3

    async def call(self, coro_factory: Callable[[], Awaitable[T]], context: str = '') -> T:
        """Выполнить async-вызов с повторами по политике.

        coro_factory — фабрика корутины (например lambda: client.complete(...));
        вызывается заново при каждой попытке.
        Бросает последнее исключение если все попытки исчерпаны.
        """
        ctx = f' [{context}]' if context else ''
        last_exc: BaseException = RuntimeError('no attempts made')
        current_delay = _DELAY

        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        'Call failed%s (attempt %d/%d): %s — retrying in %.1fs...',
                        ctx, attempt + 1, self.max_retries + 1, exc, current_delay,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= _BACKOFF
                else:
                    logger.warning(
                        'Call failed%s: all %d attempt(s) exhausted: %s',
                        ctx, self.max_retries + 1, exc,
                    )

        raise last_exc

    def call_sync(self, fn: Callable[[], T], context: str = '') -> T:
        """Выполнить синхронный вызов с повторами по политике.

        fn — вызываемый объект без аргументов.
        Бросает последнее исключение если все попытки исчерпаны.
        """
        ctx = f' [{context}]' if context else ''
        last_exc: BaseException = RuntimeError('no attempts made')
        current_delay = _DELAY

        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        'Call failed%s (attempt %d/%d): %s — retrying in %.1fs...',
                        ctx, attempt + 1, self.max_retries + 1, exc, current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= _BACKOFF
                else:
                    logger.warning(
                        'Call failed%s: all %d attempt(s) exhausted: %s',
                        ctx, self.max_retries + 1, exc,
                    )

        raise last_exc
