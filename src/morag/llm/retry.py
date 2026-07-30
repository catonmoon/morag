from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """Политика повторных попыток с экспоненциальной задержкой.

    max_retries=0 означает один вызов без повторов.
    Дефолты прежние (1s × 2.0); delay/backoff настраиваемы — транзиентный спайк провайдера
    живёт дольше секунды, и потребителю (asr-adaptor) нужны паузы порядка 10с.
    """

    max_retries: int = 3
    delay: float = 1.0
    backoff: float = 2.0

    async def call(self, coro_factory: Callable[[], Awaitable[T]], context: str = '') -> T:
        """Выполнить async-вызов с повторами по политике.

        coro_factory — фабрика корутины (например lambda: client.complete(...));
        вызывается заново при каждой попытке.
        Бросает последнее исключение если все попытки исчерпаны.
        """
        ctx = f' [{context}]' if context else ''
        last_exc: BaseException = RuntimeError('no attempts made')
        current_delay = self.delay

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
                    current_delay *= self.backoff
                else:
                    logger.warning(
                        'Call failed%s: all %d attempt(s) exhausted: %s',
                        ctx, self.max_retries + 1, exc,
                    )

        raise last_exc

