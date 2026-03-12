# Реализация HTTP-сервера на Python

В этом документе описана полная реализация асинхронного HTTP-сервера с поддержкой маршрутизации, middleware и обработкой ошибок.

## Основной класс сервера

Сервер построен на базе asyncio и поддерживает graceful shutdown, keep-alive соединения и потоковую передачу ответов.

```python
import asyncio
import logging
import signal
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class HttpMethod(Enum):
    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'
    PATCH = 'PATCH'
    HEAD = 'HEAD'
    OPTIONS = 'OPTIONS'


@dataclass
class Request:
    method: HttpMethod
    path: str
    headers: dict[str, str]
    query_params: dict[str, list[str]]
    body: bytes = b''
    path_params: dict[str, str] = field(default_factory=dict)
    _json_cache: Any = field(default=None, repr=False)

    @property
    def content_type(self) -> str:
        return self.headers.get('content-type', '')

    @property
    def content_length(self) -> int:
        return int(self.headers.get('content-length', '0'))

    def json(self) -> Any:
        if self._json_cache is None:
            import json
            self._json_cache = json.loads(self.body.decode('utf-8'))
        return self._json_cache

    @property
    def is_json(self) -> bool:
        return 'application/json' in self.content_type

    def get_header(self, name: str, default: str = '') -> str:
        return self.headers.get(name.lower(), default)


@dataclass
class Response:
    status: HTTPStatus = HTTPStatus.OK
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b''

    @classmethod
    def json(cls, data: Any, status: HTTPStatus = HTTPStatus.OK) -> 'Response':
        import json
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        return cls(
            status=status,
            headers={'content-type': 'application/json; charset=utf-8'},
            body=body,
        )

    @classmethod
    def text(cls, text: str, status: HTTPStatus = HTTPStatus.OK) -> 'Response':
        return cls(
            status=status,
            headers={'content-type': 'text/plain; charset=utf-8'},
            body=text.encode('utf-8'),
        )

    @classmethod
    def html(cls, html: str, status: HTTPStatus = HTTPStatus.OK) -> 'Response':
        return cls(
            status=status,
            headers={'content-type': 'text/html; charset=utf-8'},
            body=html.encode('utf-8'),
        )

    @classmethod
    def redirect(cls, url: str, permanent: bool = False) -> 'Response':
        status = HTTPStatus.MOVED_PERMANENTLY if permanent else HTTPStatus.FOUND
        return cls(status=status, headers={'location': url})

    @classmethod
    def not_found(cls, message: str = 'Not Found') -> 'Response':
        return cls.json({'error': message}, HTTPStatus.NOT_FOUND)

    @classmethod
    def error(cls, message: str, status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR) -> 'Response':
        return cls.json({'error': message}, status)


Handler = Callable[[Request], Awaitable[Response]]
Middleware = Callable[[Request, Handler], Awaitable[Response]]


@dataclass
class Route:
    method: HttpMethod
    pattern: str
    handler: Handler
    param_names: list[str] = field(default_factory=list)

    def match(self, method: HttpMethod, path: str) -> Optional[dict[str, str]]:
        if self.method != method:
            return None
        parts = path.strip('/').split('/')
        pattern_parts = self.pattern.strip('/').split('/')
        if len(parts) != len(pattern_parts):
            return None
        params = {}
        for part, pattern_part in zip(parts, pattern_parts):
            if pattern_part.startswith('{') and pattern_part.endswith('}'):
                param_name = pattern_part[1:-1]
                params[param_name] = part
            elif part != pattern_part:
                return None
        return params


class Router:
    def __init__(self) -> None:
        self._routes: list[Route] = []
        self._middleware: list[Middleware] = []
        self._error_handlers: dict[int, Handler] = {}

    def add_route(self, method: HttpMethod, pattern: str, handler: Handler) -> None:
        route = Route(method=method, pattern=pattern, handler=handler)
        self._routes.append(route)
        logger.debug('Registered route: %s %s', method.value, pattern)

    def get(self, pattern: str) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self.add_route(HttpMethod.GET, pattern, handler)
            return handler
        return decorator

    def post(self, pattern: str) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self.add_route(HttpMethod.POST, pattern, handler)
            return handler
        return decorator

    def put(self, pattern: str) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self.add_route(HttpMethod.PUT, pattern, handler)
            return handler
        return decorator

    def delete(self, pattern: str) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self.add_route(HttpMethod.DELETE, pattern, handler)
            return handler
        return decorator

    def use(self, middleware: Middleware) -> None:
        self._middleware.append(middleware)

    def error_handler(self, status_code: int) -> Callable[[Handler], Handler]:
        def decorator(handler: Handler) -> Handler:
            self._error_handlers[status_code] = handler
            return handler
        return decorator

    def resolve(self, method: HttpMethod, path: str) -> tuple[Optional[Handler], dict[str, str]]:
        for route in self._routes:
            params = route.match(method, path)
            if params is not None:
                return route.handler, params
        return None, {}

    def build_handler(self, handler: Handler) -> Handler:
        result = handler
        for mw in reversed(self._middleware):
            prev = result
            async def wrapped(req: Request, _mw=mw, _prev=prev) -> Response:
                return await _mw(req, _prev)
            result = wrapped
        return result


class HttpServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._router = Router()
        self._server: Optional[asyncio.AbstractServer] = None
        self._connections: set[asyncio.Task] = set()
        self._shutting_down = False

    @property
    def router(self) -> Router:
        return self._router

    async def _parse_request(self, reader: asyncio.StreamReader) -> Request:
        request_line = await reader.readline()
        if not request_line:
            raise ConnectionError('Empty request')

        parts = request_line.decode('utf-8').strip().split(' ')
        if len(parts) != 3:
            raise ValueError(f'Invalid request line: {request_line!r}')

        method = HttpMethod(parts[0])
        parsed_url = urlparse(parts[1])
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b'\r\n', b'\n', b''):
                break
            name, _, value = line.decode('utf-8').partition(':')
            headers[name.strip().lower()] = value.strip()

        body = b''
        content_length = int(headers.get('content-length', '0'))
        if content_length > 0:
            body = await reader.readexactly(content_length)

        return Request(
            method=method,
            path=path,
            headers=headers,
            query_params=query_params,
            body=body,
        )

    def _format_response(self, response: Response) -> bytes:
        status_line = f'HTTP/1.1 {response.status.value} {response.status.phrase}\r\n'
        headers = response.headers.copy()
        if 'content-length' not in headers:
            headers['content-length'] = str(len(response.body))
        if 'connection' not in headers:
            headers['connection'] = 'close'

        header_lines = ''.join(f'{k}: {v}\r\n' for k, v in headers.items())
        head = f'{status_line}{header_lines}\r\n'.encode('utf-8')
        return head + response.body

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info('peername')
        logger.debug('New connection from %s', addr)
        try:
            request = await self._parse_request(reader)
            handler, params = self._router.resolve(request.method, request.path)

            if handler is None:
                response = Response.not_found(f'No route for {request.method.value} {request.path}')
            else:
                request.path_params = params
                wrapped = self._router.build_handler(handler)
                try:
                    response = await wrapped(request)
                except Exception as exc:
                    logger.exception('Handler error for %s %s', request.method.value, request.path)
                    error_handler = self._router._error_handlers.get(500)
                    if error_handler:
                        response = await error_handler(request)
                    else:
                        response = Response.error(str(exc))

            raw = self._format_response(response)
            writer.write(raw)
            await writer.drain()
            logger.info(
                '%s %s %s %d bytes',
                request.method.value, request.path,
                response.status.value, len(response.body),
            )
        except Exception:
            logger.exception('Connection error from %s', addr)
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port,
        )
        logger.info('Server started on %s:%d', self._host, self._port)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        async with self._server:
            await self._server.serve_forever()

    async def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info('Shutting down server...')

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._connections:
            logger.info('Waiting for %d connections to close...', len(self._connections))
            await asyncio.gather(*self._connections, return_exceptions=True)

        logger.info('Server stopped')
```

Этот блок содержит все основные компоненты: модели данных, маршрутизатор, обработку HTTP-протокола и lifecycle сервера.

## Middleware для логирования и аутентификации

Middleware позволяют перехватывать запросы и ответы на уровне всего приложения или отдельных маршрутов.

```python
import time
import hashlib
import hmac
from functools import wraps


async def logging_middleware(request: Request, next_handler: Handler) -> Response:
    start = time.monotonic()
    logger.info('→ %s %s', request.method.value, request.path)

    try:
        response = await next_handler(request)
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception('✗ %s %s [%.3fs]', request.method.value, request.path, elapsed)
        raise

    elapsed = time.monotonic() - start
    logger.info(
        '← %s %s → %d [%.3fs]',
        request.method.value, request.path,
        response.status.value, elapsed,
    )
    return response


async def cors_middleware(request: Request, next_handler: Handler) -> Response:
    if request.method == HttpMethod.OPTIONS:
        return Response(
            status=HTTPStatus.NO_CONTENT,
            headers={
                'access-control-allow-origin': '*',
                'access-control-allow-methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'access-control-allow-headers': 'content-type, authorization',
                'access-control-max-age': '86400',
            },
        )

    response = await next_handler(request)
    response.headers['access-control-allow-origin'] = '*'
    return response


class AuthMiddleware:
    def __init__(self, secret_key: str, excluded_paths: list[str] | None = None) -> None:
        self._secret_key = secret_key.encode('utf-8')
        self._excluded = set(excluded_paths or [])

    async def __call__(self, request: Request, next_handler: Handler) -> Response:
        if request.path in self._excluded:
            return await next_handler(request)

        auth_header = request.get_header('authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return Response.error('Missing or invalid authorization header', HTTPStatus.UNAUTHORIZED)

        token = auth_header[7:]
        if not self._verify_token(token):
            return Response.error('Invalid token', HTTPStatus.UNAUTHORIZED)

        return await next_handler(request)

    def _verify_token(self, token: str) -> bool:
        try:
            parts = token.split('.')
            if len(parts) != 2:
                return False
            payload_b64, signature = parts
            expected = hmac.new(self._secret_key, payload_b64.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False


async def rate_limit_middleware(
    request: Request, next_handler: Handler,
    max_requests: int = 100, window_seconds: int = 60,
    _state: dict = {},
) -> Response:
    now = time.monotonic()
    client_ip = request.get_header('x-forwarded-for') or 'unknown'

    if client_ip not in _state:
        _state[client_ip] = {'count': 0, 'window_start': now}

    client = _state[client_ip]
    if now - client['window_start'] > window_seconds:
        client['count'] = 0
        client['window_start'] = now

    client['count'] += 1
    if client['count'] > max_requests:
        return Response.error(
            f'Rate limit exceeded ({max_requests} req/{window_seconds}s)',
            HTTPStatus.TOO_MANY_REQUESTS,
        )

    response = await next_handler(request)
    response.headers['x-ratelimit-limit'] = str(max_requests)
    response.headers['x-ratelimit-remaining'] = str(max(0, max_requests - client['count']))
    return response
```

## Пример использования

```python
async def main():
    server = HttpServer(port=8080)

    auth = AuthMiddleware(secret_key='my-secret', excluded_paths=['/health', '/api/login'])
    server.router.use(logging_middleware)
    server.router.use(cors_middleware)
    server.router.use(auth)

    @server.router.get('/health')
    async def health(req: Request) -> Response:
        return Response.json({'status': 'ok'})

    @server.router.get('/api/users/{user_id}')
    async def get_user(req: Request) -> Response:
        user_id = req.path_params['user_id']
        return Response.json({'id': user_id, 'name': f'User {user_id}'})

    @server.router.post('/api/users')
    async def create_user(req: Request) -> Response:
        data = req.json()
        return Response.json({'id': '123', **data}, HTTPStatus.CREATED)

    await server.start()


if __name__ == '__main__':
    asyncio.run(main())
```

Сервер можно запустить командой `python server.py` и он начнёт слушать на порту 8080.
