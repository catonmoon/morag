"""
title: Morag Agent RAG
description: Агентский RAG с function calling и Knowledge Map
version: 0.1.0
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import requests
from typing import Any, Coroutine, Dict, Generator, Iterator, List, TypeVar, Union

from markdown_it import MarkdownIt
import numpy as np
from pydantic import BaseModel

# Импорт из installed morag-пакета (ставится через services/pipeline/Dockerfile).
# Файл специально назван morag_pipeline.py (не morag.py) чтобы избежать коллизии с
# пакетом в sys.modules — OWUI регистрирует файл по filename как имя модуля.
from morag.llm.client import GenerationParams, LLMClient
from morag.indexing.embedder import HttpEmbedder, HttpGteSparseEmbedder

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


def _required_env(name: str) -> str:
    """Прочитать обязательную env. RuntimeError если пусто или не задано."""
    value = os.getenv(name, '').strip()
    if not value:
        raise RuntimeError(
            f'Required environment variable {name!r} is not set. '
            f'Configure it in docker-compose.yml or OWUI Admin → Pipelines → Valves.'
        )
    return value

_md = MarkdownIt()

_MD5_MOD = 4_294_967_295  # DO NOT CHANGE — ломает индекс

# ── Tool definitions (OpenAI function calling) ───────────────────────────────

_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search',
            'description': (
                'Поиск по базе знаний документации. '
                'Возвращает релевантные чанки с текстом, контекстом и путём документа.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Поисковый запрос на русском языке. Ключевые термины, без лишних слов.',
                    },
                    'section_ids': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': (
                            'Опционально: id разделов для РЕКУРСИВНОГО поиска — раздел И все его подразделы/страницы. '
                            'Для широких тем, когда ответ может быть в любой подстранице раздела.'
                        ),
                    },
                    'doc_ids': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': (
                            'Опционально: id конкретных страниц для ТОЧЕЧНОГО поиска — только эти страницы, БЕЗ их потомков. '
                            'Для узких запросов, когда известно что ответ на конкретной странице-разделе '
                            '(например, страница «Люди» сама перечисляет отделы, без захода в её подпапки).'
                        ),
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'find_section',
            'description': (
                'ОБЯЗАТЕЛЬНЫЙ ПЕРВЫЙ ШАГ. Найти релевантные РАЗДЕЛЫ документации по запросу. '
                'Работает через doc-level эмбеддинги (полный текст каждого документа) с агрегацией '
                'по родительскому разделу — возвращает готовые section_ids для последующего search. '
                'ВСЕГДА вызывай перед search(). Без этого шага search бьёт по всему корпусу и выдаёт шум.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Поисковый запрос на русском языке.',
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_neighbors',
            'description': (
                'Получить соседние чанки документа для расширения контекста. '
                'Используй когда нашёл релевантный чанк и хочешь увидеть что рядом.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'doc_id': {
                        'type': 'string',
                        'description': 'ID документа (из результатов search)',
                    },
                    'order': {
                        'type': 'integer',
                        'description': 'Порядковый номер чанка в документе (из результатов search)',
                    },
                    'window': {
                        'type': 'integer',
                        'description': 'Количество соседних чанков в каждую сторону (по умолчанию 2)',
                    },
                },
                'required': ['doc_id', 'order'],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    'Ты — ассистент по внутренней документации компании. '
    'Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
    '## ГЛАВНОЕ ПРАВИЛО\n'
    'ЗАПРЕЩЕНО отвечать без поиска. И ЗАПРЕЩЕНО делать search() без предварительного find_section(). '
    'Твой ПЕРВЫЙ ход — ВСЕГДА `find_section(query)`, затем для каждого аспекта из плана — '
    '`search(query, section_ids=[...])` с section_ids ИЗ результата find_section. '
    'Без исключений, даже если вопрос кажется простым.\n\n'
    'Почему так: find_section работает по doc-level эмбеддингам полного текста каждого документа '
    'и агрегирует результаты по родительскому разделу. Без него search бьёт по всему корпусу — '
    'выдача шумная, из 10+ разных документов. С ним search прицельный и релевантный.\n\n'
    '## Алгоритм работы: Plan → Find → Execute → Verify\n\n'
    '### 1. ПЛАН (перед поиском)\n'
    'Проанализируй вопрос и составь план поиска:\n'
    '- Выдели 2-4 СМЫСЛОВЫХ АСПЕКТА вопроса (не переформулировки, а разные грани).\n'
    '- Аспекты должны покрывать вопрос С РАЗНЫХ СТОРОН.\n'
    'Пример для «Какие роли у менеджера продукта?»:\n'
    '  а) Оргструктура и должности\n'
    '  б) Обязанности и процессы\n'
    '  в) Отличия от смежных ролей\n\n'
    '### 2. FIND SECTION (обязательный шаг)\n'
    'Вызови `find_section(query)` один-два раза — для основного запроса и/или ключевых аспектов. '
    'Получишь готовые `section_ids` для последующих search().\n\n'
    '### 3. ВЫПОЛНЕНИЕ\n'
    '- Делай search() для КАЖДОГО аспекта из плана, используя section_ids и/или doc_ids из find_section.\n'
    '- `section_ids` — рекурсивный поиск (раздел + все его подстраницы). Для широких тем.\n'
    '- `doc_ids` — точечный поиск (только указанные страницы, БЕЗ потомков). Для случаев когда ответ '
    'прямо на странице-разделе (например, страница «Люди» сама перечисляет отделы — её подстраницы не нужны).\n'
    '- find_section подскажет что использовать: «раздел рекурсивно» → section_ids; «страница точечно» → doc_ids.\n'
    '- Если для разных аспектов релевантны разные секции — дополнительно вызови find_section под аспект.\n'
    '- Не ищи один аспект 3 раза — ищи 3 разных аспекта.\n'
    '- Используй get_neighbors() чтобы увидеть контекст вокруг найденного чанка.\n'
    '- ⚠️ ШУМ ПРИ ШИРОКОМ ПОИСКЕ: если search вернул результаты из 10+ разных документов — '
    'это сигнал что запрос слишком общий, выдача шумная. Сузь следующий шаг: '
    'переформулируй запрос точнее (более специфичные термины) ИЛИ ограничь section_ids '
    'двумя-тремя самыми релевантными разделами из карты. '
    'Не пытайся «прочитать все 10» — выбери top-2-3 документа и углубляйся через get_neighbors().\n\n'
    '### 4. ПРОВЕРКА ПОЛНОТЫ\n'
    'После поисков проверь:\n'
    '- Все ли аспекты из плана покрыты?\n'
    '- Найдена ли информация из РАЗНЫХ разделов/документов?\n'
    '- ⚠️ КРАСНЫЙ ФЛАГ: если все результаты из одного раздела — '
    'почти наверняка ты пропустил информацию в других местах. Ищи шире.\n'
    '- Если аспект не покрыт — ищи в оставшихся разделах.\n'
    '- Делай 3-6 поисков. Качество важнее скорости.\n\n'
    'Правила ответа:\n'
    '- Отвечай КРАТКО и по существу. Не пересказывай всё найденное — '
    'выбери только то, что прямо отвечает на вопрос.\n'
    '- Отвечай ТОЛЬКО на основе найденной информации из базы знаний. '
    'Не додумывай и не дополняй информацией из общих знаний.\n'
    '- ЗАПРЕЩЕНО делать выводы о политиках, правилах и разрешениях компании, '
    'если они НЕ прописаны явно в найденных документах. '
    'Наличие инструкции (например, «как настроить Mac») '
    'НЕ означает что это разрешено или рекомендовано. '
    'Если политика не описана явно — скажи что информации нет.\n'
    '- Если в базе нет ответа — честно сообщи об этом.\n'
    '- При наличии нескольких источников предпочитай более свежие документы '
    '(ориентируйся на поле «Обновлён» в результатах поиска). '
    'Если старый и новый документ противоречат — доверяй новому.\n'
)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_URL: str
        QDRANT_COLLECTION: str
        QDRANT_DOCS_COLLECTION: str
        QDRANT_KNOWLEDGE_MAP_COLLECTION: str

        SPARSE_EMBED_URL: str
        DENSE_EMBED_URL: str            # OpenAI-compat endpoint, ОБЯЗАТЕЛЬНО с /v1 (для AsyncOpenAI SDK)
        DENSE_EMBEDDER_MODEL: str       # имя модели для /v1/embeddings (Ollama-нотация / HF-имя)
        DENSE_DIM: int                  # размерность вектора (Qwen3-Embedding-4B = 2560)
        QUERY_TEMPLATE: str             # формат входа query-side; {text} → текст запроса

        LLM_URL: str
        LLM_MODEL: str
        LLM_API_KEY: str
        LLM_TEMPERATURE: float
        LLM_MAX_TOKENS: int
        LLM_ANSWER_MAX_TOKENS: int

        SEARCH_LIMIT: int
        UNIQUE_DOCS_CAP: int            # hard cap на число уникальных документов в результате search (0 = без лимита)
        SECTIONS_LIMIT: int             # сколько top-секций возвращает find_section
        FIND_SECTION_DOC_POOL: int      # сколько документов берём из _search_docs для агрегации
        FIND_SECTION_DESCENT_THRESHOLD: float  # 0..1; если ребёнок секции покрывает ≥% votes — спускаемся в него. 0 отключает
        FIND_SECTION_TOP_DOCS: int      # сколько топовых документов из _search_docs добавляем как doc_ids в дополнение к секциям (страхует от потери одинокого чемпиона-секции при vote counting)
        MAX_ITERATIONS: int
        ENABLE_THINKING: bool
        ENABLE_DIVERSITY_NUDGE: bool
        CITATION_MAX_CHARS: int
        HTTP_TIMEOUT: int
        ADMIN_INSTRUCTIONS: str

    def __init__(self):
        self.valves = self.Valves(
            QDRANT_URL=os.getenv('QDRANT_URL', 'http://qdrant:6333'),
            QDRANT_COLLECTION=os.getenv('QDRANT_COLLECTION', 'chunks'),
            QDRANT_DOCS_COLLECTION=os.getenv('QDRANT_DOCS_COLLECTION', 'docs'),
            QDRANT_KNOWLEDGE_MAP_COLLECTION=os.getenv(
                'QDRANT_KNOWLEDGE_MAP_COLLECTION', 'knowledge_map',
            ),

            SPARSE_EMBED_URL=os.getenv('SPARSE_EMBED_URL', 'http://embedder-gte:8081'),
            # ОБЯЗАТЕЛЬНЫЕ env vars — без них pipeline бесполезен. Лучше fail-fast чем
            # дефолты на конкретный хост (host.docker.internal или localhost) — это путает
            # на проде и оставляет позорные значения в логах.
            DENSE_EMBED_URL=_required_env('DENSE_EMBED_URL'),
            DENSE_EMBEDDER_MODEL=_required_env('DENSE_EMBEDDER_MODEL'),
            DENSE_DIM=int(_required_env('DENSE_DIM')),
            # Qwen3 Instruct template — ДОЛЖЕН совпадать с indexing query_template
            # иначе query и docs эмбеддятся разными функциями → потеря качества dense-канала.
            QUERY_TEMPLATE=os.getenv(
                'QUERY_TEMPLATE',
                'Instruct: Given a user question, retrieve passages that answer the question\nQuery:{text}',
            ),

            LLM_URL=_required_env('LLM_URL'),
            LLM_MODEL=_required_env('LLM_MODEL'),
            LLM_API_KEY=_required_env('LLM_API_KEY'),
            LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE', '0.3')),
            LLM_MAX_TOKENS=int(os.getenv('LLM_MAX_TOKENS', '4096')),
            LLM_ANSWER_MAX_TOKENS=int(os.getenv('LLM_ANSWER_MAX_TOKENS', '0')),

            SEARCH_LIMIT=int(os.getenv('SEARCH_LIMIT', '50')),
            UNIQUE_DOCS_CAP=int(os.getenv('UNIQUE_DOCS_CAP', '10')),
            SECTIONS_LIMIT=int(os.getenv('SECTIONS_LIMIT', '5')),
            FIND_SECTION_DOC_POOL=int(os.getenv('FIND_SECTION_DOC_POOL', '20')),
            FIND_SECTION_DESCENT_THRESHOLD=float(os.getenv('FIND_SECTION_DESCENT_THRESHOLD', '0.5')),
            FIND_SECTION_TOP_DOCS=int(os.getenv('FIND_SECTION_TOP_DOCS', '3')),
            MAX_ITERATIONS=int(os.getenv('MAX_ITERATIONS', '9')),
            ENABLE_THINKING=os.getenv('ENABLE_THINKING', 'true').lower() == 'true',
            ENABLE_DIVERSITY_NUDGE=os.getenv('ENABLE_DIVERSITY_NUDGE', 'true').lower() == 'true',
            CITATION_MAX_CHARS=int(os.getenv('CITATION_MAX_CHARS', '5000')),
            HTTP_TIMEOUT=int(os.getenv('HTTP_TIMEOUT', '300')),
            ADMIN_INSTRUCTIONS=os.getenv('ADMIN_INSTRUCTIONS',
                'Если информация не была найдена в конкретном разделе знаний '
                'или её недостаточно для полного ответа, ОБЯЗАТЕЛЬНО сделай '
                'дополнительный поиск без указания раздела (section_ids) — '
                'по всей базе знаний.',
            ),
        )
        self._knowledge_map: str | None = None
        self._doc_titles: dict[str, str] = {}  # doc_id → title (кеш)
        self._doc_tree: dict[str, list[str]] | None = None  # parent_id → [child_ids]
        self._indexed_doc_ids: set[str] | None = None       # id всех документов в docs collection (для фильтра out-of-corpus предков)
        self._cluster_membership: dict[str, list[str]] | None = None  # cluster_id → [doc_id]

        # Persistent event loop для async LLMClient/embedders из morag-пакета.
        # OWUI Pipelines pipe() — sync-генератор, поэтому каждый async-вызов
        # пробрасываем через self._run(coro).
        self._loop = asyncio.new_event_loop()
        # enable_thinking=None — НЕ слать никаких provider-thinking-флагов в extra_body.
        # Причина: _build_extra_body шлёт 4 формата сразу (vLLM chat_template_kwargs,
        # Ollama think/options, OpenRouter reasoning). Некоторые провайдеры (xAI Grok)
        # реджектят неизвестные поля или триггерят `tool_choice='auto'` валидацию vLLM.
        # Для агентского цикла используем non-reasoning модель — server-default ОК.
        # Valve ENABLE_THINKING всё ещё управляет финальным ответом через _stream_final
        # (там raw requests, шлём только chat_template_kwargs — единственный
        # потенциально безопасный флаг для vLLM-серверов).
        self._llm = LLMClient(
            base_url=self.valves.LLM_URL,
            model=self.valves.LLM_MODEL,
            api_key=self.valves.LLM_API_KEY,
            timeout=self.valves.HTTP_TIMEOUT,
            max_retries=3,
            enable_thinking=None,
        )
        # Embeddings: те же async-классы что и в indexing — гарантия консистентности
        # query_template (важно для dense-канала retrieval'а).
        self._dense_embedder = HttpEmbedder(
            base_url=self.valves.DENSE_EMBED_URL,
            model=self.valves.DENSE_EMBEDDER_MODEL,
            dim=self.valves.DENSE_DIM,
            query_template=self.valves.QUERY_TEMPLATE,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        self._sparse_embedder = HttpGteSparseEmbedder(
            base_url=self.valves.SPARSE_EMBED_URL,
            timeout=self.valves.HTTP_TIMEOUT,
        )

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Выполнить async-корутину в нашем persistent event loop. Sync-обёртка для pipe()."""
        return self._loop.run_until_complete(coro)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict],
        body: Dict,
    ) -> Union[str, Generator, Iterator]:
        # 0. Пропустить служебные запросы Open WebUI (title, tags)
        last_content = (messages[-1].get('content', '') if messages else '').strip()
        if last_content.startswith('### Task:'):
            return

        # 1. Подтянуть карту документации
        knowledge_map = self._fetch_knowledge_map()

        # 2. Собрать system prompt
        system_content = _SYSTEM_PROMPT
        if self.valves.ADMIN_INSTRUCTIONS:
            system_content += (
                '\n\n## Обязательные инструкции администратора\n'
                + self.valves.ADMIN_INSTRUCTIONS
            )
        if knowledge_map:
            system_content += (
                '\n\nСтруктура базы знаний (используй для навигации):\n' + knowledge_map
            )

        # 3. Собрать историю для LLM (только user/assistant из Open WebUI)
        agent_messages: list[dict] = [{'role': 'system', 'content': system_content}]
        for m in messages:
            if m['role'] in ('user', 'assistant'):
                content = m.get('content', '').strip()
                if content:
                    agent_messages.append({'role': m['role'], 'content': content})

        # 4. Agent loop
        all_chunks: dict[str, dict] = {}  # chunk_id → chunk (дедупликация)
        tool_call_count = 0
        search_count = 0
        searched_section_ids: set[str] = set()
        diversity_nudge_sent = False

        for iteration in range(self.valves.MAX_ITERATIONS):
            # Вызов LLM с tools
            response = self._llm_call_with_tools(agent_messages)
            message = response['choices'][0]['message']
            finish_reason = response['choices'][0].get('finish_reason', '')

            # Если LLM решил ответить (не вызвал tool)
            if finish_reason != 'tool_calls' or not message.get('tool_calls'):
                # Diversity check: все чанки из ≤1 документа после ≥2 search →
                # инжектим nudge и продолжаем цикл вместо ответа
                unique_docs = {c['doc_id'] for c in all_chunks.values()}
                if (
                    self.valves.ENABLE_DIVERSITY_NUDGE
                    and not diversity_nudge_sent
                    and search_count >= 2
                    and len(unique_docs) <= 1
                    and all_chunks
                ):
                    diversity_nudge_sent = True
                    nudge = self._build_diversity_nudge(
                        searched_section_ids, knowledge_map,
                    )
                    agent_messages.append(message)
                    agent_messages.append({'role': 'user', 'content': nudge})
                    yield self._emit_status(
                        '🔄', 'Расширяю поиск — результаты только из одного документа', False,
                    )
                    continue

                # Emit citations (сгруппированные по документу)
                yield from self._emit_grouped_sources(all_chunks)
                doc_count = len(unique_docs)
                yield self._emit_status(
                    '✅', f'Найдено {_plural(doc_count, "документ", "документа", "документов")} за {_plural(tool_call_count, "шаг", "шага", "шагов")}', True,
                )
                # Stream финального ответа (всегда через _stream_final для thinking)
                agent_messages.append(message)
                yield from self._stream_final(agent_messages)
                return

            # LLM вызвал tools — обработать
            agent_messages.append(message)

            for tool_call in message['tool_calls']:
                tool_call_count += 1
                fn_name = tool_call['function']['name']
                fn_args = json.loads(tool_call['function']['arguments'])
                call_id = tool_call['id']

                if fn_name == 'search':
                    search_count += 1
                    for sid in (fn_args.get('section_ids') or []):
                        searched_section_ids.add(sid)

                # Выполнение + статус
                status_text = _format_tool_status(fn_name, fn_args, resolve_title=self._get_doc_title)
                icon = {'search': '🔍', 'find_section': '🗺️', 'get_neighbors': '📖'}.get(fn_name, '🛠️')
                yield self._emit_status(icon, status_text, False)

                result, chunks = self._execute_tool(fn_name, fn_args)

                # Обновить статус с результатами
                doc_names = list(dict.fromkeys(
                    self._get_doc_title(c['doc_id'])
                    for c in chunks
                ))
                if doc_names:
                    preview = ', '.join(f'"{n}"' for n in doc_names[:2])
                    if len(doc_names) > 2:
                        preview += f' и ещё {len(doc_names) - 2}'
                    yield self._emit_status('→', f'{_plural(len(doc_names), "документ", "документа", "документов")}: {preview}', False)

                # Собрать чанки
                for c in chunks:
                    all_chunks[c['chunk_id']] = c

                # Добавить tool result в историю
                agent_messages.append({
                    'role': 'tool',
                    'tool_call_id': call_id,
                    'content': result,
                })

        # Лимит итераций — принудить ответ без tools
        yield self._emit_status('⚠️', f'Лимит итераций ({self.valves.MAX_ITERATIONS}), генерирую ответ', False)
        yield from self._emit_grouped_sources(all_chunks)
        doc_count = len({c['doc_id'] for c in all_chunks.values()})
        yield self._emit_status('✅', f'Найдено {_plural(doc_count, "документ", "документа", "документов")}', True)
        yield from self._stream_final(agent_messages)

    # ── Diversity nudge ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_top_sections(knowledge_map: str) -> list[tuple[str, str]]:
        """Извлечь разделы верхнего уровня (##) из Knowledge Map.

        Returns list of (section_id, title).
        """
        import re
        sections = []
        for match in re.finditer(r'^##\s+(.+?)\s*\(id:\s*([^)]+)\)', knowledge_map, re.MULTILINE):
            title, section_id = match.group(1).strip(), match.group(2).strip()
            sections.append((section_id, title))
        return sections

    def _build_diversity_nudge(
        self, searched_section_ids: set[str], knowledge_map: str,
    ) -> str:
        """Построить сообщение-nudge для расширения поиска."""
        all_sections = self._parse_top_sections(knowledge_map)
        unsearched = [
            (sid, title) for sid, title in all_sections
            if sid not in searched_section_ids
        ]

        msg = (
            '⚠️ ВСЕ найденные результаты из ОДНОГО документа. '
            'Этого недостаточно для полного ответа.\n\n'
            'Сделай дополнительный search() в ДРУГИХ разделах, '
            'которые ты ещё НЕ проверял. '
            'Попробуй search() БЕЗ section_ids (по всей базе) '
            'или в одном из этих разделов:\n'
        )
        if unsearched:
            for sid, title in unsearched:
                msg += f'- {title} (id: {sid})\n'
        else:
            msg += '(все верхнеуровневые разделы проверены — попробуй поиск без section_ids)\n'

        msg += (
            '\nИщи с ДРУГОЙ формулировкой запроса. '
            'Информация по вопросу может быть в неожиданном месте.'
        )
        return msg

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> tuple[str, list[dict]]:
        """Выполнить tool, вернуть (текстовый результат для LLM, список чанков)."""
        if name == 'search':
            return self._tool_search(
                args['query'], args.get('limit'),
                args.get('section_ids'), args.get('doc_ids'),
            )
        elif name == 'find_section':
            return self._tool_find_section(args['query'])
        elif name == 'get_neighbors':
            return self._tool_get_neighbors(
                args['doc_id'], args['order'], args.get('window', 2),
            )
        return f'Неизвестный инструмент: {name}', []

    def _tool_search(
        self,
        query: str,
        limit: int | None = None,
        section_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        limit = min(limit or self.valves.SEARCH_LIMIT, self.valves.SEARCH_LIMIT)
        chunks = self._search(query, limit)
        if not chunks:
            return 'Поиск не дал результатов. Попробуй другую формулировку.', []
        raw_chunks = chunks  # полный нефильтрованный набор — понадобится для auto-fallback

        # Фильтрация: section_ids → рекурсивно (раздел + потомки); doc_ids → точечно.
        filtered_applied = False
        allowed_doc_ids: set[str] = set()
        if section_ids:
            allowed_doc_ids |= self._get_descendant_doc_ids(section_ids)
        if doc_ids:
            allowed_doc_ids |= set(doc_ids)
        if allowed_doc_ids:
            filtered = [c for c in chunks if c['doc_id'] in allowed_doc_ids]
            if filtered:
                chunks = filtered
                filtered_applied = True

        # LLM reranker — отфильтровать нерелевантные чанки
        reranked = self._rerank(query, chunks)
        if not reranked and filtered_applied:
            # Фильтр оставил чанки, но rerank их все выбросил — возможно классификация
            # документа по теме расходится с тем, где агент искал. Повторяем без фильтра.
            reranked = self._rerank(query, raw_chunks)
        if not reranked:
            return 'Поиск дал результаты, но ни один не оказался релевантным. Попробуй другую формулировку.', []
        chunks = reranked

        # Группировка по документу для LLM — в порядке прихода из reranker
        # (LLM возвращает номера в порядке релевантности). Затем hard cap на
        # число уникальных документов: отсекаем хвост маложелательных.
        by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c['doc_id'], []).append(c)
        cap = self.valves.UNIQUE_DOCS_CAP
        if cap > 0 and len(by_doc) > cap:
            kept_doc_ids = list(by_doc.keys())[:cap]
            by_doc = {did: by_doc[did] for did in kept_doc_ids}
            chunks = [c for c in chunks if c['doc_id'] in by_doc]

        parts = []
        for i, (doc_id, doc_chunks) in enumerate(by_doc.items(), 1):
            doc_chunks.sort(key=lambda x: x['order'])
            path_display = ' | '.join(doc_chunks[0]['path']) if doc_chunks[0]['path'] else doc_id
            doc_name = self._get_doc_title(doc_id)
            lines = [f'[{i}] Документ: {doc_name}', f'Путь: {path_display}']
            updated_at = doc_chunks[0].get('updated_at', '')
            if updated_at:
                lines.append(f'Обновлён: {updated_at}')
            url = doc_chunks[0].get('url')
            if url:
                lines.append(f'URL: {url}')
            lines.append('')
            for c in doc_chunks:
                if c.get('context'):
                    lines.append(f'Контекст: {c["context"]}')
                lines.append(c['text'])
                lines.append('')
            parts.append('\n'.join(lines))

        return f'Найдено {_plural(len(by_doc), "документ", "документа", "документов")}:\n\n' + '\n\n---\n\n'.join(parts), chunks

    def _tool_get_neighbors(
        self, doc_id: str, order: int, window: int = 2,
    ) -> tuple[str, list[dict]]:
        chunks: list[dict] = []
        for delta in range(-window, window + 1):
            target_order = order + delta
            if target_order < 0:
                continue
            chunk = self._fetch_chunk_by_order(doc_id, target_order)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return f'Чанки не найдены для doc_id={doc_id} рядом с order={order}.', []

        chunks.sort(key=lambda x: x['order'])
        parts = []
        for c in chunks:
            marker = ' ← запрошенный' if c['order'] == order else ''
            parts.append(
                f'[order={c["order"]}{marker}]\n{c["text"]}'
            )
        return '\n\n---\n\n'.join(parts), chunks

    # ── Section-level retrieval (find_section) ────────────────────────────────

    def _search_docs(self, text: str, limit: int) -> list[dict]:
        """RRF-поиск по коллекции docs (doc-level эмбеддинги полного текста).

        Возвращает список документов с payload-полями: id, title, path,
        parent_doc_ids, doc_summary, score. Аналог _search но на docs,
        без фильтра по section_ids (секции тут мы как раз и определяем).
        """
        dense = self._embed_dense(text)
        indices, values = self._embed_sparse(text)
        available_sparse = self._get_sparse_vector_names(self.valves.QDRANT_DOCS_COLLECTION)

        lexical_prefetch = []
        if 'keywords' in available_sparse:
            lexical_prefetch.append(
                {'query': {'indices': indices, 'values': values}, 'using': 'keywords', 'limit': limit * 2},
            )
        for vec_fn, vec_name in [
            (_bm25_query_vector, 'bm25'),
            (_bm25_trigram_query_vector, 'bm25_trigram'),
        ]:
            if vec_name not in available_sparse:
                continue
            idx, val = vec_fn(text)
            if idx:
                lexical_prefetch.append({
                    'query': {'indices': idx, 'values': val},
                    'using': vec_name,
                    'limit': limit * 2,
                })

        prefetch = [{'query': dense, 'using': 'full', 'limit': limit * 2}]
        if lexical_prefetch:
            prefetch.append({
                'prefetch': lexical_prefetch,
                'query': {'fusion': 'rrf'},
                'limit': limit * 2,
            })

        payload = {
            'prefetch': prefetch,
            'query': {'fusion': 'rrf'},
            'limit': limit,
            'with_payload': True,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_DOCS_COLLECTION}/points/query'
        resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        points = resp.json().get('result', {}).get('points', [])
        docs: list[dict] = []
        for p in points:
            pl = p.get('payload', {})
            path_raw = pl.get('path', '')
            paths: list[str] = path_raw if isinstance(path_raw, list) else ([path_raw] if path_raw else [])
            docs.append({
                'doc_id': pl.get('id', ''),
                'title': pl.get('title') or pl.get('id', ''),
                'path': paths,
                'parent_doc_ids': pl.get('parent_doc_ids', []) or [],
                'doc_summary': pl.get('doc_summary', ''),
                'score': p.get('score', 0.0),
            })
        return docs

    @staticmethod
    def _aggregate_to_sections(
        docs: list[dict],
        valid_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, list[dict]]]:
        """Vote counting: группируем документы по immediate parent в пределах корпуса.

        parent_doc_ids в Confluence-docs содержит цепочку ВСЕХ предков, включая
        страницы выше настроенных ancestor_ids (out-of-corpus). Такие parent_id
        в нашей коллекции docs отсутствуют — рендерятся как голые id и не имеют
        title/summary. Для каждого документа идём от immediate parent вверх и
        берём первого предка, который реально проиндексирован (`valid_doc_ids`).
        Если ни один предок не в корпусе (документ — сам корень) — он сам
        становится секцией.

        Возвращает [(section_id, [docs]), ...] отсортированный по votes desc,
        затем по score_sum desc. Список docs нужен для adaptive descent.
        """
        from collections import defaultdict
        buckets: dict[str, list[dict]] = defaultdict(list)
        for d in docs:
            parents = d.get('parent_doc_ids') or []
            section_id: str | None = None
            if valid_doc_ids is not None:
                for pid in reversed(parents):
                    if pid in valid_doc_ids:
                        section_id = pid
                        break
            elif parents:
                section_id = parents[-1]
            if section_id is None:
                section_id = d.get('doc_id', '')
            if not section_id:
                continue
            buckets[section_id].append(d)
        items = list(buckets.items())
        items.sort(
            key=lambda kv: (-len(kv[1]), -sum(d.get('score', 0.0) for d in kv[1])),
        )
        return items

    @staticmethod
    def _descend_section(
        section_id: str,
        voting_docs: list[dict],
        tree: dict[str, list[str]],
        threshold: float,
    ) -> tuple[str, bool]:
        """Adaptive descent: спускаемся к ребёнку секции, если тот покрывает ≥threshold% votes.

        Возвращает (final_section_id, self_voted):
        - self_voted=True, если сама страница-секция (doc с id=final_section_id)
          была среди voting_docs — её текст релевантен, агенту имеет смысл
          передать её как `doc_ids` (точечно, без потомков). Типичный случай —
          «Люди» перечисляет отделы прямо на своей странице, в подпапках этой
          информации нет.
        - self_voted=False → секция-контейнер: в voting_docs только её потомки,
          сама страница не содержит ответа. Передавать агенту как `section_ids`
          (рекурсивно — раздел + подстраницы).

        Descent останавливается: при self-vote, при отсутствии детей, или когда
        ни один ребёнок не набрал threshold votes.
        """
        current = section_id
        current_docs = voting_docs
        while True:
            self_voted = any(d.get('doc_id') == current for d in current_docs)
            # Если сама страница-секция попала в voting_docs — её собственный
            # текст релевантен; descent бы вырезал эту страницу из scope
            # (_get_descendant_doc_ids разворачивает только вниз).
            if self_voted:
                return current, True
            if len(current_docs) < 2:  # descent бессмыслен для 1 документа
                return current, False
            children = tree.get(current, [])
            if not children:
                return current, False
            best_child: str | None = None
            best_docs: list[dict] = []
            for child in children:
                matched = [
                    d for d in current_docs
                    if child in (d.get('parent_doc_ids') or [])
                    or d.get('doc_id') == child
                ]
                if len(matched) > len(best_docs):
                    best_child = child
                    best_docs = matched
            if best_child is None or len(best_docs) < threshold * len(current_docs):
                return current, False
            current = best_child
            current_docs = best_docs

    def _tool_find_section(self, query: str) -> tuple[str, list[dict]]:
        """Найти релевантные РАЗДЕЛЫ документации для запроса.

        1. _search_docs(limit=FIND_SECTION_DOC_POOL) — top-N документов.
        2. _aggregate_to_sections — vote counting по immediate parent.
        3. Top SECTIONS_LIMIT секций → enrich title+doc_summary.
        4. Возвращаем готовые section_ids для последующего search().
        """
        pool = self.valves.FIND_SECTION_DOC_POOL
        top = self.valves.SECTIONS_LIMIT
        docs = self._search_docs(query, pool)
        if not docs:
            return 'Не удалось найти релевантные документы для определения разделов.', []

        valid_doc_ids = self._get_indexed_doc_ids()
        aggregated = self._aggregate_to_sections(docs, valid_doc_ids=valid_doc_ids)
        if not aggregated:
            return (
                'Не удалось определить разделы для запроса. '
                'Используй обычный search() без section_ids.'
            ), []

        # Adaptive descent: для каждой секции спускаемся вглубь пока найдётся
        # ребёнок, покрывающий большинство votes (threshold default 0.5).
        # Возвращает также self_voted — была ли сама страница-секция в voting_docs.
        tree = self._build_doc_tree()
        threshold = self.valves.FIND_SECTION_DESCENT_THRESHOLD
        # kind: 'section' (раздел, искать рекурсивно) или 'doc' (сама страница, искать точечно)
        refined: list[tuple[str, int, str]] = []  # (id, votes, kind)
        seen: set[str] = set()
        for sid, section_docs in aggregated:
            if threshold > 0:
                final_sid, self_voted = self._descend_section(sid, section_docs, tree, threshold)
            else:
                final_sid = sid
                self_voted = any(d.get('doc_id') == sid for d in section_docs)
            if final_sid in seen:
                continue
            seen.add(final_sid)
            kind = 'doc' if self_voted else 'section'
            refined.append((final_sid, len(section_docs), kind))
            if len(refined) >= top:
                break

        # Top-K топ-документы из _search_docs — страховка от «одинокого чемпиона»:
        # документ с высоким score может оказаться единственным voter'ом своей
        # секции, и секция проиграет по votes другим бакетам. Явно добавляем
        # top-K документов как doc_ids, если их id не покрыты уже refined.
        top_docs_limit = self.valves.FIND_SECTION_TOP_DOCS
        refined_ids = {sid for sid, _, _ in refined}
        extra_docs: list[dict] = []
        for d in docs:
            if len(extra_docs) >= top_docs_limit:
                break
            did = d['doc_id']
            if not did or did in refined_ids:
                continue
            extra_docs.append(d)
            refined_ids.add(did)

        section_ids = [sid for sid, _, kind in refined if kind == 'section']
        doc_ids = [sid for sid, _, kind in refined if kind == 'doc']
        doc_ids.extend(d['doc_id'] for d in extra_docs)

        summaries = self._fetch_doc_summaries([sid for sid, _, _ in refined] + [d['doc_id'] for d in extra_docs])

        lines = [f'Релевантные разделы (топ-{len(refined)}):']
        for i, (sid, vote, kind) in enumerate(refined, 1):
            title = self._get_doc_title(sid)
            summary = (summaries.get(sid) or '').strip()
            summary_snippet = (summary[:300] + '…') if len(summary) > 300 else summary
            type_label = 'раздел рекурсивно' if kind == 'section' else 'страница точечно'
            lines.append(f'[{i}] {title} ({type_label}, id={sid}, {vote} dom doc(s))')
            if summary_snippet:
                lines.append(f'    {summary_snippet}')

        if extra_docs:
            lines.append('')
            lines.append(f'Дополнительно — топ-документы по прямому score (страховка):')
            for i, d in enumerate(extra_docs, 1):
                title = d.get('title') or d['doc_id']
                summary = (summaries.get(d['doc_id']) or '').strip()
                summary_snippet = (summary[:300] + '…') if len(summary) > 300 else summary
                lines.append(f"[T{i}] {title} (страница точечно, id={d['doc_id']}, score={d.get('score', 0.0):.3f})")
                if summary_snippet:
                    lines.append(f'    {summary_snippet}')

        lines.append('')
        call_parts = ['search(query="..."']
        if section_ids:
            call_parts.append(f'section_ids={json.dumps(section_ids, ensure_ascii=False)}')
        if doc_ids:
            call_parts.append(f'doc_ids={json.dumps(doc_ids, ensure_ascii=False)}')
        lines.append('Готово к использованию: ' + ', '.join(call_parts) + ')')
        return '\n'.join(lines), []

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """LLM reranker: отфильтровать нерелевантные чанки одним вызовом."""
        # Собираем список чанков для оценки
        items = []
        for i, c in enumerate(chunks):
            path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
            context = c.get('context', '')
            updated_at = c.get('updated_at', '')
            lines = [f'[{i}] {path_display}']
            if updated_at:
                lines.append(f'Обновлён: {updated_at}')
            if context:
                lines.append(f'Контекст: {context}')
            lines.append(c['text'])
            items.append('\n'.join(lines))

        prompt = (
            f'Вопрос: "{query}"\n\n'
            f'Чанки:\n' + '\n---\n'.join(items) + '\n\n'
            'Какие из этих чанков могут быть полезны для ответа на вопрос? '
            'Для оценки сымсла предпочитай более свежие чанки (по дате «Обновлён»).\n'
            'Верни ТОЛЬКО номера чанков через запятую, '
            'В ПОРЯДКЕ РЕЛЕВАНТНОСТИ — более полезные первыми. '
            'Например: 3, 0, 5\n'
            'Если ни один не релевантен — верни: none'
        )
        # SDK сам ретраит 429/5xx (max_retries=3 в конструкторе), парсит Retry-After.
        # enable_thinking=False для rerank даже если глобально включён thinking.
        try:
            answer = self._run(self._llm.complete(
                [{'role': 'user', 'content': prompt}],
                params=GenerationParams(temperature=0.0, enable_thinking=False),
                max_tokens=100,
            )).strip()
        except Exception as exc:
            logger.warning('rerank failed, returning all chunks: %s', exc)
            return chunks

        if 'none' in answer.lower():
            return []

        # Парсим номера
        import re
        indices = [int(x) for x in re.findall(r'\d+', answer)]
        filtered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        return filtered or chunks  # fallback: если парсинг сломался, вернуть всё

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _llm_call_with_tools(self, messages: list[dict]) -> dict:
        """LLM call с function-calling tools. Non-streaming, retry/429 в SDK.

        enable_thinking=False всегда (default клиента) — для агентского цикла
        (search/get_neighbors decisions) thinking не нужен.
        """
        return self._run(self._llm.complete_with_tools(
            messages,
            tools=_TOOLS,
            params=GenerationParams(temperature=self.valves.LLM_TEMPERATURE),
            max_tokens=self.valves.LLM_MAX_TOKENS,
        ))

    def _stream_final(self, messages: list[dict]) -> Generator:
        """Streaming финального ответа с thinking.

        ВНИМАНИЕ: единственное место в pipeline где остался ручной requests.post(stream=True)
        с парсингом SSE. Причина — pipe() синхронный (OWUI Pipelines не поддерживает
        async-генераторы), а AsyncOpenAI стриминг → sync iterator потребовал бы моста через
        thread + queue.Queue. Остальные LLM-вызовы (rerank, tool-calls) переведены на
        LLMClient. Если когда-то появится OWUI async support — мигрировать.
        """
        # Добавить инструкцию что tools больше нет — отвечай на основе собранного
        final_messages = messages + [{
            'role': 'user',
            'content': (
                'Теперь дай финальный ответ на основе всей собранной информации. '
                'Не вызывай инструменты, отвечай текстом. '
                'ВАЖНО: ответ должен быть коротким — не более 3-5 абзацев. '
                'Не пересказывай всё найденное, выдели только главное.'
                '- При использовании информации вставляй номер документа-источника '
                'в формате [N], где N — номер документа из результатов search. '
                'Например: "Для настройки Docker нужно установить Docker Desktop [1]." '
                'Если информация из нескольких документов — перечисляй: [1][3].\n'
                '- Структурируй ответ максимально: заголовки, подзаголовки, нумерованные и маркированные списки, '
                'таблицы. Разбивай информацию на логические блоки. Избегай сплошного текста.'
            ),
        }]
        payload = {
            'model': self.valves.LLM_MODEL,
            'messages': final_messages,
            'temperature': self.valves.LLM_TEMPERATURE,
            'stream': True,
        }
        if self.valves.LLM_ANSWER_MAX_TOKENS > 0:
            payload['max_tokens'] = self.valves.LLM_ANSWER_MAX_TOKENS
        if self.valves.ENABLE_THINKING:
            payload['reasoning_budget'] = 4096
        else:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        resp = requests.post(
            f'{self.valves.LLM_URL.rstrip("/")}/chat/completions',
            headers={
                'Authorization': f'Bearer {self.valves.LLM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json=payload,
            stream=True,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        in_thinking = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                data = json.loads(data_str)
                delta = data['choices'][0]['delta']
                # Thinking (reasoning_content или reasoning — зависит от провайдера)
                reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if reasoning:
                    if not in_thinking:
                        yield '<think>'
                        in_thinking = True
                    yield reasoning
                # Content
                content = delta.get('content') or ''
                if content:
                    if in_thinking:
                        yield '</think>'
                        in_thinking = False
                    yield content
            except Exception:
                continue
        if in_thinking:
            yield '</think>'

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _embed_dense(self, text: str) -> list:
        """Dense query embedding через morag.HttpEmbedder (тот же путь что в indexing)."""
        return self._run(self._dense_embedder.embed_query(text))

    def _embed_sparse(self, text: str) -> tuple[list, list]:
        """Sparse query embedding через morag.HttpGteSparseEmbedder."""
        return self._run(self._sparse_embedder.embed_query(text))

    # ── Qdrant ────────────────────────────────────────────────────────────────

    def _get_sparse_vector_names(self, collection: str | None = None) -> set[str]:
        """Получить имена sparse-векторов коллекции (с кешем per-collection)."""
        collection = collection or self.valves.QDRANT_COLLECTION
        if not hasattr(self, '_sparse_vector_names_cache'):
            self._sparse_vector_names_cache: dict[str, set[str]] = {}
        if collection in self._sparse_vector_names_cache:
            return self._sparse_vector_names_cache[collection]
        names: set[str] = set()
        try:
            url = f'{self.valves.QDRANT_URL}/collections/{collection}'
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            sparse = resp.json().get('result', {}).get('config', {}).get(
                'params', {},
            ).get('sparse_vectors', {})
            names = set(sparse.keys())
        except Exception as exc:
            print(f'[morag-agent] failed to get sparse vector names for {collection}: {exc}')
        self._sparse_vector_names_cache[collection] = names
        return names

    def _search(self, text: str, limit: int) -> list[dict]:
        dense = self._embed_dense(text)
        indices, values = self._embed_sparse(text)
        available_sparse = self._get_sparse_vector_names()

        # Лексический сигнал: GTE keywords + BM25 stem + BM25 trigram → nested RRF
        lexical_prefetch = []
        if 'keywords' in available_sparse:
            lexical_prefetch.append(
                {'query': {'indices': indices, 'values': values}, 'using': 'keywords', 'limit': limit * 2},
            )
        for vec_fn, vec_name in [
            (_bm25_query_vector, 'bm25'),
            (_bm25_trigram_query_vector, 'bm25_trigram'),
        ]:
            if vec_name not in available_sparse:
                continue
            idx, val = vec_fn(text)
            if idx:
                lexical_prefetch.append({
                    'query': {'indices': idx, 'values': val},
                    'using': vec_name,
                    'limit': limit * 2,
                })

        # Двухуровневый RRF: семантика (1 голос) vs лексика (1 голос)
        prefetch = [{'query': dense, 'using': 'full', 'limit': limit * 2}]
        if lexical_prefetch:
            prefetch.append({
                'prefetch': lexical_prefetch,
                'query': {'fusion': 'rrf'},
                'limit': limit * 2,
            })

        payload = {
            'prefetch': prefetch,
            'query': {'fusion': 'rrf'},
            'limit': limit,
            'with_payload': True,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/query'
        resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        points = resp.json().get('result', {}).get('points', [])
        return [_point_to_chunk(p) for p in points]

    def _fetch_chunk_by_order(self, doc_id: str, order: int) -> dict | None:
        payload = {
            'filter': {
                'must': [
                    {'key': 'doc_id', 'match': {'value': doc_id}},
                    {'key': 'order', 'match': {'value': order}},
                ]
            },
            'limit': 1,
            'with_payload': True,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/scroll'
        resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        points = resp.json().get('result', {}).get('points', [])
        if not points:
            return None
        chunk = _point_to_chunk(points[0])
        chunk['score'] = 0.0
        return chunk

    def _fetch_doc_summaries(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        payload = {
            'filter': {'must': [{'key': 'id', 'match': {'any': doc_ids}}]},
            'with_payload': ['id', 'doc_summary'],
            'with_vectors': False,
            'limit': len(doc_ids),
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_DOCS_COLLECTION}/points/scroll'
        try:
            resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            print(f'[morag-agent] _fetch_doc_summaries failed: {exc}')
            return {}
        summaries: dict[str, str] = {}
        for point in resp.json().get('result', {}).get('points', []):
            p = point.get('payload', {})
            doc_id = p.get('id')
            summary = p.get('doc_summary')
            if doc_id and summary:
                summaries[doc_id] = summary
        return summaries

    def _build_doc_tree(self) -> dict[str, list[str]]:
        """Построить дерево parent→children + set всех indexed doc_id (с кешированием)."""
        if self._doc_tree is not None:
            return self._doc_tree
        tree: dict[str, list[str]] = {}
        indexed: set[str] = set()
        offset = None
        while True:
            payload: dict = {
                'with_payload': ['id', 'parent_doc_ids'],
                'with_vectors': False,
                'limit': 100,
            }
            if offset is not None:
                payload['offset'] = offset
            url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_DOCS_COLLECTION}/points/scroll'
            try:
                resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
                resp.raise_for_status()
                result = resp.json().get('result', {})
                points = result.get('points', [])
                if not points:
                    break
                for p in points:
                    pl = p.get('payload', {})
                    doc_id = pl.get('id', '')
                    if doc_id:
                        indexed.add(doc_id)
                    for parent_id in pl.get('parent_doc_ids', []):
                        tree.setdefault(parent_id, []).append(doc_id)
                offset = result.get('next_page_offset')
                if offset is None:
                    break
            except Exception as exc:
                print(f'[morag-agent] _build_doc_tree failed: {exc}')
                break
        self._doc_tree = tree
        self._indexed_doc_ids = indexed
        return self._doc_tree

    def _get_indexed_doc_ids(self) -> set[str]:
        """Вернуть id всех документов в docs collection (через _build_doc_tree-кеш)."""
        if self._indexed_doc_ids is None:
            self._build_doc_tree()
        return self._indexed_doc_ids or set()

    def _get_descendant_doc_ids(self, section_ids: list[str]) -> set[str]:
        """Развернуть section_ids в set конкретных doc_id.

        Сначала смотрим в cluster_membership (flat_topics): если id — ключ,
        подставляем список. Остальные id идут по старой BFS-логике через
        дерево parent_doc_ids. Для fixed/weighted membership пустой, ветка
        не активируется.
        """
        membership = self._fetch_cluster_membership()
        result: set[str] = set()
        tree_ids: list[str] = []
        for sid in section_ids:
            if sid in membership:
                result.update(membership[sid])
            else:
                tree_ids.append(sid)
        if tree_ids:
            tree = self._build_doc_tree()
            result.update(tree_ids)
            queue = list(tree_ids)
            while queue:
                parent = queue.pop(0)
                for child in tree.get(parent, []):
                    if child not in result:
                        result.add(child)
                        queue.append(child)
        return result

    def _fetch_knowledge_map(self) -> str:
        if self._knowledge_map is not None:
            return self._knowledge_map
        payload = {
            'filter': {'must': [{'key': 'doc_id', 'match': {'value': '_system_prompt'}}]},
            'with_payload': ['map_text'],
            'with_vectors': False,
            'limit': 1,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_KNOWLEDGE_MAP_COLLECTION}/points/scroll'
        try:
            resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
            resp.raise_for_status()
            points = resp.json().get('result', {}).get('points', [])
            if points:
                self._knowledge_map = points[0]['payload'].get('map_text', '')
            else:
                self._knowledge_map = ''
        except Exception as exc:
            print(f'[morag-agent] _fetch_knowledge_map failed: {exc}')
            self._knowledge_map = ''
        return self._knowledge_map

    def _fetch_cluster_membership(self) -> dict[str, list[str]]:
        """Загрузить cluster_membership из knowledge_map collection (ленивый кеш).

        Возвращает {cluster_id: [doc_id, ...]}. Пустой dict если точки нет
        (например, стратегия fixed/weighted).
        """
        if self._cluster_membership is not None:
            return self._cluster_membership
        payload = {
            'filter': {'must': [{'key': 'doc_id', 'match': {'value': '_cluster_membership'}}]},
            'with_payload': ['cluster_membership'],
            'with_vectors': False,
            'limit': 1,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_KNOWLEDGE_MAP_COLLECTION}/points/scroll'
        try:
            resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
            resp.raise_for_status()
            points = resp.json().get('result', {}).get('points', [])
            if points:
                raw = points[0]['payload'].get('cluster_membership') or {}
                # Санитарная проверка типов
                self._cluster_membership = {
                    k: list(v) for k, v in raw.items()
                    if isinstance(k, str) and isinstance(v, list)
                }
            else:
                self._cluster_membership = {}
        except Exception as exc:
            print(f'[morag-agent] _fetch_cluster_membership failed: {exc}')
            self._cluster_membership = {}
        return self._cluster_membership

    def _get_doc_title(self, doc_id: str) -> str:
        """Получить title документа из Qdrant (с кешем)."""
        if doc_id in self._doc_titles:
            return self._doc_titles[doc_id]
        payload = {
            'filter': {'must': [{'key': 'id', 'match': {'value': doc_id}}]},
            'with_payload': ['title'],
            'with_vectors': False,
            'limit': 1,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_DOCS_COLLECTION}/points/scroll'
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            points = resp.json().get('result', {}).get('points', [])
            if points:
                title = points[0]['payload'].get('title', doc_id)
                self._doc_titles[doc_id] = title
                return title
        except Exception:
            pass
        self._doc_titles[doc_id] = doc_id
        return doc_id

    # ── Citations ─────────────────────────────────────────────────────────────

    def _emit_grouped_sources(self, all_chunks: dict[str, dict]) -> Generator:
        """Сгруппировать чанки по документу, объединить тексты, emit один citation на документ."""
        # Группировка по doc_id
        by_doc: dict[str, list[dict]] = {}
        for chunk in all_chunks.values():
            by_doc.setdefault(chunk['doc_id'], []).append(chunk)

        # Сортировка: внутри документа по order, документы по path
        docs = []
        for doc_id, chunks in by_doc.items():
            chunks.sort(key=lambda c: c['order'])
            path = chunks[0]['path'][0] if chunks[0]['path'] else doc_id
            doc_name = self._get_doc_title(doc_id)
            url = chunks[0].get('url')
            # Объединить тексты чанков (с разделителем), лимит по CITATION_MAX_CHARS
            combined = '\n\n---\n\n'.join(c['text'] for c in chunks)
            if len(combined) > self.valves.CITATION_MAX_CHARS:
                combined = combined[:self.valves.CITATION_MAX_CHARS] + '...'
            docs.append((path, doc_name, url, combined, doc_id))

        docs.sort(key=lambda d: d[0])

        for path, doc_name, url, combined, doc_id in docs:
            yield self._emit_source(doc_name, combined, url, source_id=doc_id)

    # ── Open WebUI events ─────────────────────────────────────────────────────

    @staticmethod
    def _emit_status(emoji: str, text: str, done: bool = False) -> dict[str, Any]:
        return {
            'event': {
                'type': 'status',
                'data': {'description': f'{emoji} {text}', 'done': done},
            }
        }

    @staticmethod
    def _emit_source(
        name: str, content: str, url: str | None = None, source_id: str | None = None,
    ) -> dict[str, Any]:
        html_content = _md.render(content)
        metadata: dict[str, Any] = {'source': source_id or name, 'name': name, 'html': True}
        source: dict[str, Any] = {'name': name}
        if url:
            metadata['url'] = url
            source['url'] = url
        return {
            'event': {
                'type': 'citation',
                'data': {
                    'document': [html_content],
                    'metadata': [metadata],
                    'source': source,
                },
            }
        }


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Склонение существительного по числу (русский язык)."""
    if 11 <= n % 100 <= 19:
        return f'{n} {many}'
    mod = n % 10
    if mod == 1:
        return f'{n} {one}'
    if 2 <= mod <= 4:
        return f'{n} {few}'
    return f'{n} {many}'


def _format_tool_status(fn_name: str, fn_args: dict, resolve_title=None) -> str:
    _title = resolve_title or (lambda x: x)
    if fn_name == 'find_section':
        query = fn_args.get('query', '')
        return f'⌞{query}⌝ поиск раздела'
    if fn_name == 'search':
        query = fn_args.get('query', '')
        section_ids = fn_args.get('section_ids') or []
        doc_ids = fn_args.get('doc_ids') or []
        scope: list[str] = []
        if section_ids:
            scope.append('в разделах: ' + ', '.join(_title(sid) for sid in section_ids))
        if doc_ids:
            scope.append('на страницах: ' + ', '.join(_title(did) for did in doc_ids))
        suffix = '; '.join(scope) if scope else 'по всей базе'
        return f'⌊{query}⌉ {suffix}'
    if fn_name == 'get_neighbors':
        doc_id = fn_args.get('doc_id', '')
        order = fn_args.get('order', 0)
        window = fn_args.get('window', 2)
        title = _title(doc_id)
        return f'【{title}】 блок #{order} (±{window}) — расширение контекста'
    return f'{fn_name}({json.dumps(fn_args, ensure_ascii=False)})'


# ── Module-level helpers ─────────────────────────────────────────────────────

import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

_WORD_RE = re.compile(r'\w+')
_CYRILLIC_RE = re.compile(r'[а-яё]')

_STOP_WORDS: frozenset[str] = frozenset(
    stopwords.words('russian') + stopwords.words('english')
)

_stemmer_ru = SnowballStemmer('russian')
_stemmer_en = SnowballStemmer('english')


def _stem(word: str) -> str:
    """Стемминг с автоопределением языка по кириллице."""
    if _CYRILLIC_RE.search(word):
        return _stemmer_ru.stem(word)
    return _stemmer_en.stem(word)


def _tokens_to_vector(tokens: list[str]) -> tuple[list, list]:
    """Список токенов → (indices, values) sparse vector. Веса = 1.0."""
    if not tokens:
        return [], []
    seen: dict[int, float] = {}
    for token in tokens:
        idx = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % _MD5_MOD
        seen[idx] = 1.0
    return list(seen.keys()), list(seen.values())


def _bm25_query_vector(text: str) -> tuple[list, list]:
    """BM25 query vector: стемминг."""
    words = [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]
    return _tokens_to_vector(words)


# ── Триграммы ─────────────────────────────────────────────────────────────

def _trigrams(word: str) -> list[str]:
    padded = f'__{word}__'
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


def _bm25_trigram_query_vector(text: str) -> tuple[list, list]:
    """BM25 trigram query vector: символьные триграммы оригинальных слов."""
    tokens = []
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP_WORDS:
            continue
        for tri in _trigrams(w):
            tokens.append(tri)
    return _tokens_to_vector(tokens)


def _point_to_chunk(p: dict) -> dict:
    payload = p.get('payload', {})
    path_raw = payload.get('path', '')
    paths: list[str] = path_raw if isinstance(path_raw, list) else ([path_raw] if path_raw else [])
    return {
        'chunk_id': str(p['id']),
        'doc_id': payload.get('doc_id', ''),
        'path': paths,
        'order': payload.get('order', 0),
        'total': payload.get('total', 0),
        'text': payload.get('text', ''),
        'context': payload.get('context', ''),
        'updated_at': payload.get('updated_at', ''),
        'creator': payload.get('creator', ''),
        'url': payload.get('url'),
        'source_type': payload.get('source_type', ''),
        'score': p.get('score', 0.0),
    }
