"""
title: Morag Agent RAG
description: Агентский RAG с function calling и Knowledge Map
version: 0.1.0
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Coroutine, Dict, Generator, Iterator, List, TypeVar, Union

from markdown_it import MarkdownIt
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

# Импорт из installed morag-пакета (ставится через services/pipeline/Dockerfile).
# Файл специально назван morag_pipeline.py (не morag.py) чтобы избежать коллизии с
# пакетом в sys.modules — OWUI регистрирует файл по filename как имя модуля.
from morag.llm.client import GenerationParams, LLMClient
from morag.indexing.embedder import HttpEmbedder, HttpGteSparseEmbedder
from morag.retrieval import (
    FindSectionConfig,
    HybridSearcher,
    LLMReranker,
    find_section,
)

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
    'Твой ПЕРВЫЙ ход — ВСЕГДА `find_section(query)`, затем `search(query, section_ids=[...])` '
    'с section_ids ИЗ результата find_section. Без исключений, даже если вопрос кажется простым.\n\n'
    'Почему так: find_section работает по doc-level эмбеддингам полного текста каждого документа '
    'и агрегирует результаты по родительскому разделу. Без него search бьёт по всему корпусу — '
    'выдача шумная, из 10+ разных документов. С ним search прицельный и релевантный.\n\n'
    '## Алгоритм работы: Find → Execute → Verify\n\n'
    '### 1. FIND SECTION (обязательный шаг)\n'
    'Вызови `find_section(query)` **минимум два раза** с разными формулировками:\n'
    '  1) первый — по оригинальному вопросу (как задал пользователь);\n'
    '  2) второй — по альтернативным формулировкам / ключевым терминам / смежным понятиям.\n'
    'Например для «Каким термином называется способ X?» — первый find_section по самому вопросу, '
    'второй по «глоссарий термин X», «определение X» или другим формулировкам.\n'
    '⚠️ **ОБЪЕДИНЯЙ** результаты всех find_section — не замещай. '
    'В search передавай union section_ids и doc_ids от всех вызовов.\n'
    'Это нужно потому, что один find_section может пропустить релевантный раздел из-за формулировки. '
    'Второй-третий вызов расширяет охват.\n\n'
    '### 2. ВЫПОЛНЕНИЕ\n'
    '- Ищи тщательно. Старайся покрыть вопрос с разных сторон — делай несколько search\'ей '
    'под РАЗНЫЕ грани (процесс vs инструменты vs ответственные), не повторяя один запрос в переформулировках.\n'
    '- ⚠️ ЯЗЫК ЗАПРОСА: сохраняй ключевые русские слова из исходного вопроса. '
    'Документация на русском — поиск на английском не сработает. '
    'Если в вопросе «доверие к сервису распознавания» — так и пиши в search, не переводи на «trust recognition service». '
    'Синонимы/переформулировки допустимы, но на русском.\n'
    '- `section_ids` — рекурсивный поиск (раздел + все его подстраницы). Для широких тем.\n'
    '- `doc_ids` — точечный поиск (только указанные страницы, БЕЗ потомков). Для случаев когда ответ '
    'прямо на странице-разделе (например, страница «Люди» сама перечисляет отделы — её подстраницы не нужны).\n'
    '- find_section подскажет что использовать: «раздел рекурсивно» → section_ids; «страница точечно» → doc_ids.\n'
    '- Если для разных аспектов релевантны разные секции — дополнительно вызови find_section под аспект.\n'
    '- Используй get_neighbors() чтобы увидеть контекст вокруг найденного чанка.\n'
    '- ⚠️ ШУМ ПРИ ШИРОКОМ ПОИСКЕ: если search вернул результаты из 10+ разных документов — '
    'это сигнал что запрос слишком общий, выдача шумная. Сузь следующий шаг: '
    'переформулируй запрос точнее (более специфичные термины) ИЛИ ограничь section_ids '
    'двумя-тремя самыми релевантными разделами из карты. '
    'Не пытайся «прочитать все 10» — выбери top-2-3 документа и углубляйся через get_neighbors().\n\n'
    '### 3. ПРОВЕРКА ПОЛНОТЫ\n'
    'После поисков проверь:\n'
    '- Найдена ли информация из РАЗНЫХ разделов/документов?\n'
    '- ⚠️ КРАСНЫЙ ФЛАГ: если все результаты из одного раздела — '
    'почти наверняка ты пропустил информацию в других местах. Ищи шире.\n'
    '- Если какая-то грань вопроса не покрыта — ищи в оставшихся разделах.\n'
    '- Делай несколько поисков. Качество важнее скорости.\n\n'
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
        # Qdrant-клиент: единый с indexing, вместо ручных requests.post.
        # Встроенный retry на network-errors + типизированные Prefetch/Filter.
        self._qdrant = AsyncQdrantClient(
            url=self.valves.QDRANT_URL,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        # HybridSearcher — инкапсулирует RRF-поиск и Qdrant-fetch хелперы
        # + ведёт кеши (sparse names, doc tree, titles, knowledge map, cluster).
        self._searcher = HybridSearcher(
            qdrant=self._qdrant,
            dense_embedder=self._dense_embedder,
            sparse_embedder=self._sparse_embedder,
            chunks_collection=self.valves.QDRANT_COLLECTION,
            docs_collection=self.valves.QDRANT_DOCS_COLLECTION,
            knowledge_map_collection=self.valves.QDRANT_KNOWLEDGE_MAP_COLLECTION,
        )
        self._reranker = LLMReranker(self._llm)
        self._find_section_config = FindSectionConfig(
            sections_limit=self.valves.SECTIONS_LIMIT,
            doc_pool=self.valves.FIND_SECTION_DOC_POOL,
            descent_threshold=self.valves.FIND_SECTION_DESCENT_THRESHOLD,
            top_docs=self.valves.FIND_SECTION_TOP_DOCS,
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

    def _tool_find_section(self, query: str) -> tuple[str, list[dict]]:
        """Найти релевантные РАЗДЕЛЫ документации для запроса.

        Вся retrieval-логика (RRF по docs + vote counting + adaptive descent +
        top-K safety + enrichment) живёт в `morag.retrieval.find_section`.
        Pipeline только форматирует SectionResult как markdown для LLM-агента.
        """
        result = self._run(find_section(query, self._searcher, self._find_section_config))
        if result.error == 'no_docs':
            return 'Не удалось найти релевантные документы для определения разделов.', []
        if result.error == 'no_sections' or not (result.refined or result.extra_docs):
            return (
                'Не удалось определить разделы для запроса. '
                'Используй обычный search() без section_ids.'
            ), []

        lines = [f'Релевантные разделы (топ-{len(result.refined)}):']
        for i, e in enumerate(result.refined, 1):
            type_label = 'раздел рекурсивно' if e.kind == 'section' else 'страница точечно'
            lines.append(f'[{i}] {e.title} ({type_label}, id={e.section_id}, {e.votes} dom doc(s))')
            if e.summary:
                snippet = (e.summary[:300] + '…') if len(e.summary) > 300 else e.summary
                lines.append(f'    {snippet}')

        if result.extra_docs:
            lines.append('')
            lines.append('Дополнительно — топ-документы по прямому score (страховка):')
            for i, d in enumerate(result.extra_docs, 1):
                lines.append(f"[T{i}] {d.title} (страница точечно, id={d.doc_id}, score={d.score:.3f})")
                if d.summary:
                    snippet = (d.summary[:300] + '…') if len(d.summary) > 300 else d.summary
                    lines.append(f'    {snippet}')

        lines.append('')
        call_parts = ['search(query="..."']
        if result.section_ids:
            call_parts.append(f'section_ids={json.dumps(result.section_ids, ensure_ascii=False)}')
        if result.doc_ids:
            call_parts.append(f'doc_ids={json.dumps(result.doc_ids, ensure_ascii=False)}')
        lines.append('Готово к использованию: ' + ', '.join(call_parts) + ')')
        return '\n'.join(lines), []

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Тонкая обёртка над LLMReranker (sync↔async через self._run)."""
        return self._run(self._reranker.rerank(query, chunks))

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
        """Streaming финального ответа через LLMClient (AsyncOpenAI).

        SDK сам ретраит connect-errors и 429/5xx через `max_retries`.
        Mid-stream обрыв (после установления соединения) не ретраится — LLM
        generation не идемпотентна. На такой обрыв выдаём graceful-сообщение.

        Sync↔async мост: `pipe()` синхронный (OWUI Pipelines требование),
        async-итератор stream_complete преобразуем пошагово через
        `self._loop.run_until_complete(agen.__anext__())`.
        """
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
        max_tokens = self.valves.LLM_ANSWER_MAX_TOKENS if self.valves.LLM_ANSWER_MAX_TOKENS > 0 else None
        agen = self._llm.stream_complete(
            final_messages,
            params=GenerationParams(temperature=self.valves.LLM_TEMPERATURE),
            max_tokens=max_tokens,
        )
        in_thinking = False
        try:
            while True:
                try:
                    chunk = self._loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                kind = chunk['kind']
                text = chunk['text']
                if kind == 'reasoning':
                    if not in_thinking:
                        yield '<think>'
                        in_thinking = True
                    yield text
                else:  # content
                    if in_thinking:
                        yield '</think>'
                        in_thinking = False
                    yield text
        except Exception as exc:
            logger.warning('stream_final failed: %s', exc, exc_info=True)
            if in_thinking:
                yield '</think>'
                in_thinking = False
            yield f'\n\n⚠️ Связь с LLM сорвалась, ответ может быть неполным. Попробуйте повторить запрос.\n\n_Техническая деталь: {type(exc).__name__}: {exc}_'
            return
        if in_thinking:
            yield '</think>'

    # ── Qdrant + retrieval — тонкие sync-обёртки над HybridSearcher ──────────

    def _search(self, text: str, limit: int) -> list[dict]:
        return self._run(self._searcher.search_chunks(text, limit))

    def _fetch_chunk_by_order(self, doc_id: str, order: int) -> dict | None:
        return self._run(self._searcher.fetch_chunk_by_order(doc_id, order))

    def _fetch_doc_summaries(self, doc_ids: list[str]) -> dict[str, str]:
        return self._run(self._searcher.fetch_doc_summaries(doc_ids))

    def _get_indexed_doc_ids(self) -> set[str]:
        return self._run(self._searcher.get_indexed_doc_ids())

    def _get_descendant_doc_ids(self, section_ids: list[str]) -> set[str]:
        return self._run(self._searcher.get_descendant_doc_ids(section_ids))

    def _fetch_knowledge_map(self) -> str:
        return self._run(self._searcher.fetch_knowledge_map())

    def _fetch_cluster_membership(self) -> dict[str, list[str]]:
        return self._run(self._searcher.fetch_cluster_membership())

    def _get_doc_title(self, doc_id: str) -> str:
        return self._run(self._searcher.get_doc_title(doc_id))

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
        return f'[{query}] поиск раздела'
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
        return f'[{query}] {suffix}'
    if fn_name == 'get_neighbors':
        doc_id = fn_args.get('doc_id', '')
        order = fn_args.get('order', 0)
        window = fn_args.get('window', 2)
        title = _title(doc_id)
        return f'[{title}] блок #{order} (±{window}) — расширение контекста'
    return f'{fn_name}({json.dumps(fn_args, ensure_ascii=False)})'


