"""
title: Morag Agent RAG
description: Агентский RAG с function calling и Knowledge Map
version: 0.1.0
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import requests
from typing import Any, Dict, Generator, Iterator, List, Union

from markdown_it import MarkdownIt
import numpy as np
from pydantic import BaseModel

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
                            'Опционально: id разделов из карты документации для прицельного поиска. '
                            'Результаты будут ограничены документами из этих разделов и их подразделов. '
                            'Можно указать несколько. Если не указано — поиск по всей базе.'
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

_SYSTEM_PROMPT_V1 = (
    'Ты — ассистент по внутренней документации компании. '
    'Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
    '## ГЛАВНОЕ ПРАВИЛО\n'
    'ЗАПРЕЩЕНО отвечать без поиска. Твой ПЕРВЫЙ ход — ВСЕГДА вызов search(). '
    'Без исключений, даже если вопрос кажется простым или творческим. '
    'Любой вопрос требует фактов из базы знаний.\n\n'
    'Алгоритм работы:\n'
    '1. Проанализируй вопрос и определи, в каких разделах карты документации искать.\n'
    '2. СРАЗУ делай search() — не рассуждай, не объясняй, не проси уточнений. '
    'Делай ОТДЕЛЬНЫЙ search() для каждого аспекта вопроса. '
    'Если в вопросе несколько частей — ищи каждую отдельно. '
    'Один search() редко находит всё нужное. '
    'Передавай section_ids из карты чтобы сузить поиск. '
    'Первый поиск — по верхнему уровню (##) или без section_ids (вся база). '
    'Можно указать несколько разделов.\n'
    '3. Изучи результаты. Если информации недостаточно — '
    'сделай ещё один поиск с другой формулировкой. '
    'При уточняющих поисках можно сужать до подразделов (###).\n'
    '4. Используй get_neighbors() чтобы увидеть контекст вокруг найденного чанка.\n'
    '5. Делай 3-5 поисков с разных сторон. '
    'Каждый дополнительный поиск повышает точность и полноту ответа. '
    'НЕ пытайся найти всё одним запросом — это работает хуже.\n'
    '6. Когда собрал достаточно информации — дай ответ.\n\n'
)

_SYSTEM_PROMPT = (
    'Ты — ассистент по внутренней документации компании. '
    'Отвечай только на русском языке.\n\n'
    'У тебя есть доступ к базе знаний через инструменты (tools). '
    'Используй их для поиска информации.\n\n'
    '## ГЛАВНОЕ ПРАВИЛО\n'
    'ЗАПРЕЩЕНО отвечать без поиска. Твой ПЕРВЫЙ ход — ВСЕГДА вызов search(). '
    'Без исключений, даже если вопрос кажется простым или творческим. '
    'Любой вопрос требует фактов из базы знаний.\n\n'
    '## Алгоритм работы: Plan → Execute → Verify\n\n'
    '### 1. ПЛАН (перед первым поиском)\n'
    'Проанализируй вопрос и составь план поиска:\n'
    '- Выдели 2-4 СМЫСЛОВЫХ АСПЕКТА вопроса (не переформулировки, а разные грани).\n'
    '- Для каждого аспекта определи раздел(ы) карты документации, где может быть информация.\n'
    '- Аспекты должны покрывать вопрос С РАЗНЫХ СТОРОН.\n'
    'Пример для «Какие роли у менеджера продукта?»:\n'
    '  а) Оргструктура и должности → раздел с дирекциями/командами\n'
    '  б) Обязанности и процессы → раздел с регламентами/процессами\n'
    '  в) Отличия от смежных ролей → раздел с методологией\n\n'
    '### 2. ВЫПОЛНЕНИЕ\n'
    '- Делай search() для КАЖДОГО аспекта из плана, в РАЗНЫХ разделах.\n'
    '- Передавай section_ids из карты чтобы сузить поиск.\n'
    '- Не ищи один аспект 3 раза — ищи 3 разных аспекта.\n'
    '- Используй get_neighbors() чтобы увидеть контекст вокруг найденного чанка.\n\n'
    '### 3. ПРОВЕРКА ПОЛНОТЫ\n'
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
    '- При использовании информации вставляй номер документа-источника '
    'в формате [N], где N — номер документа из результатов search. '
    'Например: "Для настройки Docker нужно установить Docker Desktop [1]." '
    'Если информация из нескольких документов — перечисляй: [1][3].\n'
    '- Структурируй ответ максимально: заголовки, подзаголовки, нумерованные и маркированные списки, '
    'таблицы. Разбивай информацию на логические блоки. Избегай сплошного текста.'
)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_URL: str
        QDRANT_COLLECTION: str
        QDRANT_DOCS_COLLECTION: str
        QDRANT_KNOWLEDGE_MAP_COLLECTION: str

        SPARSE_EMBED_URL: str
        DENSE_EMBED_URL: str

        LLM_URL: str
        LLM_MODEL: str
        LLM_API_KEY: str
        LLM_TEMPERATURE: float
        LLM_MAX_TOKENS: int
        LLM_ANSWER_MAX_TOKENS: int

        SEARCH_LIMIT: int
        MAX_ITERATIONS: int
        ENABLE_THINKING: bool
        CITATION_MAX_CHARS: int
        HTTP_TIMEOUT: int
        PROMPT_VERSION: int  # 1 = v1 (linear), 2 = v2 (plan-execute-verify)

    def __init__(self):
        self.valves = self.Valves(
            QDRANT_URL=os.getenv('QDRANT_URL', 'http://qdrant:6333'),
            QDRANT_COLLECTION=os.getenv('QDRANT_COLLECTION', 'chunks'),
            QDRANT_DOCS_COLLECTION=os.getenv('QDRANT_DOCS_COLLECTION', 'docs'),
            QDRANT_KNOWLEDGE_MAP_COLLECTION=os.getenv(
                'QDRANT_KNOWLEDGE_MAP_COLLECTION', 'knowledge_map',
            ),

            SPARSE_EMBED_URL=os.getenv('SPARSE_EMBED_URL', 'http://embedder-gte:8081'),
            DENSE_EMBED_URL=os.getenv('DENSE_EMBED_URL', 'http://embedder-frida:8082'),

            LLM_URL=os.getenv('LLM_URL', 'http://localhost:11434/v1'),
            LLM_MODEL=os.getenv('LLM_MODEL', 'qwen3.5-9b'),
            LLM_API_KEY=os.getenv('LLM_API_KEY', 'ollama'),
            LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE', '0.3')),
            LLM_MAX_TOKENS=int(os.getenv('LLM_MAX_TOKENS', '4096')),
            LLM_ANSWER_MAX_TOKENS=int(os.getenv('LLM_ANSWER_MAX_TOKENS', '0')),

            SEARCH_LIMIT=int(os.getenv('SEARCH_LIMIT', '50')),
            MAX_ITERATIONS=int(os.getenv('MAX_ITERATIONS', '9')),
            ENABLE_THINKING=os.getenv('ENABLE_THINKING', 'true').lower() == 'true',
            CITATION_MAX_CHARS=int(os.getenv('CITATION_MAX_CHARS', '5000')),
            HTTP_TIMEOUT=int(os.getenv('HTTP_TIMEOUT', '300')),
            PROMPT_VERSION=int(os.getenv('PROMPT_VERSION', '2')),
        )
        self._knowledge_map: str | None = None
        self._doc_titles: dict[str, str] = {}  # doc_id → title (кеш)
        self._doc_tree: dict[str, list[str]] | None = None  # parent_id → [child_ids]

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
        system_content = _SYSTEM_PROMPT if self.valves.PROMPT_VERSION >= 2 else _SYSTEM_PROMPT_V1
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
                    not diversity_nudge_sent
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
                icon = '🔍' if fn_name == 'search' else '📖'
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
            return self._tool_search(args['query'], args.get('limit'), args.get('section_ids'))
        elif name == 'get_neighbors':
            return self._tool_get_neighbors(
                args['doc_id'], args['order'], args.get('window', 2),
            )
        return f'Неизвестный инструмент: {name}', []

    def _tool_search(
        self, query: str, limit: int | None = None, section_ids: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        limit = min(limit or self.valves.SEARCH_LIMIT, self.valves.SEARCH_LIMIT)
        chunks = self._search(query, limit)
        if not chunks:
            return 'Поиск не дал результатов. Попробуй другую формулировку.', []
        # Фильтрация по разделам
        if section_ids:
            allowed_doc_ids = self._get_descendant_doc_ids(section_ids)
            if allowed_doc_ids:
                filtered = [c for c in chunks if c['doc_id'] in allowed_doc_ids]
                if filtered:
                    chunks = filtered

        # LLM reranker — отфильтровать нерелевантные чанки
        reranked = self._rerank(query, chunks)
        if not reranked:
            return 'Поиск дал результаты, но ни один не оказался релевантным. Попробуй другую формулировку.', []
        chunks = reranked

        # Группировка по документу для LLM
        by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c['doc_id'], []).append(c)

        parts = []
        for i, (doc_id, doc_chunks) in enumerate(by_doc.items(), 1):
            doc_chunks.sort(key=lambda x: x['order'])
            path_display = ' | '.join(doc_chunks[0]['path']) if doc_chunks[0]['path'] else doc_id
            doc_name = self._get_doc_title(doc_id)
            lines = [f'[{i}] Документ: {doc_name}', f'Путь: {path_display}']
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

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """LLM reranker: отфильтровать нерелевантные чанки одним вызовом."""
        # Собираем список чанков для оценки
        items = []
        for i, c in enumerate(chunks):
            path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
            context = c.get('context', '')
            lines = [f'[{i}] {path_display}']
            if context:
                lines.append(f'Контекст: {context}')
            lines.append(c['text'])
            items.append('\n'.join(lines))

        prompt = (
            f'Вопрос: "{query}"\n\n'
            f'Чанки:\n' + '\n---\n'.join(items) + '\n\n'
            'Какие из этих чанков могут быть полезны для ответа на вопрос? '
            'Включай чанки даже если они связаны с вопросом косвенно. '
            'При сомнении — включай.\n'
            'Верни ТОЛЬКО номера чанков через запятую. '
            'Например: 0, 3, 5\n'
            'Если ни один не релевантен — верни: none'
        )
        payload = {
            'model': self.valves.LLM_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0,
            'max_tokens': 100,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        try:
            resp = requests.post(
                f'{self.valves.LLM_URL.rstrip("/")}/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.valves.LLM_API_KEY}',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=self.valves.HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json()['choices'][0]['message']['content'].strip()
        except Exception as exc:
            print(f'[morag-agent] rerank failed, returning all chunks: {exc}')
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
        """Вызов LLM с tools, non-streaming. Возвращает полный response."""
        payload = {
            'model': self.valves.LLM_MODEL,
            'messages': messages,
            'tools': _TOOLS,
            'temperature': self.valves.LLM_TEMPERATURE,
            'max_tokens': self.valves.LLM_MAX_TOKENS,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        resp = requests.post(
            f'{self.valves.LLM_URL.rstrip("/")}/chat/completions',
            headers={
                'Authorization': f'Bearer {self.valves.LLM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _stream_final(self, messages: list[dict]) -> Generator:
        """Streaming финального ответа с thinking."""
        # Добавить инструкцию что tools больше нет — отвечай на основе собранного
        final_messages = messages + [{
            'role': 'user',
            'content': (
                'Теперь дай финальный ответ на основе всей собранной информации. '
                'Не вызывай инструменты, отвечай текстом. '
                'ВАЖНО: ответ должен быть коротким — не более 3-5 абзацев. '
                'Не пересказывай всё найденное, выдели только главное.'
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
        payload = {'input': f'search_query: {text}', 'encoding_format': 'base64'}
        resp = requests.post(
            f'{self.valves.DENSE_EMBED_URL}/v1/embeddings',
            json=payload, timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        b64 = resp.json()['data'][0]['embedding']
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).tolist()

    def _embed_sparse(self, text: str) -> tuple[list, list]:
        resp = requests.post(
            f'{self.valves.SPARSE_EMBED_URL}/encode',
            json={'text': text}, timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        token_weights = resp.json()['token_weights'][0]
        return _sparse_dict_to_indices_values(token_weights)

    # ── Qdrant ────────────────────────────────────────────────────────────────

    def _search(self, text: str, limit: int) -> list[dict]:
        dense = self._embed_dense(text)
        indices, values = self._embed_sparse(text)

        # Лексический сигнал: GTE keywords + BM25 stem + BM25 trigram → nested RRF
        lexical_prefetch = [
            {'query': {'indices': indices, 'values': values}, 'using': 'keywords', 'limit': limit * 2},
        ]
        for vec_fn, vec_name in [
            (_bm25_query_vector, 'bm25'),
            (_bm25_trigram_query_vector, 'bm25_trigram'),
        ]:
            idx, val = vec_fn(text)
            if idx:
                lexical_prefetch.append({
                    'query': {'indices': idx, 'values': val},
                    'using': vec_name,
                    'limit': limit * 2,
                })

        # Двухуровневый RRF: семантика (1 голос) vs лексика (1 голос)
        prefetch = [
            {'query': dense, 'using': 'full', 'limit': limit * 2},
            {
                'prefetch': lexical_prefetch,
                'query': {'fusion': 'rrf'},
                'limit': limit * 2,
            },
        ]

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
        """Построить дерево parent→children из коллекции docs (с кешированием)."""
        if self._doc_tree is not None:
            return self._doc_tree
        tree: dict[str, list[str]] = {}
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
                    for parent_id in pl.get('parent_doc_ids', []):
                        tree.setdefault(parent_id, []).append(doc_id)
                offset = result.get('next_page_offset')
                if offset is None:
                    break
            except Exception as exc:
                print(f'[morag-agent] _build_doc_tree failed: {exc}')
                break
        self._doc_tree = tree
        return self._doc_tree

    def _get_descendant_doc_ids(self, section_ids: list[str]) -> set[str]:
        """BFS от section_ids → множество всех потомков (включая сами section_ids)."""
        tree = self._build_doc_tree()
        result: set[str] = set(section_ids)
        queue = list(section_ids)
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
    if fn_name == 'search':
        query = fn_args.get('query', '')
        section_ids = fn_args.get('section_ids')
        if section_ids:
            names = [_title(sid) for sid in section_ids]
            return f'search("{query}" в {", ".join(names)})'
        return f'search("{query}")'
    elif fn_name == 'get_neighbors':
        doc_id = fn_args.get('doc_id', '')
        order = fn_args.get('order', 0)
        window = fn_args.get('window', 2)
        return f'get_neighbors({doc_id}, order={order}, ±{window})'
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


def _sparse_dict_to_indices_values(sparse_dict: dict) -> tuple[list, list]:
    """MD5(word) % 4_294_967_295 → индекс. НЕ МЕНЯТЬ — ломает индекс."""
    index_weight: dict[int, float] = {}
    for word, weight in sparse_dict.items():
        idx = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD
        if idx in index_weight:
            index_weight[idx] = max(index_weight[idx], weight)
        else:
            index_weight[idx] = weight
    return list(index_weight.keys()), list(index_weight.values())


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
