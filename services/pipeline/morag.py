"""
title: Morag RAG
description: Гибридный RAG на локальных документах (Markdown / Confluence)
version: 0.2.0
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterator, List, Union

import numpy as np
from pydantic import BaseModel

_MD5_MOD = 4_294_967_295  # DO NOT CHANGE — ломает индекс

# ── i18n prompts ──────────────────────────────────────────────────────────────

_WEEKDAYS = {
    'ru': ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'],
    'en': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
}

_SYSTEM_PROMPT = {
    'ru': (
        'Ты – RAG система, которая отвечает на вопрос пользователя, '
        'с использованием чанков текста из страниц Confluence в формате markdown.\n'
        'Ниже будет представлена следующая информация, для каждого чанка:\n'
        '1. Путь, который представляет иерархию заголовков страниц откуда был получен чанк.\n'
        '2. Контекст который дает саммари страницы, из которой взят чанк.\n'
        '3. Текст чанка.\n'
        '4. Дата и время актуальности чанка.\n'
        'Используя наиболее релевантные данные, ответь на вопрос пользователя '
        'c **оформлением в markdown**.\n'
        'Чанков может и не быть, в этом случае попроси пользователя уточнить запрос.\n'
        '**Запрещено говорить пользователю о существовании чанков. '
        'Важна только информация которая в них содержится! '
        'Ссылки на источники в формате [N] допустимы.**\n'
        'Если релевантной информации недостаточно, то задай уточняющие вопросы пользователю.\n'
        'Не придумывай и не додумывай, руководствуйся только информацией из чанков!\n'
        'При формировании ответа обращай внимание на дату и время актуальности информации. '
        'Отдавай предпочтение более свежей информации в чанках.\n'
        'В конце выдай пользователю ссылки для самостоятельного уточнения информации '
        '(только на основании чанков, если они есть).\n'
        'Если в ответе есть диаграмма, переделай ее в нотацию mermaid!\n'
        'Важно: Диаграммы и схемы включай в ответ только если об этом попросит пользователь!\n\n'
        'Еще информация, которая может тебе понадобиться (говори только если об этом спрашивают):\n'
        'Тебя создали в Машинном отделении (МО).\n'
        'Твое имя – Мораг, а если спросят почему, то отшучивайся.\n'
        'Текущая дата и время: {current_datetime}\n'
        'Текущий день недели: {current_weekday}\n'
        'Имя текущего пользователя: {user_name}'
    ),
    'en': (
        'You are a RAG system that answers the user\'s question '
        'using text chunks from documents in markdown format.\n'
        'Below you will find the following information for each chunk:\n'
        '1. Path — the hierarchy of document headings the chunk originates from.\n'
        '2. Context — a summary of the document the chunk belongs to.\n'
        '3. The chunk text itself.\n'
        '4. The date and time indicating when the information was last updated.\n'
        'Using the most relevant data, answer the user\'s question '
        'with **markdown formatting**.\n'
        'There may be no chunks at all — in that case ask the user to clarify.\n'
        '**Never mention the existence of chunks to the user. '
        'Only the information they contain matters! '
        'References in [N] format are allowed.**\n'
        'If there is not enough relevant information, ask the user clarifying questions.\n'
        'Do not invent or assume — rely only on information from the chunks!\n'
        'When composing an answer, pay attention to the date and time of the information. '
        'Prefer the most recent information.\n'
        'At the end, provide the user with links for further reading '
        '(only based on chunks, if available).\n'
        'If the answer contains a diagram, convert it to mermaid notation!\n'
        'Important: Include diagrams and schemas only if the user asks for them!\n\n'
        'Additional information you may need (mention only if asked):\n'
        'You were created by the Machine Room team.\n'
        'Your name is Morag — if asked why, make a joke.\n'
        'Current date and time: {current_datetime}\n'
        'Current day of the week: {current_weekday}\n'
        'User name: {user_name}'
    ),
}

_INTENT_PROMPT = {
    'ru': (
        'Ты агент с базой знаний документации.\n'
        'Прочитай диалог и определи: какие конкретные факты, термины или инструкции тебе не хватает,\n'
        'чтобы дать исчерпывающий ответ пользователю.\n'
        'Сформулируй 1-3 коротких поисковых запроса — каждый покрывает отдельный аспект вопроса.\n'
        'Только ключевые термины, без лишних слов.\n\n'
    ),
    'en': (
        'You are an agent with a documentation knowledge base.\n'
        'Read the dialog and determine: what specific facts, terms or instructions you are missing\n'
        'to give a comprehensive answer to the user.\n'
        'Formulate 1-3 short search queries — each covering a separate aspect of the question.\n'
        'Only key terms, no filler words.\n\n'
    ),
}

_FILTER_PROMPT = {
    'ru': (
        'Ты фильтр чанков для ответа на вопрос: "{query}"\n\n'
        'Основной текст чанка:\n{text}\n\n'
        'Контекст чанка:\n{context}\n\n'
        'Путь документа: {path}\n\n'
        'Если чанк содержит информацию, относящуюся к вопросу, верни:\n'
        '1 | <2-4 слова: краткое пояснение>\n\n'
        'Если чанк НЕ содержит релевантной информации, верни только:\n'
        '0\n\n'
        'ВАЖНО: Только указанный формат, ничего лишнего.'
    ),
    'en': (
        'You are a chunk filter for the question: "{query}"\n\n'
        'Chunk text:\n{text}\n\n'
        'Chunk context:\n{context}\n\n'
        'Document path: {path}\n\n'
        'If the chunk contains information relevant to the question, return:\n'
        '1 | <2-4 words: brief explanation>\n\n'
        'If the chunk does NOT contain relevant information, return only:\n'
        '0\n\n'
        'IMPORTANT: Only the specified format, nothing else.'
    ),
}

_CONTEXT_LABELS = {
    'ru': {
        'header': 'Информация из базы знаний:',
        'chunk_start': 'Начало чанка',
        'chunk_end': 'Конец чанка',
        'path': 'Путь',
        'doc_summary': 'Обзор документа',
        'context': 'Контекст',
        'text': 'Текст',
        'updated_at': 'Дата актуальности',
        'citation_instruction': (
            'При использовании информации из чанков вставляй маркер [N] '
            'прямо в текст ответа сразу после утверждения, где N — номер чанка-источника. '
            'Например: "Функция X делает Y [1]." '
            'Если утверждение основано на нескольких чанках — перечисляй: [1][2].'
        ),
        'no_results': 'Не удалось найти релевантную информацию по вашему запросу.',
    },
    'en': {
        'header': 'Information from knowledge base:',
        'chunk_start': 'Chunk start',
        'chunk_end': 'Chunk end',
        'path': 'Path',
        'doc_summary': 'Document overview',
        'context': 'Context',
        'text': 'Text',
        'updated_at': 'Last updated',
        'citation_instruction': (
            'When using information from chunks, insert a [N] marker '
            'right after the statement in your answer, where N is the source chunk number. '
            'Example: "Function X does Y [1]." '
            'If a statement is based on multiple chunks, list them: [1][2].'
        ),
        'no_results': 'Could not find relevant information for your query.',
    },
}

_LOGO = r"""
    ▄▀▀▀▀▀▀▀▀▄
   █  /\_/\   █      Catonmoon
   █ ( =^.^=) █      ╔╦╗ ╔═╗ ┬─┐ ┌─┐ ┌─┐
   █  /> < /  █      ║║║ ║ ║ ├┬┘ ├─┤ │ ┬
    ▀▄▄▄▄▄▄▄▀       ╩ ╩ ╚═╝ ┴└─ ┴ ┴ └─┘
                     pipeline v0.2.0
"""


class Pipeline:
    class Valves(BaseModel):
        QDRANT_URL: str
        QDRANT_COLLECTION: str
        QDRANT_DOCS_COLLECTION: str
        QDRANT_NUM_RESULTS: int
        NEIGHBOR_WINDOW: int

        SPARSE_EMBED_URL: str
        DENSE_EMBED_URL: str

        # Основная LLM (финальный ответ)
        LLM_URL: str
        LLM_MODEL: str
        LLM_API_KEY: str
        LLM_TEMPERATURE: float
        LLM_MAX_TOKENS: int
        LLM_REPETITION_PENALTY: float

        # LLM для reranker (бинарный фильтр)
        FILTER_MODEL_URL: str
        FILTER_MODEL: str
        FILTER_API_KEY: str
        FILTER_MAX_TOKENS: int
        FILTER_TEMPERATURE: float

        # LLM для извлечения intent из диалога
        INTENT_MODEL_URL: str
        INTENT_MODEL: str
        INTENT_API_KEY: str

        LANGUAGE: str = 'ru'  # 'ru' | 'en'

        CITATION_MAX_CHARS: int = 5000  # лимит символов в citation-превью
        HTTP_TIMEOUT: int = 180  # таймаут HTTP-запросов (секунды)
        FILTER_EMIT_THINKING: bool = False  # показывать результаты фильтрации в <think>
        EMIT_STATUS: bool = True   # отправлять status-события в UI
        EMIT_CITATIONS: bool = True  # отправлять citation-события в UI

    def __init__(self):
        print(_LOGO, flush=True)
        self.valves = self.Valves(
            QDRANT_URL=os.getenv('QDRANT_URL', 'http://qdrant:6333'),
            QDRANT_COLLECTION=os.getenv('QDRANT_COLLECTION', 'chunks'),
            QDRANT_DOCS_COLLECTION=os.getenv('QDRANT_DOCS_COLLECTION', 'docs'),
            QDRANT_NUM_RESULTS=int(os.getenv('QDRANT_NUM_RESULTS', '50')),
            NEIGHBOR_WINDOW=int(os.getenv('NEIGHBOR_WINDOW', '1')),

            SPARSE_EMBED_URL=os.getenv('SPARSE_EMBED_URL', 'http://embedder-gte:8081'),
            DENSE_EMBED_URL=os.getenv('DENSE_EMBED_URL', 'http://embedder-frida:8082'),

            LLM_URL=os.getenv('LLM_URL', 'http://localhost:11434/v1'),
            LLM_MODEL=os.getenv('LLM_MODEL', 'qwen2.5:7b'),
            LLM_API_KEY=os.getenv('LLM_API_KEY', 'ollama'),
            LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            LLM_MAX_TOKENS=int(os.getenv('LLM_MAX_TOKENS', '2024')),
            LLM_REPETITION_PENALTY=float(os.getenv('LLM_REPETITION_PENALTY', '1.3')),

            FILTER_MODEL_URL=os.getenv('FILTER_MODEL_URL', os.getenv('LLM_URL', 'http://localhost:11434/v1')),
            FILTER_MODEL=os.getenv('FILTER_MODEL', os.getenv('LLM_MODEL', 'qwen2.5:7b')),
            FILTER_API_KEY=os.getenv('FILTER_API_KEY', os.getenv('LLM_API_KEY', 'ollama')),
            FILTER_MAX_TOKENS=int(os.getenv('FILTER_MAX_TOKENS', '50')),
            FILTER_TEMPERATURE=float(os.getenv('FILTER_TEMPERATURE', '0.0')),

            INTENT_MODEL_URL=os.getenv('INTENT_MODEL_URL', os.getenv('LLM_URL', 'http://localhost:11434/v1')),
            INTENT_MODEL=os.getenv('INTENT_MODEL', os.getenv('LLM_MODEL', 'qwen2.5:7b')),
            INTENT_API_KEY=os.getenv('INTENT_API_KEY', os.getenv('LLM_API_KEY', 'ollama')),
            HTTP_TIMEOUT=int(os.getenv('HTTP_TIMEOUT', '180')),
            LANGUAGE=os.getenv('LANGUAGE', 'ru'),
            FILTER_EMIT_THINKING=os.getenv('FILTER_EMIT_THINKING', 'false').lower() in ('true', '1', 'yes'),
            EMIT_STATUS=os.getenv('EMIT_STATUS', 'true').lower() in ('true', '1', 'yes'),
            EMIT_CITATIONS=os.getenv('EMIT_CITATIONS', 'true').lower() in ('true', '1', 'yes'),
        )

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict],
        body: Dict,
    ) -> Union[str, Generator, Iterator]:
        # 1. Извлечь intent (список поисковых запросов)
        intents = self._extract_intent(messages)
        if self.valves.EMIT_STATUS:
            yield self._emit_status('🔎', ' | '.join(intents), False)

        # 2. Гибридный поиск по всем запросам параллельно
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda q: self._search(q, self.valves.QDRANT_NUM_RESULTS), intents,
            ))
        # Дедупликация по chunk_id, оставляем максимальный score
        seen: dict[str, dict] = {}
        for batch in results:
            for chunk in batch:
                cid = chunk['chunk_id']
                if cid not in seen or chunk['score'] > seen[cid]['score']:
                    seen[cid] = chunk
        chunks = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
        chunks = chunks[:self.valves.QDRANT_NUM_RESULTS]

        # 3. Расширить соседними чанками
        if self.valves.NEIGHBOR_WINDOW > 0 and chunks:
            chunks = self._expand_neighbors(chunks, self.valves.NEIGHBOR_WINDOW)

        # 4. Слить контигуальные группы соседей в один чанк для реранкинга
        chunks = self._merge_into_groups(chunks)

        if self.valves.EMIT_STATUS:
            yield self._emit_status('🔍', f'Фильтрую {len(chunks)} чанков...', False)

        # 5. Reranker: бинарный фильтр по merged-чанкам
        emit_thinking = self.valves.FILTER_EMIT_THINKING
        if emit_thinking:
            yield '<think>'
        result_chunks: list[dict] = []
        for chunk in chunks:
            answer = self._filter_chunk(' | '.join(intents), chunk)
            if not answer.startswith('0'):
                result_chunks.append(chunk)
                if emit_thinking:
                    comment = answer.split('|', 1)[1].strip() if '|' in answer else answer.strip()
                    doc_name = chunk['path'][0].split('/')[-1] if chunk['path'] else chunk['doc_id']
                    yield f'[{doc_name}]: ✔ {comment}\n'
        if emit_thinking:
            yield '</think>'

        result_chunks.sort(key=lambda x: (-_parse_ts(x['updated_at']), x['doc_id'], x['order']))

        lang = self.valves.LANGUAGE
        L = _CONTEXT_LABELS.get(lang, _CONTEXT_LABELS['en'])
        if not result_chunks:
            if self.valves.EMIT_STATUS:
                yield self._emit_status('❌', 'Релевантных чанков не найдено', True)
            yield L['no_results']
            return

        if self.valves.EMIT_STATUS:
            yield self._emit_status('✅', f'Найдено {len(result_chunks)} релевантных чанков', True)

        # Emit citations (один на чанк, source_id=chunk_id чтобы избежать дедупликации по имени файла)
        if self.valves.EMIT_CITATIONS:
            for chunk in result_chunks:
                doc_name = chunk['path'][0].split('/')[-1] if chunk['path'] else chunk['doc_id']
                yield self._emit_source(
                    doc_name, chunk['text'][:self.valves.CITATION_MAX_CHARS], chunk.get('url'),
                    source_id=chunk['chunk_id'],
                    pages=chunk.get('pages'),
                )

        # Достать doc_summary для каждого уникального документа из результатов
        unique_doc_ids = list({c['doc_id'] for c in result_chunks})
        doc_summaries = self._fetch_doc_summaries(unique_doc_ids)

        # 5. Стриминг финального ответа
        context = self._build_context(result_chunks, doc_summaries, lang=lang)
        user_name = ''
        user_info = body.get('__user__', {}) if body else {}
        if isinstance(user_info, dict):
            user_name = user_info.get('name', user_info.get('email', ''))
        yield from self._stream_answer(messages, context, user_name=user_name)

    # ── Intent extraction ─────────────────────────────────────────────────────

    _INTENTS_SCHEMA = {
        'type': 'object',
        'properties': {
            'queries': {
                'type': 'array',
                'items': {'type': 'string'},
                'minItems': 1,
                'maxItems': 3,
            },
        },
        'required': ['queries'],
        'additionalProperties': False,
    }

    def _extract_intent(self, messages: List[dict]) -> list[str]:
        """Сформулировать 1-3 поисковых запроса по истории диалога."""
        lang = self.valves.LANGUAGE
        dialog = '\n'.join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m.get('content', '').strip()}"
            for m in messages if m['role'] in ('user', 'assistant')
        )
        prompt = _INTENT_PROMPT.get(lang, _INTENT_PROMPT['en']) + f'Dialog:\n{dialog}'
        result = self._llm_complete_json(
            self.valves.INTENT_MODEL_URL, self.valves.INTENT_MODEL, self.valves.INTENT_API_KEY,
            [{'role': 'user', 'content': prompt}],
            schema=self._INTENTS_SCHEMA,
            temperature=0.0,
            seed=42,
            max_tokens=150,
        )
        queries = [q.strip() for q in result.get('queries', []) if q.strip()]
        return queries or [messages[-1].get('content', '').strip()]

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _filter_chunk(self, query: str, chunk: dict) -> str:
        lang = self.valves.LANGUAGE
        path_display = ' | '.join(chunk['path']) if chunk['path'] else chunk['doc_id']
        template = _FILTER_PROMPT.get(lang, _FILTER_PROMPT['en'])
        prompt = template.format(
            query=query, text=chunk['text'], context=chunk['context'], path=path_display,
        )
        return self._llm_complete(
            self.valves.FILTER_MODEL_URL, self.valves.FILTER_MODEL, self.valves.FILTER_API_KEY,
            [{'role': 'user', 'content': prompt}],
            temperature=self.valves.FILTER_TEMPERATURE,
            max_tokens=self.valves.FILTER_MAX_TOKENS,
            seed=42,
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _llm_complete_json(
        self, url: str, model: str, api_key: str,
        messages: list, schema: dict,
        temperature: float = 0.0, seed: int | None = None, max_tokens: int | None = None,
    ) -> dict:
        payload: dict = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {'name': 'result', 'schema': schema, 'strict': True},
            },
        }
        if seed is not None:
            payload['seed'] = seed
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return json.loads(resp.json()['choices'][0]['message']['content'])

    def _llm_complete(
        self, url: str, model: str, api_key: str,
        messages: list, temperature: float = 0.1, max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        payload: dict = {'model': model, 'messages': messages, 'temperature': temperature}
        if max_tokens:
            payload['max_tokens'] = max_tokens
        if seed is not None:
            payload['seed'] = seed
        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=self.valves.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    def _stream_answer(self, messages: list, context: str, user_name: str = '') -> Generator:
        lang = self.valves.LANGUAGE
        now = datetime.now(timezone.utc)
        weekdays = _WEEKDAYS.get(lang, _WEEKDAYS['en'])
        unknown = 'неизвестен' if lang == 'ru' else 'unknown'
        prompt_text = _SYSTEM_PROMPT.get(lang, _SYSTEM_PROMPT['en']).format(
            current_datetime=now.strftime('%Y-%m-%d %H:%M:%S UTC'),
            current_weekday=weekdays[now.weekday()],
            user_name=user_name or unknown,
        )
        system_msg = {'role': 'system', 'content': prompt_text}
        augmented = [system_msg] + messages + [{'role': 'user', 'content': context}]
        payload = {
            'model': self.valves.LLM_MODEL,
            'messages': augmented,
            'temperature': self.valves.LLM_TEMPERATURE,
            'max_tokens': self.valves.LLM_MAX_TOKENS,
            'repetition_penalty': self.valves.LLM_REPETITION_PENALTY,
            'stream': True,
        }
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
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                data = json.loads(data_str)
                content = data['choices'][0]['delta'].get('content') or ''
                if content:
                    yield content
            except Exception:
                continue

    @staticmethod
    def _build_context(
        chunks: list[dict], doc_summaries: dict[str, str] | None = None, lang: str = 'ru',
    ) -> str:
        doc_summaries = doc_summaries or {}
        L = _CONTEXT_LABELS.get(lang, _CONTEXT_LABELS['en'])
        parts = []
        for n, c in enumerate(chunks, start=1):
            path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
            lines = [
                f'{L["chunk_start"]} [{n}]',
                f'{L["path"]}: {path_display}',
            ]
            if c.get('url'):
                lines.append(f'URL: {c["url"]}')
            summary = doc_summaries.get(c['doc_id'])
            if summary:
                lines.append(f'{L["doc_summary"]}: {summary}')
            lines += [
                f'{L["context"]}: {c["context"]}',
                f'{L["text"]}: {c["text"]}',
                f'{L["updated_at"]}: {c["updated_at"]}',
                f'{L["chunk_end"]} [{n}]',
            ]
            parts.append('\n'.join(lines))
        return L['header'] + '\n\n' + '\n\n'.join(parts) + '\n\n' + L['citation_instruction']

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _embed_dense(self, text: str) -> list:
        payload = {'input': f'search_query: {text}', 'encoding_format': 'base64'}
        resp = requests.post(f'{self.valves.DENSE_EMBED_URL}/v1/embeddings', json=payload, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        b64 = resp.json()['data'][0]['embedding']
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).tolist()

    def _embed_sparse(self, text: str) -> tuple[list, list]:
        resp = requests.post(f'{self.valves.SPARSE_EMBED_URL}/encode', json={'text': text}, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        token_weights = resp.json()['token_weights'][0]
        return _sparse_dict_to_indices_values(token_weights)

    # ── Qdrant search ─────────────────────────────────────────────────────────

    def _search(self, text: str, limit: int) -> list[dict]:
        dense = self._embed_dense(text)
        indices, values = self._embed_sparse(text)

        payload = {
            'prefetch': [
                {'query': {'indices': indices, 'values': values}, 'using': 'keywords', 'limit': limit * 2},
                {'query': dense, 'using': 'full', 'limit': limit * 2},
            ],
            'query': {'fusion': 'rrf'},
            'limit': limit,
            'with_payload': True,
        }
        url = f'{self.valves.QDRANT_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/query'
        resp = requests.post(url, json=payload, timeout=self.valves.HTTP_TIMEOUT)
        resp.raise_for_status()
        points = resp.json().get('result', {}).get('points', [])
        return [_point_to_chunk(p) for p in points]

    def _expand_neighbors(self, chunks: list[dict], window: int) -> list[dict]:
        """Добавить соседние чанки (±window по order в рамках одного doc_id)."""
        existing_ids: set[str] = {c['chunk_id'] for c in chunks}
        by_doc: dict[str, set[int]] = {}
        for c in chunks:
            by_doc.setdefault(c['doc_id'], set()).add(c['order'])

        extra: list[dict] = []
        for doc_id, orders in by_doc.items():
            for order in list(orders):
                for delta in range(-window, window + 1):
                    if delta == 0:
                        continue
                    neighbor_order = order + delta
                    if neighbor_order < 0 or neighbor_order in orders:
                        continue
                    chunk = self._fetch_chunk_by_order(doc_id, neighbor_order)
                    if chunk and chunk['chunk_id'] not in existing_ids:
                        extra.append(chunk)
                        existing_ids.add(chunk['chunk_id'])
                        orders.add(neighbor_order)

        all_chunks = chunks + extra
        return sorted(all_chunks, key=lambda x: (x['doc_id'], x['order']))

    @staticmethod
    def _merge_into_groups(chunks: list[dict]) -> list[dict]:
        """Слить контигуальные последовательности чанков одного документа в один merged-чанк.

        Чанки уже отсортированы по (doc_id, order) после _expand_neighbors.
        Центральный чанк группы — тот у кого наибольший score (оригинал из RRF);
        соседи имеют score=0.0. Текст объединяется через двойной перенос строки.
        """
        if not chunks:
            return []

        groups: list[list[dict]] = []
        current: list[dict] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = current[-1]
            if chunk['doc_id'] == prev['doc_id'] and chunk['order'] == prev['order'] + 1:
                current.append(chunk)
            else:
                groups.append(current)
                current = [chunk]
        groups.append(current)

        merged: list[dict] = []
        for group in groups:
            central = max(group, key=lambda x: x['score'])
            result = dict(central)
            result['text'] = '\n\n'.join(c['text'] for c in group)
            merged.append(result)
        return merged

    def _fetch_doc_summaries(self, doc_ids: list[str]) -> dict[str, str]:
        """Получить doc_summary из коллекции docs для заданных doc_id.

        Один батч-запрос по полю payload.id (MatchAny).
        Возвращает {doc_id: summary} только для документов у которых есть doc_summary.
        """
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
            print(f'[morag] _fetch_doc_summaries failed, skipping doc summaries: {exc}')
            return {}
        summaries: dict[str, str] = {}
        for point in resp.json().get('result', {}).get('points', []):
            p = point.get('payload', {})
            doc_id = p.get('id')
            summary = p.get('doc_summary')
            if doc_id and summary:
                summaries[doc_id] = summary
        return summaries

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
        p = points[0]
        chunk = _point_to_chunk(p)
        chunk['score'] = 0.0
        return chunk

    # ── Open WebUI events ─────────────────────────────────────────────────────

    @staticmethod
    def _emit_status(emoji: str, text: str, done: bool = False) -> dict[str, Any]:
        return {'event': {'type': 'status', 'data': {'description': f'{emoji} {text}', 'done': done}}}

    @staticmethod
    def _emit_source(
        name: str, content: str, url: str | None = None,
        source_id: str | None = None, pages: list[int] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {'source': source_id or name, 'name': name, 'html': False}
        source: dict[str, Any] = {'name': name}
        if url:
            metadata['url'] = url
            source['url'] = url
        if pages:
            metadata['pages'] = pages
        return {
            'event': {
                'type': 'citation',
                'data': {
                    'document': [content],
                    'metadata': [metadata],
                    'source': source,
                },
            }
        }


# ── Module-level helpers ───────────────────────────────────────────────────────

def _parse_ts(s: str) -> float:
    """ISO-строку → unix timestamp для сортировки. При ошибке возвращает 0.0."""
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0


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
    # path может быть списком (новый формат) или строкой (старые данные)
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
        'pages': payload.get('pages', []),
        'score': p.get('score', 0.0),
    }
