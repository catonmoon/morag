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


class Pipeline:
    class Valves(BaseModel):
        QDRANT_URL: str
        QDRANT_COLLECTION: str
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

        HTTP_TIMEOUT: int = 180  # таймаут HTTP-запросов (секунды)

    def __init__(self):
        self.valves = self.Valves(
            QDRANT_URL=os.getenv('QDRANT_URL', 'http://qdrant:6333'),
            QDRANT_COLLECTION=os.getenv('QDRANT_COLLECTION', 'chunks'),
            QDRANT_NUM_RESULTS=int(os.getenv('QDRANT_NUM_RESULTS', '50')),
            NEIGHBOR_WINDOW=int(os.getenv('NEIGHBOR_WINDOW', '1')),

            SPARSE_EMBED_URL=os.getenv('SPARSE_EMBED_URL', 'http://embedder-gte:8081'),
            DENSE_EMBED_URL=os.getenv('DENSE_EMBED_URL', 'http://embedder-frida:8082'),

            LLM_URL=os.getenv('LLM_URL', 'http://localhost:11434/v1'),
            LLM_MODEL=os.getenv('LLM_MODEL', 'qwen2.5:7b'),
            LLM_API_KEY=os.getenv('LLM_API_KEY', 'ollama'),
            LLM_TEMPERATURE=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            LLM_MAX_TOKENS=int(os.getenv('LLM_MAX_TOKENS', '2048')),
            LLM_REPETITION_PENALTY=float(os.getenv('LLM_REPETITION_PENALTY', '1.1')),

            FILTER_MODEL_URL=os.getenv('FILTER_MODEL_URL', os.getenv('LLM_URL', 'http://localhost:11434/v1')),
            FILTER_MODEL=os.getenv('FILTER_MODEL', os.getenv('LLM_MODEL', 'qwen2.5:7b')),
            FILTER_API_KEY=os.getenv('FILTER_API_KEY', os.getenv('LLM_API_KEY', 'ollama')),
            FILTER_MAX_TOKENS=int(os.getenv('FILTER_MAX_TOKENS', '50')),
            FILTER_TEMPERATURE=float(os.getenv('FILTER_TEMPERATURE', '0.0')),

            INTENT_MODEL_URL=os.getenv('INTENT_MODEL_URL', os.getenv('LLM_URL', 'http://localhost:11434/v1')),
            INTENT_MODEL=os.getenv('INTENT_MODEL', os.getenv('LLM_MODEL', 'qwen2.5:7b')),
            INTENT_API_KEY=os.getenv('INTENT_API_KEY', os.getenv('LLM_API_KEY', 'ollama')),
            HTTP_TIMEOUT=int(os.getenv('HTTP_TIMEOUT', '180')),
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

        yield self._emit_status('🔍', f'Фильтрую {len(chunks)} чанков...', False)

        # 5. Reranker: бинарный фильтр по merged-чанкам
        yield '<think>'
        result_chunks: list[dict] = []
        for chunk in chunks:
            answer = self._filter_chunk(' | '.join(intents), chunk)
            if not answer.startswith('0'):
                result_chunks.append(chunk)
                comment = answer.split('|', 1)[1].strip() if '|' in answer else answer.strip()
                doc_name = chunk['path'].split('/')[-1]
                yield f'[{doc_name}]: ✔ {comment}\n'
        yield '</think>'

        result_chunks.sort(key=lambda x: (-_parse_ts(x['updated_at']), x['doc_id'], x['order']))

        if not result_chunks:
            yield self._emit_status('❌', 'Релевантных чанков не найдено', True)
            yield 'Не удалось найти релевантную информацию по вашему запросу.'
            return

        yield self._emit_status('✅', f'Найдено {len(result_chunks)} релевантных чанков', True)

        # Emit citations (один на чанк, source_id=chunk_id чтобы избежать дедупликации по имени файла)
        for chunk in result_chunks:
            yield self._emit_source(
                chunk['path'].split('/')[-1], chunk['text'], chunk.get('url'),
                source_id=chunk['chunk_id'],
            )

        # 5. Стриминг финального ответа
        context = self._build_context(result_chunks)
        yield from self._stream_answer(messages, context)

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
        dialog = '\n'.join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m.get('content', '').strip()}"
            for m in messages if m['role'] in ('user', 'assistant')
        )
        prompt = (
            'Ты агент с базой знаний документации.\n'
            'Прочитай диалог и определи: какие конкретные факты, термины или инструкции тебе не хватает,\n'
            'чтобы дать исчерпывающий ответ пользователю.\n'
            'Сформулируй 1-3 коротких поисковых запроса — каждый покрывает отдельный аспект вопроса.\n'
            'Только ключевые термины, без лишних слов.\n\n'
            f'Диалог:\n{dialog}'
        )
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
        prompt = (
            f'Ты фильтр чанков для ответа на вопрос: "{query}"\n\n'
            f'Основной текст чанка:\n{chunk["text"]}\n\n'
            f'Контекст чанка:\n{chunk["context"]}\n\n'
            f'Путь документа: {chunk["path"]}\n\n'
            'Если чанк содержит информацию, относящуюся к вопросу, верни:\n'
            '1 | <2-4 слова: краткое пояснение>\n\n'
            'Если чанк НЕ содержит релевантной информации, верни только:\n'
            '0\n\n'
            'ВАЖНО: Только указанный формат, ничего лишнего.'
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

    def _stream_answer(self, messages: list, context: str) -> Generator:
        augmented = messages + [{'role': 'user', 'content': context}]
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
    def _build_context(chunks: list[dict]) -> str:
        parts = []
        for n, c in enumerate(chunks, start=1):
            lines = [
                f'Начало чанка [{n}]',
                f'Путь: {c["path"]}',
            ]
            if c.get('url'):
                lines.append(f'URL: {c["url"]}')
            lines += [
                f'Контекст: {c["context"]}',
                f'Текст: {c["text"]}',
                f'Дата актуальности: {c["updated_at"]}',
                f'Конец чанка [{n}]',
            ]
            parts.append('\n'.join(lines))
        instruction = (
            'При использовании информации из чанков вставляй маркер [N] '
            'прямо в текст ответа сразу после утверждения, где N — номер чанка-источника. '
            'Например: "Функция X делает Y [1]." '
            'Если утверждение основано на нескольких чанках — перечисляй: [1][2].'
        )
        return 'Информация из базы знаний:\n\n' + '\n\n'.join(parts) + '\n\n' + instruction

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
    def _emit_source(name: str, content: str, url: str | None = None, source_id: str | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {'source': source_id or name, 'name': name, 'html': False}
        source: dict[str, Any] = {'name': name}
        if url:
            metadata['url'] = url
            source['url'] = url
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
    return {
        'chunk_id': str(p['id']),
        'doc_id': payload.get('doc_id', ''),
        'path': payload.get('path', ''),
        'order': payload.get('order', 0),
        'total': payload.get('total', 0),
        'text': payload.get('text', ''),
        'context': payload.get('context', ''),
        'updated_at': payload.get('updated_at', ''),
        'creator': payload.get('creator', ''),
        'url': payload.get('url'),
        'score': p.get('score', 0.0),
    }
