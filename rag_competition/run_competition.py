#!/usr/bin/env python3
"""
Run competition: answer all questions and produce submission.json.

Usage:
    python scripts/run_competition.py                         # answer all, save submission.json
    python scripts/run_competition.py --questions questions.json  # use cached questions
    python scripts/run_competition.py --submit                # answer + submit to platform
    python scripts/run_competition.py --resume                # skip already answered questions
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

# ── arlc import ───────────────────────────────────────────────────────────────

STARTER_KIT = Path(__file__).resolve().parents[1] / 'rag_competition' / 'starter_kit'
sys.path.insert(0, str(STARTER_KIT))

from arlc import (  # noqa: E402
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('competition')
logging.getLogger('httpx').setLevel(logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

_MD5_MOD = 4_294_967_295  # DO NOT CHANGE

_WORD_RE = re.compile(r'\w+')
# fmt: off
_STOP_WORDS: frozenset[str] = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'not', 'nor',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'over', 'about', 'against', 'upon',
    'it', 'its', 'this', 'that', 'these', 'those',
    'he', 'she', 'they', 'we', 'i', 'you', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their',
    'what', 'which', 'who', 'whom', 'whose',
    'if', 'then', 'else', 'when', 'where', 'how', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'any', 'no',
    'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there',
})
# fmt: on

_CHUNKS_USED_INSTRUCTION = (
    '\nAlso indicate which chunk numbers [N] you used to produce the answer.\n'
    'Include "chunks_used" as an array of integers in your JSON response.\n'
    'Example: {"answer": 42, "chunks_used": [1, 3]}'
)

_ANSWER_TYPE_PROMPTS = {
    'number': (
        'Extract the answer as a single number (integer or decimal).\n'
        'Return ONLY a JSON object: {"answer": <number>, "chunks_used": [N, ...]}\n'
        'If the information cannot be found in the provided context, '
        'return: {"answer": null, "chunks_used": []}'
    ),
    'boolean': (
        'Answer the question with true or false.\n'
        'NEVER return null — ALWAYS return true or false based on the context.\n'
        'If you can see relevant information and there is no match → return false.\n'
        'Absence of evidence = false, NOT unanswerable.\n'
        'IMPORTANT: For questions about whether something was appealed, granted, dismissed, '
        'approved, or succeeded — look for the FINAL OUTCOME (the court order/judgment), '
        'not just the attempt or application. If permission to appeal was GRANTED, '
        'the answer is true. If permission was REFUSED or DISMISSED, the answer is false.\n'
        'For questions about common judges: a judge "presided over" or "was involved in" a case '
        'means they were the presiding/issuing judge of that case. A judge merely mentioned '
        'or referenced in the text of another case does NOT count as involvement.\n'
        'Return ONLY a JSON object: {"answer": true, "chunks_used": [N, ...]}'
    ),
    'name': (
        'Extract the exact name that answers the question.\n'
        'If the question asks "which case" or "which document", return the CASE NUMBER '
        '(e.g. "SCT 295/2025"), NOT party names.\n'
        'For case numbers, use spaces and slashes (e.g. "ENF 316/2023", '
        'NOT "ENF-316-2023"). Strip any "/N" application suffix.\n'
        'For person/company names, use proper title case (e.g. "Fursa Consulting", '
        'NOT "FURSA CONSULTING").\n'
        'For law/regulation names, use the FULL official title as it appears in the document '
        '(e.g. "Real Property Law DIFC Law No. 4 of 2007", NOT just "DIFC Law No. 4 of 2007").\n'
        'For comparison questions ("which case was earlier", "which has higher amount"), '
        'carefully compare the ACTUAL values (dates, amounts) from BOTH cases in the context. '
        'Do NOT default to either the first or last mentioned case.\n'
        'Return ONLY a JSON object: {"answer": "the name", "chunks_used": [N, ...]}\n'
        'If the information cannot be found in the provided context, '
        'return: {"answer": null, "chunks_used": []}'
    ),
    'names': (
        'Extract all names that answer the question as a JSON array.\n'
        'Use proper title case (e.g. ["Fursa Consulting"], NOT ["FURSA CONSULTING"]).\n'
        'Return ONLY a JSON object: {"answer": ["name1", "name2"], "chunks_used": [N, ...]}\n'
        'If the information cannot be found in the provided context, '
        'return: {"answer": null, "chunks_used": []}'
    ),
    'date': (
        'Extract the date that answers the question in ISO 8601 format (YYYY-MM-DD).\n'
        'If both "Date of Issue" and "Date of Re-issue" are present, use the original Date of Issue.\n'
        'Return ONLY a JSON object: {"answer": "YYYY-MM-DD", "chunks_used": [N, ...]}\n'
        'If the information cannot be found in the provided context, '
        'return: {"answer": null, "chunks_used": []}'
    ),
    'free_text': (
        'Answer the question in 1-3 concise sentences (max 280 characters total).\n'
        'Return ONLY a JSON object: {"answer": "your answer text", "chunks_used": [N, ...]}\n'
        'If the information cannot be found in the provided context, '
        'return: {"answer": "There is no information on this question in the provided '
        'documents.", "chunks_used": []}'
    ),
}


# ── Config ────────────────────────────────────────────────────────────────────

def _make_session(retries: int = 10, backoff: float = 2.0) -> requests.Session:
    """Create a requests.Session with automatic retries on transport errors."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET', 'POST'],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class Config:
    """Pipeline config from environment."""

    def __init__(self) -> None:
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.collection_chunks = os.getenv('QDRANT_COLLECTION', 'chunks')
        self.collection_docs = os.getenv('QDRANT_DOCS_COLLECTION', 'docs')
        self.num_results = int(os.getenv('QDRANT_NUM_RESULTS', '100'))
        self.neighbor_window = int(os.getenv('NEIGHBOR_WINDOW', '0'))

        self.sparse_embed_url = os.getenv('SPARSE_EMBED_URL', 'http://localhost:8081')
        self.dense_embed_url = os.getenv('DENSE_EMBED_URL', 'http://localhost:8082')

        self.llm_url = os.getenv('LLM_URL', 'http://localhost:11434/v1')
        self.llm_model = os.getenv('LLM_MODEL', 'qwen2.5:7b')
        self.llm_api_key = os.getenv('LLM_API_KEY', 'ollama')
        self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.llm_max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2048'))

        self.filter_model_url = os.getenv('FILTER_MODEL_URL', self.llm_url)
        self.filter_model = os.getenv('FILTER_MODEL', self.llm_model)
        self.filter_api_key = os.getenv('FILTER_API_KEY', self.llm_api_key)
        self.filter_max_tokens = int(os.getenv('FILTER_MAX_TOKENS', '50'))
        self.filter_temperature = float(os.getenv('FILTER_TEMPERATURE', '0.0'))

        self.intent_model_url = os.getenv('INTENT_MODEL_URL', self.llm_url)
        self.intent_model = os.getenv('INTENT_MODEL', self.llm_model)
        self.intent_api_key = os.getenv('INTENT_API_KEY', self.llm_api_key)

        self.http_timeout = int(os.getenv('HTTP_TIMEOUT', '180'))
        self.session = _make_session()

        # OpenAI SDK clients (handle rate limits properly)
        self.oai_llm = OpenAI(
            base_url=self.llm_url, api_key=self.llm_api_key,
            timeout=self.http_timeout, max_retries=10,
        )
        self.oai_filter = OpenAI(
            base_url=self.filter_model_url, api_key=self.filter_api_key,
            timeout=self.http_timeout, max_retries=10,
        )
        self.oai_intent = OpenAI(
            base_url=self.intent_model_url, api_key=self.intent_api_key,
            timeout=self.http_timeout, max_retries=10,
        )


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_dense(cfg: Config, text: str) -> list[float]:
    payload = {'input': f'search_query: {text}', 'encoding_format': 'base64'}
    resp = cfg.session.post(
        f'{cfg.dense_embed_url}/v1/embeddings', json=payload, timeout=cfg.http_timeout,
    )
    resp.raise_for_status()
    b64 = resp.json()['data'][0]['embedding']
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).tolist()


def embed_sparse(cfg: Config, text: str) -> tuple[list[int], list[float]]:
    resp = cfg.session.post(
        f'{cfg.sparse_embed_url}/encode', json={'text': text}, timeout=cfg.http_timeout,
    )
    resp.raise_for_status()
    token_weights: dict[str, float] = resp.json()['token_weights'][0]
    index_weight: dict[int, float] = {}
    for word, weight in token_weights.items():
        idx = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD
        if idx in index_weight:
            index_weight[idx] = max(index_weight[idx], weight)
        else:
            index_weight[idx] = weight
    return list(index_weight.keys()), list(index_weight.values())


def embed_bm25_query(text: str) -> tuple[list[int], list[float]]:
    """BM25 query vector: токенизация + хеширование, вес = 1.0 для каждого терма."""
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]
    index_weight: dict[int, float] = {}
    for word in words:
        idx = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % _MD5_MOD
        index_weight[idx] = 1.0
    return list(index_weight.keys()), list(index_weight.values())


# ── Qdrant search ─────────────────────────────────────────────────────────────

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
        'url': payload.get('url'),
        'source_type': payload.get('source_type', ''),
        'pages': payload.get('pages', []),
        'score': p.get('score', 0.0),
    }


def search(cfg: Config, text: str, limit: int) -> list[dict]:
    dense = embed_dense(cfg, text)
    indices, values = embed_sparse(cfg, text)
    bm25_indices, bm25_values = embed_bm25_query(text)
    payload = {
        'prefetch': [
            {'query': {'indices': indices, 'values': values}, 'using': 'keywords', 'limit': limit * 2},
            {'query': dense, 'using': 'full', 'limit': limit * 2},
            {'query': {'indices': bm25_indices, 'values': bm25_values}, 'using': 'bm25', 'limit': limit * 2},
        ],
        'query': {'fusion': 'rrf'},
        'limit': limit,
        'with_payload': True,
    }
    url = f'{cfg.qdrant_url}/collections/{cfg.collection_chunks}/points/query'
    resp = cfg.session.post(url, json=payload, timeout=cfg.http_timeout)
    resp.raise_for_status()
    points = resp.json().get('result', {}).get('points', [])
    return [_point_to_chunk(p) for p in points]


def fetch_chunk_by_order(cfg: Config, doc_id: str, order: int) -> dict | None:
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
    url = f'{cfg.qdrant_url}/collections/{cfg.collection_chunks}/points/scroll'
    resp = cfg.session.post(url, json=payload, timeout=cfg.http_timeout)
    resp.raise_for_status()
    points = resp.json().get('result', {}).get('points', [])
    if not points:
        return None
    chunk = _point_to_chunk(points[0])
    chunk['score'] = 0.0
    return chunk


def expand_neighbors(cfg: Config, chunks: list[dict], window: int) -> list[dict]:
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
                chunk = fetch_chunk_by_order(cfg, doc_id, neighbor_order)
                if chunk and chunk['chunk_id'] not in existing_ids:
                    extra.append(chunk)
                    existing_ids.add(chunk['chunk_id'])
                    orders.add(neighbor_order)
    return sorted(chunks + extra, key=lambda x: (x['doc_id'], x['order']))


def merge_into_groups(chunks: list[dict]) -> list[dict]:
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


def fetch_doc_summaries(cfg: Config, doc_ids: list[str]) -> dict[str, str]:
    if not doc_ids:
        return {}
    payload = {
        'filter': {'must': [{'key': 'id', 'match': {'any': doc_ids}}]},
        'with_payload': ['id', 'doc_summary'],
        'with_vectors': False,
        'limit': len(doc_ids),
    }
    url = f'{cfg.qdrant_url}/collections/{cfg.collection_docs}/points/scroll'
    try:
        resp = cfg.session.post(url, json=payload, timeout=cfg.http_timeout)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning('fetch_doc_summaries failed: %s', exc)
        return {}
    summaries: dict[str, str] = {}
    for point in resp.json().get('result', {}).get('points', []):
        p = point.get('payload', {})
        doc_id = p.get('id')
        summary = p.get('doc_summary')
        if doc_id and summary:
            summaries[doc_id] = summary
    return summaries


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model that doesn't support temperature/seed."""
    return any(tag in model for tag in ('gpt-5', 'o1', 'o3', 'o4'))


def _oai_client(cfg: Config, url: str) -> OpenAI:
    """Pick the right OpenAI SDK client by URL."""
    if url == cfg.filter_model_url:
        return cfg.oai_filter
    if url == cfg.intent_model_url:
        return cfg.oai_intent
    return cfg.oai_llm


def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


def llm_complete(
    cfg: Config, url: str, model: str, api_key: str,
    messages: list[dict], temperature: float = 0.0,
    max_tokens: int | None = None, seed: int | None = None,
    enable_thinking: bool = False,
) -> str:
    client = _oai_client(cfg, url)
    kwargs: dict = {'model': model, 'messages': messages, 'temperature': temperature}
    if max_tokens:
        kwargs['max_tokens'] = max_tokens
    if seed is not None:
        kwargs['seed'] = seed
    if 'openrouter' in url:
        kwargs['extra_body'] = {'reasoning': {'enabled': enable_thinking}}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ''
    return _strip_think(content)


def llm_complete_json(
    cfg: Config, url: str, model: str, api_key: str,
    messages: list[dict], schema: dict,
    temperature: float = 0.0, seed: int | None = None, max_tokens: int | None = None,
) -> dict:
    client = _oai_client(cfg, url)
    if 'openai.com' in url:
        response_format = {
            'type': 'json_schema',
            'json_schema': {'name': 'result', 'schema': schema, 'strict': True},
        }
    else:
        response_format = {'type': 'json_object'}
    kwargs: dict = {
        'model': model, 'messages': messages, 'temperature': temperature,
        'response_format': response_format,
    }
    if seed is not None:
        kwargs['seed'] = seed
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if 'openrouter' in url:
        kwargs['extra_body'] = {'reasoning': {'enabled': False}}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ''
    content = _strip_think(content)
    return json.loads(content)


# ── Intent extraction ─────────────────────────────────────────────────────────

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


def extract_intent(cfg: Config, question: str) -> list[str]:
    prompt = (
        'You are an agent with a documentation knowledge base.\n'
        'Given a question, formulate 1-3 short search queries — '
        'each covering a separate aspect. Only key terms, no filler words.\n\n'
        f'Question:\n{question}'
    )
    result = llm_complete_json(
        cfg, cfg.intent_model_url, cfg.intent_model, cfg.intent_api_key,
        [{'role': 'user', 'content': prompt}],
        schema=_INTENTS_SCHEMA,
        temperature=0.0, seed=42, max_tokens=150,
    )
    queries = [q.strip() for q in result.get('queries', []) if q.strip()]
    return queries or [question]


# ── Reranker ──────────────────────────────────────────────────────────────────

def select_relevant_chunks(
    cfg: Config, query: str, chunks: list[dict],
    doc_summaries: dict[str, str] | None = None,
) -> list[dict]:
    """Select relevant chunks in a single LLM call by showing a summary list."""
    doc_summaries = doc_summaries or {}

    # Build summary list — show doc header only once per document
    lines = []
    shown_docs: set[str] = set()
    for i, chunk in enumerate(chunks, start=1):
        parts = []
        doc_id = chunk['doc_id']
        if doc_id not in shown_docs:
            path = ' | '.join(chunk['path']) if chunk['path'] else doc_id[:30]
            parts.append(f'--- Document: {path} ---')
            summary = doc_summaries.get(doc_id, '')
            if summary:
                parts.append(f'    {summary}')
            shown_docs.add(doc_id)
        context = chunk.get('context', '')
        text = chunk['text'][:50000].replace('\n', ' ')
        parts.append(f'{i}.')
        if context:
            parts.append(f'   Context: {context}')
        parts.append(f'   Text: "{text}"')
        lines.append('\n'.join(parts))

    prompt = (
        f'Question: "{query}"\n\n'
        f'Below are {len(chunks)} text chunks retrieved from a document corpus.\n'
        'Each shows: number, document info, chunk context, and text.\n\n'
        + '\n'.join(lines) + '\n\n'
        'Which chunks contain information USEFUL for answering the question?\n'
        'Select chunks from the relevant document(s) — even if a chunk alone does not answer\n'
        'the question, include it if it provides context needed to reason about the answer.\n'
        'IMPORTANT: Prioritize chunks containing COURT ORDERS, JUDGMENTS, and FINAL DECISIONS\n'
        '(e.g. "IT IS HEREBY ORDERED", "Permission to Appeal is refused", "Claim is dismissed").\n'
        'These are more important than background details or schedule of reasons.\n'
        'For comparison questions, include chunks from ALL mentioned entities/cases/documents.\n'
        'Do NOT select chunks from completely unrelated documents.\n\n'
        'Return ONLY the chunk numbers, comma-separated. Example: 3,5,7\n'
        'If none are relevant, return: none'
    )
    response = llm_complete(
        cfg, cfg.filter_model_url, cfg.filter_model, cfg.filter_api_key,
        [{'role': 'user', 'content': prompt}],
        temperature=0.0, max_tokens=100, seed=42,
    )
    logger.info('  Select response: %s', response.strip())

    # Parse response
    response = response.strip().lower()
    if response == 'none':
        return []
    numbers: set[int] = set()
    for token in re.split(r'[,\s]+', response):
        token = token.strip().rstrip('.')
        if token.isdigit():
            numbers.add(int(token))
    return [chunks[n - 1] for n in sorted(numbers) if 1 <= n <= len(chunks)]


# ── Context building ─────────────────────────────────────────────────────────

def build_context(chunks: list[dict], doc_summaries: dict[str, str] | None = None) -> str:
    parts = []
    doc_summaries = doc_summaries or {}
    prev_doc_id = None
    for n, c in enumerate(chunks, start=1):
        if c['doc_id'] != prev_doc_id:
            path_display = ' | '.join(c['path']) if c['path'] else c['doc_id']
            parts.append(f'=== Document: {path_display} ===')
            summary = doc_summaries.get(c['doc_id'], '')
            if summary:
                parts.append(summary)
            prev_doc_id = c['doc_id']
        lines = [
            f'Chunk [{n}]',
            f'Context: {c["context"]}',
            c['text'],
        ]
        parts.append('\n'.join(lines))
    return 'Information from knowledge base:\n\n' + '\n\n'.join(parts)


# ── Answer generation ─────────────────────────────────────────────────────────

_ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {
        'answer': {},  # any type
        'chunks_used': {
            'type': 'array',
            'items': {'type': 'integer'},
        },
    },
    'required': ['answer', 'chunks_used'],
    'additionalProperties': False,
}


def generate_answer(
    cfg: Config, question: str, answer_type: str, context: str,
) -> tuple[object, list[int], TimingMetrics, UsageMetrics]:
    """Generate a typed answer. Returns (answer_value, chunks_used, timing, usage)."""
    type_instruction = _ANSWER_TYPE_PROMPTS.get(answer_type, _ANSWER_TYPE_PROMPTS['free_text'])
    system = (
        'You are a precise legal document QA system.\n'
        'Answer the question using ONLY the provided context.\n'
        'Do not invent or assume — rely only on the context.\n\n'
        f'{type_instruction}'
    )
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': f'{context}\n\nQuestion: {question}'},
    ]

    timer = TelemetryTimer()
    # Use reasoning model for boolean/name/names (multi-doc questions benefit)
    use_reasoning = answer_type in ('boolean', 'name', 'names')
    if use_reasoning:
        client = cfg.oai_filter  # reasoning model
        model = cfg.filter_model
    else:
        client = cfg.oai_llm
        model = cfg.llm_model
    extra = {}
    if 'openrouter' in cfg.llm_url and not use_reasoning:
        extra['extra_body'] = {'reasoning': {'enabled': False}}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=cfg.llm_max_tokens,
        seed=42,
        stream=True,
        stream_options={'include_usage': True},
        **extra,
    )

    content_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    for chunk in stream:
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
        if chunk.choices:
            delta = chunk.choices[0].delta
            token = delta.content or ''
            if token:
                timer.mark_token()
                content_parts.append(token)

    timing = timer.finish()
    usage_metrics = UsageMetrics(input_tokens=input_tokens, output_tokens=output_tokens)

    raw = ''.join(content_parts).strip()
    raw = _strip_think(raw)
    answer, chunks_used = parse_answer(raw, answer_type)
    return answer, chunks_used, timing, usage_metrics


def parse_answer(raw: str, answer_type: str) -> tuple[object, list[int]]:
    """Parse LLM response into the correct type + chunks_used list."""

    def _extract_chunks_used(obj: dict) -> list[int]:
        cu = obj.get('chunks_used', [])
        if isinstance(cu, list):
            return [int(x) for x in cu if isinstance(x, (int, float))]
        return []

    # Try JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and 'answer' in obj:
            return _coerce(obj['answer'], answer_type), _extract_chunks_used(obj)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and 'answer' in obj:
                return _coerce(obj['answer'], answer_type), _extract_chunks_used(obj)
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract value from raw text
    return _coerce_raw(raw, answer_type), []


def _coerce(value: object, answer_type: str) -> object:
    """Coerce a parsed JSON value to the expected type."""
    if value is None:
        return None
    if answer_type == 'number':
        if isinstance(value, (int, float)):
            return value
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None
    if answer_type == 'boolean':
        if isinstance(value, bool):
            return value
        s = str(value).lower().strip()
        if s in ('true', 'yes', '1'):
            return True
        if s in ('false', 'no', '0'):
            return False
        return None
    if answer_type == 'date':
        s = str(value).strip()
        if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
            return s
        return None
    if answer_type == 'name':
        return str(value).strip() if value else None
    if answer_type == 'names':
        if isinstance(value, list):
            seen: set[str] = set()
            deduped: list[str] = []
            for v in value:
                s = str(v).strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    deduped.append(s)
            return deduped or None
        return None
    # free_text
    return str(value).strip()


def _coerce_raw(raw: str, answer_type: str) -> object:
    """Last-resort extraction from unstructured text."""
    text = raw.strip()
    if answer_type == 'number':
        m = re.search(r'-?[\d,]+\.?\d*', text)
        if m:
            try:
                return float(m.group().replace(',', ''))
            except ValueError:
                pass
        return None
    if answer_type == 'boolean':
        low = text.lower()
        if 'true' in low or 'yes' in low:
            return True
        if 'false' in low or 'no' in low:
            return False
        return None
    if answer_type == 'date':
        m = re.search(r'\d{4}-\d{2}-\d{2}', text)
        return m.group() if m else None
    if answer_type == 'name':
        return text.strip('"\'') if text else None
    if answer_type == 'names':
        # try comma-separated
        parts = [p.strip().strip('"\'') for p in text.split(',')]
        return [p for p in parts if p] or None
    # free_text
    return text[:280]


# ── Retrieval pages for telemetry ─────────────────────────────────────────────

def collect_retrieval_refs(chunks: list[dict]) -> list[RetrievalRef]:
    """Build retrieval refs from filtered chunks for submission telemetry."""
    by_doc: dict[str, set[int]] = {}
    for c in chunks:
        doc_id = c['doc_id']
        # doc_id in Qdrant is "sha.md", submission needs "sha"
        if doc_id.endswith('.md'):
            doc_id = doc_id[:-3]
        pages = c.get('pages', [])
        if pages:
            by_doc.setdefault(doc_id, set()).update(pages)
    return [
        RetrievalRef(doc_id=did, page_numbers=sorted(pages))
        for did, pages in sorted(by_doc.items())
    ]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def answer_question(cfg: Config, question: str, answer_type: str) -> SubmissionAnswer | None:
    """Full RAG pipeline for a single question."""
    q_id = hashlib.sha256(question.encode()).hexdigest()

    # 1. Intent extraction (crash on failure — fix and resume)
    intents = extract_intent(cfg, question)
    logger.info('  Intents: %s', intents)

    # 2. Hybrid search (parallel across intents)
    #    Multi-doc questions (2+ case numbers) get more chunks for better coverage
    case_re = re.compile(r'(?:CFI|SCT|CA|ARB|ENF|DEC|TCD)\s+\d+/\d+')
    mentioned_cases = set(case_re.findall(question))
    is_multidoc = len(mentioned_cases) >= 2
    if is_multidoc:
        num_results = 200
    elif mentioned_cases:
        num_results = 150
    else:
        num_results = 150

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda q: search(cfg, q, num_results), intents,
        ))
    seen: dict[str, dict] = {}
    for batch in results:
        for chunk in batch:
            cid = chunk['chunk_id']
            if cid not in seen or chunk['score'] > seen[cid]['score']:
                seen[cid] = chunk
    chunks = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
    chunks = chunks[:num_results]
    logger.info('  Search: %d chunks', len(chunks))

    # 3. Sort: group by doc (best RRF score first), within doc by order
    doc_best_score: dict[str, float] = {}
    for c in chunks:
        doc_best_score[c['doc_id']] = max(doc_best_score.get(c['doc_id'], 0.0), c['score'])
    chunks = sorted(chunks, key=lambda x: (-doc_best_score[x['doc_id']], x['order']))

    # 4. Fetch doc summaries + select relevant chunks
    unique_doc_ids = list({c['doc_id'] for c in chunks})
    doc_summaries = fetch_doc_summaries(cfg, unique_doc_ids)
    query_str = ' | '.join(intents)
    result_chunks = select_relevant_chunks(cfg, query_str, chunks, doc_summaries)
    logger.info('  Selected: %d / %d', len(result_chunks), len(chunks))

    # # 5. Heuristic retry (temporarily disabled — using pre-answer analysis instead)
    # found_docs = {c['doc_id'] for c in result_chunks} if result_chunks else set()
    # needs_retry = False
    #
    # if not result_chunks:
    #     needs_retry = True
    #     logger.info('  Retry: select returned 0 chunks')
    # elif is_multidoc:
    #     found_cases = set()
    #     for c in result_chunks:
    #         searchable = ' '.join([
    #             c.get('text', ''),
    #             ' '.join(c.get('path', [])),
    #             c.get('context', ''),
    #             doc_summaries.get(c['doc_id'], ''),
    #         ]).lower().replace(' ', '')
    #         for mc in mentioned_cases:
    #             if mc.lower().replace(' ', '') in searchable:
    #                 found_cases.add(mc)
    #     missing = mentioned_cases - found_cases
    #     if missing:
    #         needs_retry = True
    #         logger.info('  Retry: missing cases %s', missing)
    #
    # if needs_retry:
    #     retry_prompt = (
    #         f'Question: "{question}"\n\n'
    #         'The initial search did not find enough relevant documents.\n'
    #         'Formulate 2-3 very specific search queries to find the missing information.\n'
    #         'Focus on document titles, case numbers, party names, or article numbers.\n'
    #         'Return ONLY the queries, one per line.'
    #     )
    #     retry_queries_raw = llm_complete(
    #         cfg, cfg.intent_model_url, cfg.intent_model, cfg.intent_api_key,
    #         [{'role': 'user', 'content': retry_prompt}],
    #         temperature=0.0, max_tokens=200,
    #     )
    #     retry_queries = [q.strip().strip('-•*') for q in retry_queries_raw.strip().split('\n') if q.strip()]
    #     logger.info('  Retry queries: %s', retry_queries)
    #
    #     with ThreadPoolExecutor() as executor:
    #         retry_results = list(executor.map(
    #             lambda q: search(cfg, q, num_results), retry_queries,
    #         ))
    #     for batch in retry_results:
    #         for chunk in batch:
    #             cid = chunk['chunk_id']
    #             if cid not in seen or chunk['score'] > seen[cid]['score']:
    #                 seen[cid] = chunk
    #     chunks = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
    #     chunks = chunks[:num_results]
    #
    #     doc_best_score = {}
    #     for c in chunks:
    #         doc_best_score[c['doc_id']] = max(doc_best_score.get(c['doc_id'], 0.0), c['score'])
    #     chunks = sorted(chunks, key=lambda x: (-doc_best_score[x['doc_id']], x['order']))
    #
    #     unique_doc_ids = list({c['doc_id'] for c in chunks})
    #     doc_summaries = fetch_doc_summaries(cfg, unique_doc_ids)
    #     result_chunks = select_relevant_chunks(cfg, query_str, chunks, doc_summaries)
    #     logger.info('  Retry selected: %d / %d', len(result_chunks), len(chunks))

    # 6. Pre-answer analysis: ask model if it has enough info to answer
    if result_chunks:
        preview_context = build_context(result_chunks, doc_summaries)
        analysis_prompt = (
            f'Question: "{question}"\n'
            f'Answer type: {answer_type}\n\n'
            f'{preview_context}\n\n'
            'Do you have enough information in the chunks above to answer this question?\n'
            'If YES, respond with exactly: SUFFICIENT\n'
            'If NO, respond with 1-3 specific search queries to find the missing information '
            '(one per line, no numbering).'
        )
    else:
        # Select returned 0 — ask model what to search for
        analysis_prompt = (
            f'Question: "{question}"\n'
            f'Answer type: {answer_type}\n\n'
            'No relevant chunks were found in the initial search.\n'
            'Formulate 1-3 specific search queries to find the information needed '
            'to answer this question.\n'
            'Focus on document titles, case numbers, party names, or article numbers.\n'
            'Return ONLY the queries, one per line.'
        )

    analysis = llm_complete(
        cfg, cfg.llm_url, cfg.llm_model, cfg.llm_api_key,
        [{'role': 'user', 'content': analysis_prompt}],
        temperature=0.0, max_tokens=200,
    ).strip()

    if 'SUFFICIENT' not in analysis.upper():
        logger.info('  Pre-answer retry: %s', analysis.replace('\n', ' | ')[:100])
        retry_queries = [q.strip().strip('-•*') for q in analysis.split('\n') if q.strip()]
        if retry_queries:
            with ThreadPoolExecutor() as executor:
                extra_results = list(executor.map(
                    lambda q: search(cfg, q, num_results), retry_queries,
                ))
            for batch in extra_results:
                for chunk in batch:
                    cid = chunk['chunk_id']
                    if cid not in seen or chunk['score'] > seen[cid]['score']:
                        seen[cid] = chunk
            chunks = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
            chunks = chunks[:num_results]

            doc_best_score = {}
            for c in chunks:
                doc_best_score[c['doc_id']] = max(
                    doc_best_score.get(c['doc_id'], 0.0), c['score'],
                )
            chunks = sorted(
                chunks, key=lambda x: (-doc_best_score[x['doc_id']], x['order']),
            )
            unique_doc_ids = list({c['doc_id'] for c in chunks})
            doc_summaries = fetch_doc_summaries(cfg, unique_doc_ids)
            result_chunks = select_relevant_chunks(
                cfg, query_str, chunks, doc_summaries,
            )
            logger.info('  Pre-answer retry selected: %d / %d',
                        len(result_chunks), len(chunks))

    # 6b. Expand neighbors only for selected chunks
    if cfg.neighbor_window > 0 and result_chunks:
        result_chunks = expand_neighbors(cfg, result_chunks, cfg.neighbor_window)
        result_chunks = sorted(result_chunks, key=lambda x: (x['doc_id'], x['order']))
        logger.info('  After expand: %d chunks', len(result_chunks))

    if not result_chunks:
        # No relevant chunks — return appropriate "not found" value by type
        if answer_type == 'free_text':
            answer_value = 'There is no information on this question in the provided documents.'
        elif answer_type == 'boolean':
            answer_value = False  # absence of evidence = false
        else:
            answer_value = None
        return SubmissionAnswer(
            question_id=q_id,
            answer=answer_value,
            telemetry=Telemetry(
                timing=TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0),
                retrieval=[],
                usage=UsageMetrics(input_tokens=0, output_tokens=0),
                model_name=cfg.llm_model,
            ),
        )

    # 7. Build context (doc_summaries already fetched in step 4)
    context = build_context(result_chunks, doc_summaries)

    # 8. Generate answer with timing
    answer_value, chunks_used, timing, usage = generate_answer(
        cfg, question, answer_type, context,
    )
    # 8b. Summarize free_text if over 280 chars
    if answer_type == 'free_text' and isinstance(answer_value, str) and len(answer_value) > 280:
        max_tokens_steps = [150, 100, 70]
        for attempt, mt in enumerate(max_tokens_steps):
            logger.info('  Summarizing free_text (%d chars > 280, attempt %d, max_tokens=%d)...',
                        len(answer_value), attempt + 1, mt)
            answer_value = llm_complete(
                cfg, cfg.llm_url, cfg.llm_model, cfg.llm_api_key,
                [{'role': 'user', 'content': (
                    f'Summarize the following text to UNDER 280 characters '
                    f'while preserving the key legal meaning. '
                    f'Currently it is {len(answer_value)} characters. '
                    f'Return ONLY the summarized text, no quotes, no explanation:\n\n'
                    f'{answer_value}'
                )}],
                temperature=0.0, max_tokens=mt,
            ).strip()
            logger.info('  Summarized to %d chars', len(answer_value))
            if len(answer_value) <= 280:
                break

    logger.info(
        '  Answer: %s  chunks_used=%s (ttft=%dms)',
        repr(answer_value)[:80], chunks_used, timing.ttft_ms,
    )

    # 9. Narrow retrieval refs to only chunks the LLM actually used
    #    If answer is null/None → retrieval must be [] (both empty = grounding 1.0)
    _no_info = 'there is no information'
    if answer_value is None or (
        isinstance(answer_value, str) and _no_info in answer_value.lower()
    ):
        retrieval_refs = []
    elif chunks_used:
        used_chunks = [
            result_chunks[n - 1] for n in chunks_used
            if 1 <= n <= len(result_chunks)
        ]
        retrieval_refs = collect_retrieval_refs(used_chunks)
    else:
        # Fallback: use all selected chunks
        retrieval_refs = collect_retrieval_refs(result_chunks)

    return SubmissionAnswer(
        question_id=q_id,
        answer=answer_value,
        telemetry=Telemetry(
            timing=timing,
            retrieval=retrieval_refs,
            usage=usage,
            model_name=cfg.llm_model,
        ),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Run RAG competition pipeline')
    parser.add_argument('--questions', type=str, help='Path to cached questions.json')
    parser.add_argument('--output', type=str, default='submission.json', help='Output path')
    parser.add_argument('--submit', action='store_true', help='Submit to platform after answering')
    parser.add_argument('--resume', action='store_true', help='Skip already answered questions')
    args = parser.parse_args()

    cfg = Config()
    output_path = Path(args.output)

    # Load questions
    if args.questions and Path(args.questions).exists():
        logger.info('Loading questions from %s', args.questions)
        questions = json.loads(Path(args.questions).read_text(encoding='utf-8'))
    else:
        logger.info('Downloading questions from platform...')
        client = EvaluationClient.from_env()
        cache_path = args.questions or 'questions.json'
        questions = client.download_questions(cache_path)
        logger.info('Saved %d questions to %s', len(questions), cache_path)

    # Load existing answers for --resume
    existing: dict[str, dict] = {}
    if args.resume and output_path.exists():
        prev = json.loads(output_path.read_text(encoding='utf-8'))
        for a in prev.get('answers', []):
            existing[a['question_id']] = a
        logger.info('Resuming: %d answers already done', len(existing))

    # Process questions
    builder = SubmissionBuilder(
        architecture_summary='Morag RAG: hybrid search (FRIDA dense + GTE sparse), '
        'semantic chunking, LLM reranker, typed answer extraction',
    )

    for i, q in enumerate(questions, start=1):
        q_id = q['id']
        question = q['question']
        answer_type = q['answer_type']

        if q_id in existing:
            logger.info('[%d/%d] SKIP (already answered): %s', i, len(questions), question[:60])
            # Re-add from previous submission
            prev_answer = existing[q_id]
            tel = prev_answer['telemetry']
            builder.add_answer(SubmissionAnswer(
                question_id=q_id,
                answer=prev_answer['answer'],
                telemetry=Telemetry(
                    timing=TimingMetrics(**tel['timing']),
                    retrieval=normalize_retrieved_pages(
                        tel.get('retrieval', {}).get('retrieved_chunk_pages', []),
                    ),
                    usage=UsageMetrics(**tel['usage']),
                    model_name=tel.get('model_name'),
                ),
            ))
            continue

        logger.info('[%d/%d] %s [%s]', i, len(questions), question[:60], answer_type)
        sa = answer_question(cfg, question, answer_type)
        if sa:
            builder.add_answer(sa)
            # Save incrementally
            builder.save(output_path)

        # # Pause between questions to avoid TPM rate limits
        # if i < len(questions):
        #     time.sleep(10)

    builder.save(output_path)
    logger.info('Saved %d answers to %s', len(builder.answers), output_path)

    if args.submit:
        logger.info('Submitting...')
        client = EvaluationClient.from_env()
        # TODO: create code archive
        result = client.submit_submission(output_path, 'code_archive.zip')
        logger.info('Submitted: %s', result)


if __name__ == '__main__':
    main()
