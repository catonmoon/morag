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
import re
from typing import Any, Coroutine, Dict, Generator, Iterator, List, TypeVar, Union

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from markdown_it import MarkdownIt

# Импорт из installed morag-пакета (ставится через services/pipeline/Dockerfile).
# Файл специально назван morag_pipeline.py (не morag.py) чтобы избежать коллизии с
# пакетом в sys.modules — OWUI регистрирует файл по filename как имя модуля.
from morag.config import Config, load_config
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


# ── Config ↔ Valves merge helpers ─────────────────────────────────────────────
# Источник истины — config.yml (читается в __init__). OWUI Valves остаются как
# OVERRIDE: значение Valve != sentinel (пусто/0/None) → перебивает config.
# Sentinel-defaults в Valves позволяют свежим установкам сразу унаследовать config.

def _str_or(valve: str, cfg: str | None, default: str = '') -> str:
    if valve and valve.strip():
        return valve.strip()
    return cfg if cfg is not None else default


def _int_or(valve: int, cfg: int | None, default: int = 0) -> int:
    if valve:                                        # 0 = sentinel (use config)
        return valve
    return cfg if cfg is not None else default


def _float_or(valve: float, cfg: float | None, default: float = 0.0) -> float:
    if valve:                                        # 0.0 = sentinel
        return valve
    return cfg if cfg is not None else default


def _bool_or(valve: bool | None, cfg: bool | None, default: bool = True) -> bool:
    if valve is not None:                            # None = sentinel
        return valve
    return cfg if cfg is not None else default


def _try_load_config() -> Config | None:
    """Прочитать config.yml + overlay. None если не нашли/не валиден.

    Pipeline должен загружаться даже на свежей установке без config — тогда
    он работает в env-only mode. Все обязательные поля должны быть заданы
    через Valves либо env-vars.
    """
    cfg_path = os.getenv('MORAG_CONFIG_PATH', '/app/conf/config.yml')
    if not os.path.exists(cfg_path):
        logger.info('Config file not found at %s — env-only mode', cfg_path)
        return None
    try:
        return load_config(cfg_path)
    except Exception as exc:
        logger.warning('Failed to load config %s: %s — env-only mode', cfg_path, exc)
        return None


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
            'name': 'get_doc',
            'description': (
                'Глубокое чтение одного документа: тянет все его чанки, реранкер '
                'выбирает релевантные query. Используй когда: (а) после search ты '
                'понимаешь что нужный документ найден, но один-два чанка из выдачи '
                'не дают полной картины; (б) нужно проверить все части большого '
                'документа на релевантность query (search мог пропустить '
                'релевантный фрагмент в хвосте документа).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'doc_id': {
                        'type': 'string',
                        'description': (
                            'ID документа из результатов find_section/search '
                            '(полный prefixed id вида `<kind>:<name>:<external_id>`).'
                        ),
                    },
                    'query': {
                        'type': 'string',
                        'description': (
                            'Какую информацию ищешь в этом документе (на русском, '
                            'словами пользователя из последнего вопроса).'
                        ),
                    },
                },
                'required': ['doc_id', 'query'],
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
    'Первый ход — ВСЕГДА `find_section(query)` со словами пользователя из вопроса. '
    'СОХРАНЯЙ имена, фамилии, названия, ID, специфические термины — это самые сильные '
    'различающие сигналы и их нельзя обобщать.\n'
    'Второй find_section вызывай ТОЛЬКО если первый не дал релевантных секций '
    '(пустой результат, либо найденные разделы явно мимо темы). При втором — '
    'варьируй УГОЛ вопроса (другой аспект, переставленные слова, синонимы тех же '
    'терминов), НО не подменяй конкретные сущности на категории-абстракции.\n'
    'ЗАПРЕЩЕНО:\n'
    '  - «Евгений Чуканов» → «сотрудник Евгений» (выбросил фамилию, добавил категорию)\n'
    '  - «MODP-12345» → «задача разработки» (выбросил конкретный ID)\n'
    '  - «Express» → «JavaScript-фреймворк» (добавил категорию из общей эрудиции)\n'
    '⚠️ Если делал несколько find_section — **ОБЪЕДИНЯЙ** результаты, не замещай. '
    'В search передавай union section_ids и doc_ids от всех вызовов.\n\n'
    '### 2. ВЫПОЛНЕНИЕ\n'
    '- Ищи тщательно. Старайся покрыть вопрос с разных сторон — делай несколько search\'ей '
    'под РАЗНЫЕ грани (процесс vs инструменты vs ответственные), не повторяя один запрос в переформулировках.\n'
    '- ⚠️ ЯЗЫК ЗАПРОСА: сохраняй ключевые русские слова из исходного вопроса. '
    'Документация на русском — поиск на английском не сработает. '
    'Если в вопросе «доверие к сервису распознавания» — так и пиши в search, не переводи на «trust recognition service». '
    'Синонимы/переформулировки допустимы, но на русском.\n'
    '- 🎯 СЛОВАРЬ ЗАПРОСА: формулируй query словами пользователя из ПОСЛЕДНЕГО вопроса. '
    'Не добавляй термины «общей эрудиции» (npm, Python, Kafka и т.п.) пока корпус сам не показал что они применимы. '
    'Корпус — внутренняя документация, термины могут иметь специфичное значение. '
    'Пример: «Как установить Express?» → search: «установка Express». '
    'Расширять/менять формулировку — только если буквальный поиск ничего релевантного не нашёл. '
    'При расширении используй близкие синонимы и переформулировки тех же терминов '
    '(«установка» → «инструкция», «как настроить», «гайд»), не подменяй имена '
    'технологий/инструментов (npm, Node.js, Python и т.п.) — если их не было в '
    'исходном вопросе, их нет и в корпусе.\n'
    '- 🔄 МНОГОХОДОВЫЙ ДИАЛОГ: помни предыдущие вопросы, но в search идут слова из ПОСЛЕДНЕГО. '
    'Подставляй контекст из прошлого хода только если последний вопрос ссылается на него '
    '(местоимения, эллипсис: «а как его установить?» → подставь предмет из прошлого вопроса).\n'
    '- `section_ids` — рекурсивный поиск (раздел + все его подстраницы). Для широких тем.\n'
    '- `doc_ids` — точечный поиск (только указанные страницы, БЕЗ потомков). Для случаев когда ответ '
    'прямо на странице-разделе (например, страница «Люди» сама перечисляет отделы — её подстраницы не нужны).\n'
    '- find_section подскажет что использовать: «раздел рекурсивно» → section_ids; «страница точечно» → doc_ids.\n'
    '- Если для разных аспектов релевантны разные секции — дополнительно вызови find_section под аспект.\n'
    '- Используй get_doc(doc_id, query) для ГЛУБОКОГО ЧТЕНИЯ одного документа '
    'относительно вопроса — реранкер пройдётся по ВСЕМ его чанкам и вернёт '
    'тематически релевантные. Полезно когда search нашёл нужный документ, '
    'но один-два чанка не дают полной картины, или если документ большой '
    'и search мог пропустить релевантный фрагмент в его хвосте.\n'
    '- ⚠️ ШУМ ПРИ ШИРОКОМ ПОИСКЕ: если search вернул результаты из 10+ разных документов — '
    'это сигнал что запрос слишком общий, выдача шумная. Сузь следующий шаг: '
    'переформулируй запрос точнее (более специфичные термины) ИЛИ ограничь section_ids '
    'двумя-тремя самыми релевантными разделами из карты. '
    'Не пытайся «прочитать все 10» — выбери top-2-3 документа и углубляйся через get_doc().\n\n'
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

def _init_valves_from_env() -> dict:
    """Bootstrap-инициализация Valves из env. Существующие OWUI-инсталляции
    могли указывать значения через docker-compose env (легаси-режим). Эти env
    становятся initial-значениями Valves; OWUI потом override'ит из своей DB.

    Default '' / 0 / None — sentinel «брать из config».
    """
    def _e(name: str) -> str:
        return os.getenv(name, '').strip()

    def _ei(name: str) -> int:
        v = _e(name)
        return int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else 0

    def _ef(name: str) -> float:
        v = _e(name)
        try:
            return float(v) if v else 0.0
        except ValueError:
            return 0.0

    def _eb(name: str) -> bool | None:
        v = _e(name).lower()
        if v in ('true', '1', 'yes', 'on'):
            return True
        if v in ('false', '0', 'no', 'off'):
            return False
        return None

    return {
        'QDRANT_URL': _e('QDRANT_URL'),
        'QDRANT_COLLECTION': _e('QDRANT_COLLECTION'),
        'QDRANT_DOCS_COLLECTION': _e('QDRANT_DOCS_COLLECTION'),
        'QDRANT_KNOWLEDGE_MAP_COLLECTION': _e('QDRANT_KNOWLEDGE_MAP_COLLECTION'),
        'SPARSE_EMBED_URL': _e('SPARSE_EMBED_URL'),
        'DENSE_EMBED_URL': _e('DENSE_EMBED_URL'),
        'DENSE_EMBED_API_KEY': _e('DENSE_EMBED_API_KEY'),
        'DENSE_EMBEDDER_MODEL': _e('DENSE_EMBEDDER_MODEL'),
        'DENSE_DIM': _ei('DENSE_DIM'),
        'QUERY_TEMPLATE': _e('QUERY_TEMPLATE'),
        'LLM_URL': _e('LLM_URL'),
        'LLM_MODEL': _e('LLM_MODEL'),
        'LLM_API_KEY': _e('LLM_API_KEY'),
        'LLM_TEMPERATURE': _ef('LLM_TEMPERATURE'),
        'LLM_MAX_TOKENS': _ei('LLM_MAX_TOKENS'),
        'LLM_ANSWER_MAX_TOKENS': _ei('LLM_ANSWER_MAX_TOKENS'),
        'RERANK_LLM_URL': _e('RERANK_LLM_URL'),
        'RERANK_LLM_MODEL': _e('RERANK_LLM_MODEL'),
        'RERANK_LLM_API_KEY': _e('RERANK_LLM_API_KEY'),
        'RERANK_MAX_TOKENS': _ei('RERANK_MAX_TOKENS'),
        'SEARCH_LIMIT': _ei('SEARCH_LIMIT'),
        'UNIQUE_DOCS_CAP': _ei('UNIQUE_DOCS_CAP'),
        'SECTIONS_LIMIT': _ei('SECTIONS_LIMIT'),
        'FIND_SECTION_DOC_POOL': _ei('FIND_SECTION_DOC_POOL'),
        'FIND_SECTION_DESCENT_THRESHOLD': _ef('FIND_SECTION_DESCENT_THRESHOLD'),
        'FIND_SECTION_TOP_DOCS': _ei('FIND_SECTION_TOP_DOCS'),
        'FIND_SECTION_CHUNK_PEEK_LIMIT': _ei('FIND_SECTION_CHUNK_PEEK_LIMIT'),
        'FIND_SECTION_CHUNK_PEEK_DOCS': _ei('FIND_SECTION_CHUNK_PEEK_DOCS'),
        'MAX_ITERATIONS': _ei('MAX_ITERATIONS'),
        'ENABLE_THINKING': _eb('ENABLE_THINKING'),
        'RERANK_ENABLE_THINKING': _eb('RERANK_ENABLE_THINKING'),
        'ENABLE_DIVERSITY_NUDGE': _eb('ENABLE_DIVERSITY_NUDGE'),
        'HTTP_TIMEOUT': _ei('HTTP_TIMEOUT'),
        'ADMIN_INSTRUCTIONS': _e('ADMIN_INSTRUCTIONS'),
    }


def _resolve_settings(v: 'Pipeline.Valves', cfg: Config | None) -> dict:
    """Merge Valves + config → плоский dict с финальными значениями.

    Приоритет: Valve (если != sentinel) → config → hardcoded fallback.
    """
    retr = cfg.retrieval if cfg else None
    idx = cfg.indexing if cfg else None
    qdrant_cfg = cfg.qdrant if cfg else None
    dense = idx.dense_embedder if (idx and idx.dense_embedder) else None
    sparse = idx.sparse_embedder if idx else None
    km = idx.knowledge_map if idx else None
    agent_role = retr.agent if retr else None
    rerank_role = retr.reranker if retr else None
    search = retr.search if retr else None
    find_sec = search.find_section if search else None
    features = retr.features if retr else None
    prompts = retr.prompts if retr else None

    # Резолв agent LLM-инстанса из llms-pool
    agent_llm = cfg.llm_by_name(agent_role.llm) if (agent_role and cfg) else None
    rerank_llm = cfg.llm_by_name(rerank_role.llm) if (rerank_role and cfg) else None

    # Default fallback для Qdrant URL: docker-compose hostname
    qdrant_url_cfg = (
        f'http://{qdrant_cfg.host}:{qdrant_cfg.port}' if qdrant_cfg else None
    )

    s = {
        'qdrant_url': _str_or(v.QDRANT_URL, qdrant_url_cfg, default='http://qdrant:6333'),
        'chunks_collection': _str_or(
            v.QDRANT_COLLECTION,
            qdrant_cfg.collection_chunks if qdrant_cfg else None,
            default='chunks',
        ),
        'docs_collection': _str_or(
            v.QDRANT_DOCS_COLLECTION,
            qdrant_cfg.collection_docs if qdrant_cfg else None,
            default='docs',
        ),
        'knowledge_map_collection': _str_or(
            v.QDRANT_KNOWLEDGE_MAP_COLLECTION,
            km.collection if km else None,
            default='knowledge_map',
        ),

        'sparse_url': _str_or(
            v.SPARSE_EMBED_URL, sparse.base_url if sparse else None,
            default='http://embedder-gte:8081',
        ),
        'dense_url': _str_or(v.DENSE_EMBED_URL, dense.base_url if dense else None),
        'dense_api_key': _str_or(v.DENSE_EMBED_API_KEY, dense.api_key if dense else None),
        'dense_model': _str_or(v.DENSE_EMBEDDER_MODEL, dense.model if dense else None),
        'dense_dim': _int_or(v.DENSE_DIM, dense.dim if dense else None),
        'query_template': _str_or(
            v.QUERY_TEMPLATE,
            (dense.query_template if dense else None),
            default=(
                'Instruct: Given a user question, retrieve passages that answer '
                'the question\nQuery:{text}'
            ),
        ),

        'agent_url': _str_or(v.LLM_URL, agent_llm.base_url if agent_llm else None),
        'agent_model': _str_or(v.LLM_MODEL, agent_llm.model if agent_llm else None),
        'agent_api_key': _str_or(v.LLM_API_KEY, agent_llm.api_key if agent_llm else None),
        'agent_context_window': agent_llm.context_window if agent_llm else 0,
        'agent_temperature': _float_or(
            v.LLM_TEMPERATURE,
            agent_role.temperature if agent_role else None,
            default=0.3,
        ),
        # Анти-петельные параметры для Qwen3 9B (см. QwenLM Issue #145).
        # Не выносим в Valves — это не пользовательский настройка, а тех-параметр.
        'agent_top_p': agent_role.top_p if agent_role else 1.0,
        'agent_top_k': agent_role.top_k if agent_role else 0,
        'agent_presence_penalty': agent_role.presence_penalty if agent_role else 0.0,
        'agent_max_tokens': _int_or(
            v.LLM_MAX_TOKENS,
            agent_role.max_tokens if agent_role else None,
            default=4096,
        ),
        'agent_answer_max_tokens': _int_or(v.LLM_ANSWER_MAX_TOKENS, None, default=0),
        # bool|None: None = НЕ слать reasoning-флаги (xAI compat). Значение из
        # valve приоритетнее; при отсутствии — из config; иначе None (default).
        'agent_enable_thinking': (
            v.ENABLE_THINKING if v.ENABLE_THINKING is not None
            else (agent_role.enable_thinking if agent_role else None)
        ),

        # Reranker — fallback на agent (через config OR через valves) если не задан отдельно.
        # Цепочка: RERANK Valve → reranker config → agent Valve → agent config.
        'rerank_url': _str_or(
            v.RERANK_LLM_URL,
            rerank_llm.base_url if rerank_llm else None,
        ) or _str_or(v.LLM_URL, agent_llm.base_url if agent_llm else None),
        'rerank_model': _str_or(
            v.RERANK_LLM_MODEL,
            rerank_llm.model if rerank_llm else None,
        ) or _str_or(v.LLM_MODEL, agent_llm.model if agent_llm else None),
        'rerank_api_key': _str_or(
            v.RERANK_LLM_API_KEY,
            rerank_llm.api_key if rerank_llm else None,
        ) or _str_or(v.LLM_API_KEY, agent_llm.api_key if agent_llm else None),
        'rerank_context_window': (
            (rerank_llm.context_window if rerank_llm else 0)
            or (agent_llm.context_window if agent_llm else 0)
        ),
        'rerank_max_tokens': _int_or(
            v.RERANK_MAX_TOKENS,
            rerank_role.max_tokens if rerank_role else None,
            default=100,
        ),
        # rerank по умолчанию thinking=False (быстрее, structured output);
        # None допустим если юзер явно хочет «не слать флаги».
        'rerank_enable_thinking': (
            v.RERANK_ENABLE_THINKING if v.RERANK_ENABLE_THINKING is not None
            else (rerank_role.enable_thinking if rerank_role else False)
        ),

        'search_limit': _int_or(v.SEARCH_LIMIT, search.limit if search else None, default=100),
        'hnsw_ef': _int_or(v.HNSW_EF, search.hnsw_ef if search else None, default=0),
        'search_rerank_max_tokens': search.rerank_max_tokens if search else 0,
        'get_doc_rerank_batch_max_tokens': (
            search.get_doc.rerank_batch_max_tokens if (search and search.get_doc) else 0
        ),
        # source_name → role и source_name → kind snapshot для role-aware фильтрации
        # в HybridSearcher (см. RetrievalSearchConfig + _SourceBase.role).
        'source_roles': cfg.source_roles_map() if cfg else {},
        'source_kinds': cfg.source_kinds_map() if cfg else {},
        'unique_docs_cap': _int_or(
            v.UNIQUE_DOCS_CAP, search.unique_docs_cap if search else None, default=10,
        ),
        'sections_limit': _int_or(
            v.SECTIONS_LIMIT, search.sections_limit if search else None, default=5,
        ),
        'find_section_doc_pool': _int_or(
            v.FIND_SECTION_DOC_POOL,
            find_sec.doc_pool if find_sec else None, default=20,
        ),
        'find_section_descent_threshold': _float_or(
            v.FIND_SECTION_DESCENT_THRESHOLD,
            find_sec.descent_threshold if find_sec else None, default=0.5,
        ),
        'find_section_top_docs': _int_or(
            v.FIND_SECTION_TOP_DOCS,
            find_sec.top_docs if find_sec else None, default=3,
        ),
        'find_section_chunk_peek_limit': _int_or(
            v.FIND_SECTION_CHUNK_PEEK_LIMIT,
            find_sec.chunk_peek_limit if find_sec else None, default=10,
        ),
        'find_section_chunk_peek_docs': _int_or(
            v.FIND_SECTION_CHUNK_PEEK_DOCS,
            find_sec.chunk_peek_docs if find_sec else None, default=3,
        ),
        'max_iterations': _int_or(
            v.MAX_ITERATIONS, search.max_iterations if search else None, default=9,
        ),
        'enable_diversity_nudge': _bool_or(
            v.ENABLE_DIVERSITY_NUDGE,
            features.enable_diversity_nudge if features else None,
            default=True,
        ),

        'http_timeout': _int_or(
            v.HTTP_TIMEOUT, retr.http_timeout if retr else None, default=300,
        ),
        'admin_instructions': _str_or(
            v.ADMIN_INSTRUCTIONS,
            prompts.admin_instructions if prompts else None,
            default=(
                'Если информация не была найдена в конкретном разделе знаний '
                'или её недостаточно для полного ответа, ОБЯЗАТЕЛЬНО сделай '
                'дополнительный поиск без указания раздела (section_ids) — '
                'по всей базе знаний.'
            ),
        ),
    }
    return s


class Pipeline:
    """OWUI Pipelines class. Конфигурация source-of-truth — `config.yml` в
    bind-mount'е `/app/conf/`. Pipeline читает его в `__init__` и больше не
    перечитывает (изменения требуют `docker compose restart pipelines`).

    Valves остаются как override-механизм для админа через OWUI UI: пустые/
    нулевые значения = «использовать config», непустые = override этого поля.
    """

    class Valves(BaseModel):
        # Все sentinel-default ('' / 0 / 0.0 / None) — означают «брать из config».
        # OWUI юзер вписывает реальное значение → перебивает config-сторону.
        QDRANT_URL: str = ''
        QDRANT_COLLECTION: str = ''
        QDRANT_DOCS_COLLECTION: str = ''
        QDRANT_KNOWLEDGE_MAP_COLLECTION: str = ''

        SPARSE_EMBED_URL: str = ''
        DENSE_EMBED_URL: str = ''           # OpenAI-compat endpoint, ОБЯЗАТЕЛЬНО с /v1
        DENSE_EMBED_API_KEY: str = ''       # api_key для dense-embedder (если требует auth)
        DENSE_EMBEDDER_MODEL: str = ''
        DENSE_DIM: int = 0
        QUERY_TEMPLATE: str = ''

        # Agent LLM (function calling)
        LLM_URL: str = ''
        LLM_MODEL: str = ''
        LLM_API_KEY: str = ''
        LLM_TEMPERATURE: float = 0.0
        LLM_MAX_TOKENS: int = 0
        LLM_ANSWER_MAX_TOKENS: int = 0

        # Reranker LLM (по умолчанию — тот же что agent, override через Valves)
        RERANK_LLM_URL: str = ''
        RERANK_LLM_MODEL: str = ''
        RERANK_LLM_API_KEY: str = ''
        RERANK_MAX_TOKENS: int = 0

        # Search params
        SEARCH_LIMIT: int = 0
        HNSW_EF: int = 0                    # HNSW search-time ef (0 = Qdrant default)
        UNIQUE_DOCS_CAP: int = 0            # hard cap на уникальные документы (0 = config / без лимита)
        SECTIONS_LIMIT: int = 0
        FIND_SECTION_DOC_POOL: int = 0
        FIND_SECTION_DESCENT_THRESHOLD: float = 0.0
        FIND_SECTION_TOP_DOCS: int = 0
        FIND_SECTION_CHUNK_PEEK_LIMIT: int = 0       # chunk-level peek (ADR-0013)
        FIND_SECTION_CHUNK_PEEK_DOCS: int = 0
        MAX_ITERATIONS: int = 0
        ENABLE_THINKING: bool | None = None       # None = config; True/False = override agent thinking
        RERANK_ENABLE_THINKING: bool | None = None
        ENABLE_DIVERSITY_NUDGE: bool | None = None
        HTTP_TIMEOUT: int = 0
        ADMIN_INSTRUCTIONS: str = ''

    def __init__(self):
        # OWUI идентифицирует pipeline по self.id (а имя модели берёт из self.name).
        # Если их нет — fallback на имя файла. Чтобы локализовать различия между
        # деплоями в одну env-переменную (вместо переименования файла билд-таймом),
        # читаем PIPELINE_NAME из env. Каждый деплой ставит свой `PIPELINE_NAME`
        # в `docker-compose.yml` — Dockerfile/код остаются общими.
        self.id = os.getenv('PIPELINE_NAME', 'morag')
        self.name = self.id

        # 0. Опц. DEBUG-логирование — для диагностики retrieval/find_section
        # через `docker compose logs pipelines`. Включается env MORAG_LOG_LEVEL=DEBUG.
        log_level = (os.getenv('MORAG_LOG_LEVEL') or '').upper()
        if log_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
            for name in ('morag_pipeline', 'morag.retrieval'):
                logging.getLogger(name).setLevel(log_level)

        # 1. Прочитать config (fail-soft) и инициализировать Valves из env (back-compat)
        self._config: Config | None = _try_load_config()
        self.valves = self.Valves(**_init_valves_from_env())

        # 2. Резолв всех настроек: Valve если задан, иначе config, иначе hardcoded fallback.
        s = _resolve_settings(self.valves, self._config)

        # 3. Persistent event loop — pipe() синхронный, async-вызовы через self._run().
        # OWUI Pipelines может обрабатывать запросы параллельно (worker thread per request).
        # Persistent self._loop ОДИН на инстанс Pipeline, и run_until_complete не выносит
        # повторный заход пока loop ещё работает (бывший баг «this event loop is already
        # running»). Lock сериализует доступ — N запросов выстраиваются в очередь.
        # Цена: при двух параллельных вопросах второй ждёт первого. Допустимо: пайплайн
        # тяжёлый (LLM-вызовы), параллель в пределах одного инстанса не выигрывает.
        import threading
        self._loop = asyncio.new_event_loop()
        self._loop_lock = threading.Lock()

        # 4. Два LLMClient: agent (tool calls + final stream) и reranker.
        #    enable_thinking=None — НЕ слать reasoning-флаги в extra_body
        #    (xAI Grok реджектит unknown body fields). True/False — слать явные.
        #    Если agent_url пустой после полного резолва (нет ни в config, ни в Valves)
        #    — pipeline технически жив, но при первом вызове ответит юзеру понятной
        #    ошибкой (см. _ensure_agent_ready в pipe()).
        self._llm_agent = LLMClient(
            base_url=s['agent_url'] or 'http://invalid', model=s['agent_model'] or 'invalid',
            api_key=s['agent_api_key'] or 'invalid',
            timeout=s['http_timeout'], max_retries=3,
            enable_thinking=s['agent_enable_thinking'],
            context_window=s.get('agent_context_window') or 32768,
        )
        self._llm_rerank = LLMClient(
            base_url=s['rerank_url'] or 'http://invalid', model=s['rerank_model'] or 'invalid',
            api_key=s['rerank_api_key'] or 'invalid',
            timeout=s['http_timeout'], max_retries=3,
            enable_thinking=s['rerank_enable_thinking'],
            context_window=s.get('rerank_context_window') or 32768,
        )

        # 5. Embedders: те же async-классы что в indexing — гарантия совпадения
        #    query_template для dense-канала.
        self._dense_embedder = HttpEmbedder(
            base_url=s['dense_url'], model=s['dense_model'], dim=s['dense_dim'],
            api_key=s['dense_api_key'],
            query_template=s['query_template'], timeout=s['http_timeout'],
        )
        self._sparse_embedder = HttpGteSparseEmbedder(
            base_url=s['sparse_url'], timeout=s['http_timeout'],
        )
        self._qdrant = AsyncQdrantClient(url=s['qdrant_url'], timeout=s['http_timeout'])
        self._searcher = HybridSearcher(
            qdrant=self._qdrant,
            dense_embedder=self._dense_embedder, sparse_embedder=self._sparse_embedder,
            chunks_collection=s['chunks_collection'],
            docs_collection=s['docs_collection'],
            knowledge_map_collection=s['knowledge_map_collection'],
            hnsw_ef=s['hnsw_ef'],
            source_roles=s['source_roles'],
            source_kinds=s['source_kinds'],
        )
        # Реранкеры (search и get_doc) — оба используют rerank-LLM + TiktokenCounter
        # для подсчёта токенов. Бюджет input'а считается по `llm.context_window`.
        from morag.indexing.token_counter import TiktokenCounter
        from morag.retrieval import DocReranker
        self._reranker = LLMReranker(
            self._llm_rerank,
            token_counter=TiktokenCounter(),
            max_tokens=s['rerank_max_tokens'] or 100,
            enable_thinking=s['rerank_enable_thinking'],
            max_input_tokens=s.get('search_rerank_max_tokens', 0),
        )
        self._doc_reranker = DocReranker(
            self._llm_rerank,
            token_counter=TiktokenCounter(),
            max_tokens=s['rerank_max_tokens'] or 200,
            enable_thinking=s['rerank_enable_thinking'],
            max_input_tokens=s.get('get_doc_rerank_batch_max_tokens', 0),
        )
        # DocRepository — для get_doc tool (читает full doc metadata).
        from morag.storage.repository import DocRepository
        self._doc_repo = DocRepository(self._qdrant, s['docs_collection'])
        self._find_section_config = FindSectionConfig(
            sections_limit=s['sections_limit'],
            doc_pool=s['find_section_doc_pool'],
            descent_threshold=s['find_section_descent_threshold'],
            top_docs=s['find_section_top_docs'],
            chunk_peek_limit=s['find_section_chunk_peek_limit'],
            chunk_peek_docs=s['find_section_chunk_peek_docs'],
        )

        # 6. Сохранить merged settings — pipe() читает их вместо self.valves
        #    (там sentinel-defaults, fully resolved тут).
        self._s = s
        logger.info(
            'Pipeline initialized: agent=%s/%s, rerank=%s/%s, qdrant=%s, '
            'config_loaded=%s', s['agent_url'], s['agent_model'],
            s['rerank_url'], s['rerank_model'], s['qdrant_url'],
            self._config is not None,
        )

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Выполнить async-корутину в нашем persistent event loop. Sync-обёртка для pipe().

        Lock сериализует параллельные вызовы (OWUI worker threads): пока loop
        работает над одной корутиной — следующая ждёт. Без него вторая получит
        `RuntimeError: this event loop is already running`.
        """
        with self._loop_lock:
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

        # 0.5 Sanity-check agent LLM. Если конфиг неполный (нет retrieval.agent
        # и Valves пусты) — отвечаем понятным сообщением вместо HTTP-падения.
        if not self._s.get('agent_url') or not self._s.get('agent_model'):
            yield (
                '⚠️ Pipeline не сконфигурирован. Зайдите в Console UI '
                '(http://localhost:8000) → Retrieval → выберите agent.llm и '
                'reranker.llm из пула, нажмите «Сохранить retrieval-настройки», '
                'затем выполните `docker compose restart pipelines`.'
            )
            return

        # 1. Подтянуть карту документации
        knowledge_map = self._fetch_knowledge_map()

        # 2. Собрать system prompt
        system_content = _SYSTEM_PROMPT
        if self._s['admin_instructions']:
            system_content += (
                '\n\n## Обязательные инструкции администратора\n'
                + self._s['admin_instructions']
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
        # Сквозная нумерация документов per-pipe-call. Каждый tool-выдача форматирует
        # `[N] Документ:` используя глобальный N — повторный документ из второго
        # search получит тот же номер что в первом. Решает проблему конфликтующих
        # цитат при многошаговом ретривале (ранее каждый tool_result начинал с [1]).
        self._doc_numbering: dict[str, int] = {}

        logger.info('=' * 70)
        logger.info('[agent] query: %r', last_content[:200])

        for iteration in range(self._s['max_iterations']):
            # Вызов LLM с tools
            response = self._llm_call_with_tools(agent_messages)
            message = response['choices'][0]['message']
            finish_reason = response['choices'][0].get('finish_reason', '')

            tool_calls = message.get('tool_calls') or []
            logger.info(
                '[agent] iter=%d finish_reason=%s tool_calls=%d',
                iteration + 1, finish_reason, len(tool_calls),
            )

            # Если LLM решил ответить (не вызвал tool)
            if finish_reason != 'tool_calls' or not message.get('tool_calls'):
                # Diversity check: все чанки из ≤1 документа после ≥2 search →
                # инжектим nudge и продолжаем цикл вместо ответа
                unique_docs = {c['doc_id'] for c in all_chunks.values()}
                if (
                    self._s['enable_diversity_nudge']
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

                doc_count = len(unique_docs)
                yield self._emit_status(
                    '✅', f'Найдено {_plural(doc_count, "документ", "документа", "документов")} за {_plural(tool_call_count, "шаг", "шага", "шагов")}', True,
                )
                logger.info(
                    '[agent] DONE: %d unique docs, %d tool calls, %d iters; by source_type=%s',
                    doc_count, tool_call_count, iteration + 1,
                    _count_by(list(all_chunks.values()), 'source_type'),
                )
                # Stream финального ответа, параллельно буферим текст для подсветки
                # цитат — после стрима оборачиваем в <mark> предложения из чанков,
                # которые встречаются в ответе.
                agent_messages.append(message)
                answer_parts: list[str] = []
                for token in self._stream_final(agent_messages):
                    answer_parts.append(token)
                    yield token
                yield from self._emit_grouped_sources(
                    _highlight_chunks(all_chunks, ''.join(answer_parts)),
                )
                return

            # LLM вызвал tools — обработать
            agent_messages.append(message)

            for tool_call in message['tool_calls']:
                tool_call_count += 1
                fn_name = tool_call['function']['name']
                fn_args = json.loads(tool_call['function']['arguments'])
                call_id = tool_call['id']

                logger.info(
                    '[agent] tool_call #%d: %s args=%s',
                    tool_call_count, fn_name,
                    json.dumps(fn_args, ensure_ascii=False)[:200],
                )

                if fn_name == 'search':
                    search_count += 1
                    for sid in (fn_args.get('section_ids') or []):
                        searched_section_ids.add(sid)

                # Выполнение + статус
                status_text = _format_tool_status(fn_name, fn_args, resolve_title=self._get_doc_title)
                icon = {'search': '🔍', 'find_section': '🗺️', 'get_doc': '📄'}.get(fn_name, '🛠️')
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
        yield self._emit_status('⚠️', f'Лимит итераций ({self._s["max_iterations"]}), генерирую ответ', False)
        doc_count = len({c['doc_id'] for c in all_chunks.values()})
        yield self._emit_status('✅', f'Найдено {_plural(doc_count, "документ", "документа", "документов")}', True)
        answer_parts: list[str] = []
        for token in self._stream_final(agent_messages):
            answer_parts.append(token)
            yield token
        yield from self._emit_grouped_sources(
            _highlight_chunks(all_chunks, ''.join(answer_parts)),
        )

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

    def _global_doc_id(self, doc_id: str) -> int:
        """Сквозной номер документа для текущего pipe()-вызова.

        Назначает 1, 2, 3, ... по порядку первого появления doc_id в любом
        tool-результате. Повторный doc_id получает тот же номер.
        Используется для стабильного цитирования `[N] Документ:` через все
        search/get_doc в рамках одного агентского цикла.
        """
        n = self._doc_numbering.get(doc_id)
        if n is not None:
            return n
        n = len(self._doc_numbering) + 1
        self._doc_numbering[doc_id] = n
        return n

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
        elif name == 'get_doc':
            return self._tool_get_doc(args['doc_id'], args['query'])
        return f'Неизвестный инструмент: {name}', []

    def _tool_search(
        self,
        query: str,
        limit: int | None = None,
        section_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        limit = min(limit or self._s['search_limit'], self._s['search_limit'])
        # scope_active = есть какие-то фильтры от агента → не отрезаем supplementary
        # (descendants раздела естественно тянут привязанные тикеты).
        scope_active = bool(section_ids or doc_ids)
        logger.info(
            '[search] q=%r scope=%s section_ids=%d doc_ids=%d limit=%d',
            query[:100], scope_active,
            len(section_ids or []), len(doc_ids or []), limit,
        )
        chunks = self._search(query, limit, scope_active=scope_active)
        logger.info(
            '[search] retrieved %d chunks; by source_type=%s',
            len(chunks), _count_by(chunks, 'source_type'),
        )
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
            logger.info(
                '[search] descendants filter: %d chunks → %d (allowed_doc_ids=%d)',
                len(chunks), len(filtered), len(allowed_doc_ids),
            )
            if filtered:
                chunks = filtered
                filtered_applied = True

        # LLM reranker — отфильтровать нерелевантные чанки
        rerank_in = len(chunks)
        reranked = self._rerank(query, chunks)
        logger.info(
            '[search] rerank: %d → %d (dropped %d, filter_applied=%s)',
            rerank_in, len(reranked), rerank_in - len(reranked), filtered_applied,
        )
        if not reranked and filtered_applied:
            # Фильтр оставил чанки, но rerank их все выбросил — возможно классификация
            # документа по теме расходится с тем, где агент искал. Повторяем без фильтра.
            logger.info('[search] rerank auto-fallback on raw chunks (%d)', len(raw_chunks))
            reranked = self._rerank(query, raw_chunks)
            logger.info('[search] rerank fallback result: %d chunks', len(reranked))
        if not reranked:
            return 'Поиск дал результаты, но ни один не оказался релевантным. Попробуй другую формулировку.', []
        chunks = reranked

        # Группировка по документу для LLM — в порядке прихода из reranker
        # (LLM возвращает номера в порядке релевантности). Затем hard cap на
        # число уникальных документов: отсекаем хвост маложелательных.
        by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c['doc_id'], []).append(c)
        cap = self._s['unique_docs_cap']
        unique_pre_cap = len(by_doc)
        if cap > 0 and len(by_doc) > cap:
            kept_doc_ids = list(by_doc.keys())[:cap]
            by_doc = {did: by_doc[did] for did in kept_doc_ids}
            chunks = [c for c in chunks if c['doc_id'] in by_doc]
        logger.info(
            '[search] final: %d chunks, %d unique docs (cap=%d, pre=%d); by source_type=%s',
            len(chunks), len(by_doc), cap, unique_pre_cap,
            _count_by(chunks, 'source_type'),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, (did, dcs) in enumerate(by_doc.items(), 1):
                logger.debug(
                    '[search]   [%d] %s (chunks=%d, source_type=%s, top_score=%.3f)',
                    i, did, len(dcs), dcs[0].get('source_type'), max(c['score'] for c in dcs),
                )

        parts = []
        for doc_id, doc_chunks in by_doc.items():
            n = self._global_doc_id(doc_id)
            doc_chunks.sort(key=lambda x: x['order'])
            path_display = ' | '.join(doc_chunks[0]['path']) if doc_chunks[0]['path'] else doc_id
            doc_name = self._get_doc_title(doc_id)
            lines = [f'[{n}] Документ: {doc_name}', f'Путь: {path_display}']
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

    # ── Section-level retrieval (find_section) ────────────────────────────────

    def _tool_find_section(self, query: str) -> tuple[str, list[dict]]:
        """Найти релевантные РАЗДЕЛЫ документации для запроса.

        Вся retrieval-логика (RRF по docs + vote counting + adaptive descent +
        top-K safety + enrichment) живёт в `morag.retrieval.find_section`.
        Pipeline только форматирует SectionResult как markdown для LLM-агента.
        """
        logger.info('[find_section] q=%r', query[:100])
        result = self._run(find_section(query, self._searcher, self._find_section_config))
        if result.error == 'no_docs':
            logger.info('[find_section] no_docs')
            return 'Не удалось найти релевантные документы для определения разделов.', []
        if result.error == 'no_sections' or not (result.refined or result.extra_docs):
            logger.info('[find_section] no_sections')
            return (
                'Не удалось определить разделы для запроса. '
                'Используй обычный search() без section_ids.'
            ), []
        logger.info(
            '[find_section] sections=%d extras=%d → section_ids=%s doc_ids=%s',
            len(result.section_ids), len(result.extra_docs),
            result.section_ids[:5], result.doc_ids[:5],
        )
        if logger.isEnabledFor(logging.DEBUG):
            for e in result.refined:
                logger.debug(
                    '[find_section]   refined: kind=%s id=%s votes=%d title=%r',
                    e.kind, e.section_id, e.votes, e.title[:60],
                )
            for d in result.extra_docs:
                logger.debug(
                    '[find_section]   extra:   id=%s score=%.3f title=%r',
                    d.doc_id, d.score, d.title[:60],
                )

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

    # ── Get-doc (один документ с rerank по чанкам) ────────────────────────────

    def _tool_get_doc(self, doc_id: str, query: str) -> tuple[str, list[dict]]:
        """Получить релевантные фрагменты одного документа.

        Алгоритм:
        1. Подтянуть метаданные документа (title, path, url).
        2. Lite-чанки документа: только {order, text}, без context/path.
        3. DocReranker батчами выбирает релевантные order'ы.
        4. Полные чанки этих order'ов из БД.
        5. Форматирование как обычный search-результат для агента.
        """
        logger.info('[get_doc] doc_id=%s q=%r', doc_id, query[:80])
        # 1. Метаданные документа
        doc = self._run(self._doc_repo.get_by_id(doc_id))
        if doc is None:
            return f'Документ {doc_id} не найден.', []

        # 2. Lite-чанки (без context, в порядке order)
        lite = self._run(self._searcher.fetch_doc_chunks_lite(doc_id))
        if not lite:
            return f'У документа {doc_id} нет проиндексированных чанков.', []
        logger.info('[get_doc] lite chunks=%d', len(lite))

        # 3. DocReranker
        useful_orders = self._run(
            self._doc_reranker.rerank(query, doc.path, lite),
        )
        if not useful_orders:
            return (
                f'Документ найден ({doc.title or doc_id}), '
                f'но ни один из {len(lite)} фрагментов не релевантен запросу. '
                'Попробуй другой документ или search() без doc_ids.'
            ), []
        logger.info('[get_doc] reranker kept orders=%d/%d', len(useful_orders), len(lite))

        # 4. Полные чанки для выбранных order'ов
        full_chunks = self._run(
            self._searcher.fetch_chunks_by_orders(doc_id, useful_orders),
        )

        # 5. Формат как search-результат (с глобальной нумерацией документов)
        n = self._global_doc_id(doc_id)
        path_display = ' | '.join(doc.path) if doc.path else doc_id
        lines = [f'[{n}] Документ: {doc.title or doc_id}', f'Путь: {path_display}']
        if doc.updated_at:
            lines.append(f'Обновлён: {doc.updated_at.isoformat()}')
        if doc.url:
            lines.append(f'URL: {doc.url}')
        lines.append(f'Релевантных фрагментов: {len(full_chunks)} из {len(lite)}')
        lines.append('')
        for c in full_chunks:
            order = c.get('order', 0)
            context = c.get('context', '')
            lines.append(f'[order={order}]')
            if context:
                lines.append(f'Контекст: {context}')
            lines.append(c.get('text', ''))
            lines.append('')
        return '\n'.join(lines), full_chunks

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Тонкая обёртка над LLMReranker (sync↔async через self._run)."""
        return self._run(self._reranker.rerank(query, chunks))

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _llm_call_with_tools(self, messages: list[dict]) -> dict:
        """LLM call с function-calling tools. Non-streaming, retry/429 в SDK.

        enable_thinking=False всегда (default клиента) — для агентского цикла
        (search/find_section decisions) thinking не нужен.
        """
        return self._run(self._llm_agent.complete_with_tools(
            messages,
            tools=_TOOLS,
            params=GenerationParams(
                temperature=self._s['agent_temperature'],
                top_p=self._s['agent_top_p'],
                top_k=self._s['agent_top_k'],
                presence_penalty=self._s['agent_presence_penalty'],
            ),
            max_tokens=self._s['agent_max_tokens'],
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
                'Не пересказывай всё найденное, выдели только главное.\n'
                '- При использовании информации вставляй номер документа-источника '
                'в формате [N], где N — это число из заголовка `[N] Документ: ...` '
                'в результатах tool-вызовов на ЭТОМ ходу. '
                'Нумерация СКВОЗНАЯ — один и тот же документ во всех search/get_doc '
                'имеет ОДНО И ТО ЖЕ N. Если в выдаче встречается `[5] Документ: X` — '
                'для цитирования X всегда используй [5]. '
                'ЗАПРЕЩЕНО ссылаться на номера из прошлых ходов диалога — они уже не действительны. '
                'Например: "Для настройки Docker нужно установить Docker Desktop [1]." '
                'Если информация из нескольких документов — перечисляй: [1][3].\n'
                '- Структурируй ответ максимально: заголовки, подзаголовки, нумерованные и маркированные списки, '
                'таблицы. Разбивай информацию на логические блоки. Избегай сплошного текста.'
            ),
        }]
        max_tokens = self._s['agent_answer_max_tokens'] if self._s['agent_answer_max_tokens'] > 0 else None
        agen = self._llm_agent.stream_complete(
            final_messages,
            params=GenerationParams(
                temperature=self._s['agent_temperature'],
                top_p=self._s['agent_top_p'],
                top_k=self._s['agent_top_k'],
                presence_penalty=self._s['agent_presence_penalty'],
            ),
            max_tokens=max_tokens,
        )
        in_thinking = False
        try:
            while True:
                try:
                    with self._loop_lock:
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

    def _search(self, text: str, limit: int, scope_active: bool = False) -> list[dict]:
        return self._run(self._searcher.search_chunks(text, limit, scope_active=scope_active))

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
        """Сгруппировать чанки по документу, emit один citation на документ
        с массивом чанков в `data.document` (каждый чанк — отдельный элемент)."""
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
            docs.append((path, doc_name, url, chunks, doc_id))

        docs.sort(key=lambda d: d[0])

        for path, doc_name, url, chunks, doc_id in docs:
            yield self._emit_source_multi(
                doc_name, chunks, url,
                source_id=doc_id,
                number=self._doc_numbering.get(doc_id),
            )

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
    def _emit_source_multi(
        name: str, chunks: list[dict], url: str | None = None,
        source_id: str | None = None, number: int | None = None,
    ) -> dict[str, Any]:
        """Citation для документа с несколькими чанками — каждый чанк отдельный
        элемент `data.document` (без склеивания и принудительной обрезки).
        OWUI рендерит массив как развёрнутый список под одной карточкой источника.

        Каждый чанк рендерится как HTML с встроенным CSS и отдаётся с
        `metadata.html: true`. OWUI кладёт такой документ в sandboxed iframe
        через `srcdoc` — это даёт полную изоляцию от темы UI: чёрный текст
        на белом фоне читается одинаково в любой OWUI-теме."""
        documents: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        for c in chunks:
            documents.append(_render_chunk_html(c.get('text', '')))
            meta: dict[str, Any] = {
                'source': source_id or name, 'name': name,
                'html': True,
            }
            if url:
                meta['url'] = url
            if number is not None:
                meta['citation_number'] = number
            if 'order' in c:
                meta['order'] = c['order']
            metadata_list.append(meta)
        source: dict[str, Any] = {'name': name}
        if url:
            source['url'] = url
        return {
            'event': {
                'type': 'citation',
                'data': {
                    'document': documents,
                    'metadata': metadata_list,
                    'source': source,
                },
            }
        }


# Markdown→HTML рендерер для citation-чанков. CommonMark + GFM-таблицы.
# html=True — пропускаем inline-HTML, в частности `<mark>` теги, которые мы
# добавляем для подсветки совпадающих с ответом предложений. Content наш
# собственный (легальные документы из lex.uz), не пользовательский.
_MD = MarkdownIt('commonmark', {'breaks': True, 'html': True}).enable('table')


# Подсветка предложений в чанках, совпадающих с ответом агента ──────────────

_WORD_RE = re.compile(r'[A-Za-zА-Яа-яЁё0-9]+', re.UNICODE)
# Разрезаем на «предложения»: по концу предложения (./!/?) ИЛИ переводам строк,
# чтобы пункты списка и заголовки markdown тоже шли отдельными единицами.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
# 2-словные n-граммы из стеммированных токенов — терпимо к парафразам со
# вставленными словами (например «реклама алкогольной продукции» vs «реклама
# алкоголя»). Stemming устраняет разницу окончаний (запрещается / запрещена).
_SHINGLE_SIZE = 2

# Snowball Russian — uses no external data, bundled in nltk.
try:
    from nltk.stem.snowball import SnowballStemmer
    _STEMMER_RU = SnowballStemmer('russian')
    _STEMMER_EN = SnowballStemmer('english')
except ImportError:
    _STEMMER_RU = None
    _STEMMER_EN = None


def _stem(word: str) -> str:
    """Стем слова — русский для кириллических, английский для латиницы.
    Без nltk возвращаем исходное слово (graceful degradation)."""
    if not _STEMMER_RU:
        return word
    if _CYR_RE.search(word):
        return _STEMMER_RU.stem(word)
    return _STEMMER_EN.stem(word)


_CYR_RE = re.compile(r'[А-Яа-яЁё]')

# Стоп-слова, которые сами по себе не должны давать совпадение (всегда есть в ответах).
_STOPWORDS = frozenset({
    'и', 'в', 'на', 'с', 'по', 'из', 'для', 'не', 'а', 'но', 'или', 'что', 'как',
    'это', 'тот', 'этот', 'который', 'если', 'при', 'до', 'от', 'к', 'у', 'о', 'об',
    'же', 'ли', 'бы', 'то', 'так', 'там', 'тут', 'был', 'была', 'было', 'быть',
    'is', 'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'with',
})


def _normalize_tokens(text: str) -> list[str]:
    """Нижний регистр + стемминг (RU/EN), без стоп-слов."""
    return [
        _stem(w) for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS
    ]


def _shingles(tokens: list[str], size: int = _SHINGLE_SIZE) -> set[tuple[str, ...]]:
    """Множество n-грамм длины size."""
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def _strip_thinking(text: str) -> str:
    """Удалить <think>...</think> блоки — это не часть видимого ответа."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


# Структурные markdown-маркеры в начале строки (заголовок/список/blockquote).
# Их нельзя помещать внутрь <mark>, иначе markdown теряет распознавание.
_MD_PREFIX_RE = re.compile(
    r'^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+))?(.*)$', re.DOTALL,
)


def _highlight_sentences(chunk_text: str, answer_shingles: set[tuple[str, ...]]) -> str:
    """Найти в chunk_text предложения, чьи n-граммы пересекаются с
    answer_shingles, обернуть их в <mark>...</mark>.

    Структура markdown сохраняется один-в-один: разделители (`\\n`, пробелы
    после знака конца предложения) попадают в split через capturing group и
    включаются обратно в результат. Markdown-маркеры (`##`, `-`, `1.`, `>`)
    остаются вне обёртки <mark>, чтобы парсер их видел.
    """
    if not answer_shingles:
        return chunk_text
    # Capturing group в split → separator'ы попадают в результат как чередующиеся
    # элементы (sentence, sep, sentence, sep, ...).
    pieces = re.split(r'((?:(?<=[.!?])\s+)|(?:\n+))', chunk_text)
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i % 2 == 1:  # separator — кладём как есть
            out.append(piece)
            continue
        if not piece.strip():
            out.append(piece)
            continue
        sent_shingles = _shingles(_normalize_tokens(piece))
        if not (sent_shingles & answer_shingles):
            out.append(piece)
            continue
        # Совпадение: оборачиваем содержимое (без структурного префикса).
        m = _MD_PREFIX_RE.match(piece)
        prefix, content = (m.group(1) or ''), m.group(2)
        if content.strip():
            out.append(f'{prefix}<mark>{content}</mark>')
        else:
            out.append(piece)
    return ''.join(out)


def _highlight_chunks(all_chunks: dict[str, dict], answer: str) -> dict[str, dict]:
    """Вернуть новый dict чанков, в которых text-поле обёрнуто <mark> вокруг
    предложений, чьи 4-граммы пересекаются с ответом агента.

    Если ответ пустой или короткий — возвращаем исходные чанки без изменений.
    """
    answer_visible = _strip_thinking(answer)
    answer_tokens = _normalize_tokens(answer_visible)
    answer_shingles = _shingles(answer_tokens)
    if not answer_shingles:
        return all_chunks
    out: dict[str, dict] = {}
    for cid, chunk in all_chunks.items():
        new_chunk = dict(chunk)
        new_chunk['text'] = _highlight_sentences(chunk.get('text', ''), answer_shingles)
        out[cid] = new_chunk
    return out


_CHUNK_HTML_CSS = """\
:root { color-scheme: light; }
html, body { background: #ffffff; color: #1a1a1a; }
body {
  margin: 0; padding: 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-size: 14px; line-height: 1.5;
}
h1, h2, h3, h4 { color: #0a0a0a; margin: 0.6em 0 0.3em; font-weight: 600; }
h1 { font-size: 1.25em; }
h2 { font-size: 1.15em; }
h3 { font-size: 1.05em; }
p { margin: 0.4em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.6em; }
li { margin: 0.2em 0; }
table { border-collapse: collapse; margin: 0.6em 0; width: 100%; }
td, th { border: 1px solid #d0d0d0; padding: 4px 8px; vertical-align: top; }
th { background: #f4f4f4; font-weight: 600; }
code { background: #f5f5f5; padding: 1px 4px; border-radius: 3px;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }
pre { background: #f5f5f5; padding: 8px; border-radius: 4px; overflow-x: auto; }
pre code { background: transparent; padding: 0; }
blockquote { border-left: 3px solid #ccc; margin: 0.5em 0; padding-left: 0.8em; color: #555; }
mark { background: #fff3a0; padding: 0 2px; border-radius: 2px; }
a { color: #0a58ca; }
hr { border: none; border-top: 1px solid #e0e0e0; margin: 1em 0; }
"""


def _render_chunk_html(chunk_text: str) -> str:
    """Markdown-чанк → standalone HTML с встроенным CSS под белый фон.

    OWUI отдаёт `metadata.html: true` контент в sandboxed iframe (`srcdoc`),
    поэтому наши стили не конфликтуют с темой OWUI. Цвета фиксированные —
    чёрный текст на белом фоне в любой теме (требование юристов).
    """
    body = _MD.render(chunk_text or '')
    return (
        '<!DOCTYPE html>'
        '<html lang="ru"><head><meta charset="utf-8">'
        f'<style>{_CHUNK_HTML_CSS}</style></head>'
        f'<body>{body}</body></html>'
    )


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    """Группировка списка dict'ов по значению поля. Для logging-сводок."""
    counts: dict[str, int] = {}
    for it in items:
        k = it.get(key) or '?'
        counts[k] = counts.get(k, 0) + 1
    return counts


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
    if fn_name == 'get_doc':
        doc_id = fn_args.get('doc_id', '')
        query = fn_args.get('query', '')
        title = _title(doc_id)
        return f'[{query}] глубокое чтение документа: {title}'
    return f'{fn_name}({json.dumps(fn_args, ensure_ascii=False)})'


