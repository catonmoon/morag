"""Core-тулы агента: search / find_section / get_doc.

Системные — нельзя выключить или удалить (без них RAG не работает); через
конфиг тюнится только `description` (и `required`-политика у find_section).
Handler'ы = методы Pipeline (dispatch по имени в `_execute_tool`), поэтому
`handler=None`. Тексты схем перенесены из `_TOOLS` morag_pipeline.py дословно.
"""
from __future__ import annotations

from morag.retrieval.tools import ToolSpec

# ── find_section: политика REQUIRED/OPTIONAL ─────────────────────────────────
# Вставляется в {find_section_policy} системного промпта (середина, не хвост).
# REQUIRED (дефолт) — иерархичный корпус (Confluence): мандат «всегда сначала».
# OPTIONAL — плоский корпус (подкаст/собрания): жёсткий мандат ломает агрегатные
# и точечные вопросы (агент обязан искать там, где надо читать каталог).
FIND_SECTION_REQUIRED_POLICY = (
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
)

FIND_SECTION_OPTIONAL_POLICY = (
    '## ИНСТРУМЕНТЫ ПОИСКА\n'
    '`find_section(query)` СУЖАЕТ выдачу до релевантных разделов (по doc-level эмбеддингам '
    'полного текста документов). Полезен в большом или иерархичном корпусе, когда нужно '
    'прицелиться. Но он НЕ обязателен: для точечных и агрегатных вопросов '
    '(перечисли/сколько/конкретный документ, либо обращение к каталогу) можешь идти в '
    'search или соответствующий инструмент напрямую — без предварительного find_section.\n\n'
    '## Алгоритм работы: Find → Execute → Verify\n\n'
    '### 1. FIND SECTION (по необходимости)\n'
    'Если используешь — `find_section(query)` со словами пользователя из вопроса. '
)

_FIND_SECTION_DESC_REQUIRED = (
    'ОБЯЗАТЕЛЬНЫЙ ПЕРВЫЙ ШАГ. Найти релевантные РАЗДЕЛЫ документации по запросу. '
    'Работает через doc-level эмбеддинги (полный текст каждого документа) с агрегацией '
    'по родительскому разделу — возвращает готовые section_ids для последующего search. '
    'ВСЕГДА вызывай перед search(). Без этого шага search бьёт по всему корпусу и выдаёт шум.'
)

_FIND_SECTION_DESC_OPTIONAL = (
    'Найти релевантные РАЗДЕЛЫ корпуса по запросу — сузить область для search. '
    'Работает через doc-level эмбеддинги (полный текст каждого документа) с агрегацией '
    'по родительскому разделу — возвращает section_ids для последующего search. '
    'Полезен для широких тем в большом корпусе; для точечных и агрегатных вопросов не обязателен.'
)


def _search_status(cfg: dict, args: dict, resolve_title) -> str:
    query = args.get('query', '')
    scope: list[str] = []
    if args.get('section_ids'):
        scope.append('в разделах: ' + ', '.join(resolve_title(sid) for sid in args['section_ids']))
    if args.get('doc_ids'):
        scope.append('на страницах: ' + ', '.join(resolve_title(did) for did in args['doc_ids']))
    return f'[{query}] ' + ('; '.join(scope) if scope else 'по всей базе')


SPECS = [
    ToolSpec(
        type='search',
        default_name='search',
        core=True,
        parameters={
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
                        '(например, страница-указатель сама перечисляет разделы, без захода в её подпапки).'
                    ),
                },
            },
            'required': ['query'],
        },
        describe=lambda cfg: cfg.get('description') or (
            'Поиск по базе знаний документации. '
            'Возвращает релевантные чанки с текстом, контекстом и путём документа.'
        ),
        prompt_section=lambda cfg: '',
        handler=None,
        status=_search_status,
        icon='🔍',
    ),
    ToolSpec(
        type='find_section',
        default_name='find_section',
        core=True,
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Поисковый запрос на русском языке.',
                },
            },
            'required': ['query'],
        },
        describe=lambda cfg: cfg.get('description') or (
            _FIND_SECTION_DESC_REQUIRED if cfg.get('required', True)
            else _FIND_SECTION_DESC_OPTIONAL
        ),
        prompt_section=lambda cfg: '',
        handler=None,
        status=lambda cfg, args, _t: f"[{args.get('query', '')}] поиск раздела",
        icon='🗺️',
    ),
    ToolSpec(
        type='get_doc',
        default_name='get_doc',
        core=True,
        parameters={
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
        describe=lambda cfg: cfg.get('description') or (
            'Глубокое чтение одного документа: тянет все его чанки, реранкер '
            'выбирает релевантные query. Используй когда: (а) после search ты '
            'понимаешь что нужный документ найден, но один-два чанка из выдачи '
            'не дают полной картины; (б) нужно проверить все части большого '
            'документа на релевантность query (search мог пропустить '
            'релевантный фрагмент в хвосте документа).'
        ),
        prompt_section=lambda cfg: '',
        handler=None,
        status=lambda cfg, args, t: (
            f"[{args.get('query', '')}] глубокое чтение документа: {t(args.get('doc_id', ''))}"
        ),
        icon='📄',
    ),
]

