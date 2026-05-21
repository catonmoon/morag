"""Генерация Knowledge Map — иерархической карты документации.

Обходит дерево документов снизу вверх (BFS по уровням) и для каждого узла
генерирует карту раздела из doc_summary потомков. Матрёшка: каждый уровень
обобщает уровень ниже.

Результат сохраняется в отдельную коллекцию Qdrant.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from morag.indexing.token_counter import TokenCounter, TiktokenCounter
from morag.llm.client import GenerationParams
from morag.sources.base import Document
from morag.storage.repository import DocRepository

logger = logging.getLogger(__name__)


_MAP_PROMPT = """\
You are a documentation analyst. Describe the contents of a documentation section \
in a descriptive narrative style.

Section: {section_title}

Below are summaries of documents in this section:

{children_summaries}

Write a descriptive overview of this section:
- Describe what this section covers and what information can be found here
- Mention key topics, tools, processes and roles covered
- Be specific: include names of systems, frameworks, teams where relevant
- Write in flowing prose, not as a list or table of contents
- Do NOT use markdown headings (#, ##, ###)
- Do NOT format as a bulleted list of subsections
- Use the same language as the original documents
"""


_COMPACT_PROMPT = """\
You are a documentation analyst. The following knowledge map layer is too long.

{layer_text}

Compact it while strictly preserving:
- ALL markdown headings (##, ###, ####) with their section ids
- The hierarchical structure
- Section ids in parentheses

Remove or shorten ONLY the descriptive text under headings. Keep each description to 1 sentence max.
Respond in the same language as the original.
"""


_SYSTEM_PROMPT_TEMPLATE = """\
You are a documentation analyst. Your task is to create a single, concise knowledge map \
that serves as a system prompt for a RAG assistant.

Below are knowledge maps of all documentation sections:

{section_maps}

Merge them into a single structured document that:
- Provides a clear overview of the entire knowledge base
- Groups related sections together
- For each section: 1-2 sentence description + id in parentheses
- Highlights what types of questions each section can answer
- Is concise enough to fit in a system prompt

Requirements:
- Use the same language as the original documents
- Format as markdown
- Include section ids for search filtering
- Total length should not exceed the available budget
"""


_SLUG_RE = re.compile(r'[^a-z0-9]+')


def _slugify(text: str, fallback: str = 'topic') -> str:
    """Простейший транслит в ascii-slug. Кириллица → транслит, остальное режется."""
    table = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z',
        'и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r',
        'с':'s','т':'t','у':'u','ф':'f','х':'h','ц':'c','ч':'ch','ш':'sh','щ':'sch',
        'ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
    }
    s = text.lower().strip()
    s = ''.join(table.get(ch, ch) for ch in s)
    s = _SLUG_RE.sub('_', s).strip('_')
    return s or fallback


def _first_sentence(s: str, max_chars: int = 240) -> str:
    """Вернуть первое предложение строки или первые max_chars символов."""
    s = (s or '').strip().replace('\n', ' ')
    if not s:
        return ''
    m = re.search(r'[.!?](?:\s|$)', s[:max_chars + 50])
    if m:
        return s[:m.end()].strip()
    return s[:max_chars].strip()


def _node_title(doc: Document, parent: Document | None = None) -> str:
    """Получить название узла: из поля title или fallback на path."""
    if doc.title:
        return doc.title
    if not doc.path:
        return doc.id
    full_path = doc.path[0]
    last_slash = full_path.rfind('/')
    return full_path[last_slash + 1:] if last_slash >= 0 else full_path


class KnowledgeMapGenerator:
    """Генератор карты документации по дереву документов."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        llm_client,
        doc_repo: DocRepository,
        collection: str = 'knowledge_map',
        depth: int = 2,
        max_depth: int | None = None,
        node_max_tokens: int = 256,
        node_min_tokens: int = 256,
        prompt_strategy: str = 'fixed',
        prompt_budget: int = 8192,
        token_counter: TokenCounter | None = None,
        concurrency: int = 4,
        exclude_source_types: list[str] | None = None,
        flat_topics_target: int | None = None,
        flat_topics_max_input_docs: int = 3000,
        flat_topics_assign_batch: int = 5,
        depth1_section_ids: list[str] | None = None,
        auto_depth1_children_threshold: int | None = None,
    ) -> None:
        self._client = client
        self._llm_client = llm_client
        self._doc_repo = doc_repo
        self._collection = collection
        self._depth = depth
        self._max_depth = max_depth  # None = до самого дна
        self._node_max_tokens = node_max_tokens
        self._node_min_tokens = node_min_tokens
        self._prompt_strategy = prompt_strategy
        self._prompt_budget = prompt_budget
        self._counter = token_counter or TiktokenCounter()
        self._sem = asyncio.Semaphore(concurrency)
        self._exclude_source_types = frozenset(exclude_source_types or [])
        self._flat_topics_target = flat_topics_target
        self._flat_topics_max_input_docs = flat_topics_max_input_docs
        self._flat_topics_assign_batch = flat_topics_assign_batch
        # _depth1_section_ids: явно указанные ID. Эффективный набор (с учётом
        # auto_threshold) формируется в generate() когда children_map известен.
        self._depth1_section_ids_explicit = frozenset(depth1_section_ids or [])
        self._depth1_section_ids: frozenset[str] = self._depth1_section_ids_explicit
        self._auto_depth1_threshold = auto_depth1_children_threshold

    async def ensure_collection(self) -> None:
        """Создать коллекцию для карт если не существует."""
        collections = await self._client.get_collections()
        names = {c.name for c in collections.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
            logger.info('Created collection: %s', self._collection)

    async def generate(self, root_ids: set[str] | None = None) -> dict[str, str]:
        """Сгенерировать карты для всех корневых разделов.

        Args:
            root_ids: явные id корневых разделов (ancestor_ids из конфига).
                      Если не заданы — корни определяются по отсутствию parent.

        Возвращает dict: doc_id → map_text.
        """
        await self.ensure_collection()

        # Собрать все документы из Qdrant (без вложений)
        all_docs = await self._load_all_docs()
        if self._exclude_source_types:
            before = len(all_docs)
            all_docs = [d for d in all_docs if d.source_type not in self._exclude_source_types]
            if before != len(all_docs):
                logger.info('KnowledgeMap: filtered %d → %d docs (excluded: %s)',
                            before, len(all_docs), ', '.join(self._exclude_source_types))
        if not all_docs:
            logger.warning('KnowledgeMap: no documents found')
            return {}

        # flat_topics: LLM-кластеризация плоского списка в темы.
        # Не использует существующий tree-building — собирает prompt напрямую.
        if self._prompt_strategy == 'flat_topics':
            map_text, membership = await self._build_flat_topics_prompt(all_docs)
            maps = {'_system_prompt': map_text}
            logger.info(
                'KnowledgeMap: flat_topics prompt built (%d chars, %d tok, %d clusters)',
                len(map_text), self._counter.count(map_text), len(membership),
            )
            await self._save_maps(
                maps,
                extra_points=[('_cluster_membership', {'cluster_membership': membership})],
            )
            return maps

        # Построить дерево
        children_map = defaultdict(list)  # parent_id → [child_docs]
        roots = []
        all_ids = {d.id for d in all_docs}

        for doc in all_docs:
            for parent_id in doc.parent_doc_ids:
                if parent_id in all_ids:
                    children_map[parent_id].append(doc)

        # Эффективный depth1_section_ids: явно перечисленные + auto-detected по
        # порогу количества детей. Auto-добавление логируется (видно в логе какие
        # секции свернулись, по сколько детей у каждой).
        effective = set(self._depth1_section_ids_explicit)
        if self._auto_depth1_threshold is not None:
            for pid, kids in children_map.items():
                if len(kids) > self._auto_depth1_threshold and pid not in effective:
                    effective.add(pid)
                    logger.info(
                        'KnowledgeMap auto-depth1: %s (%d children > threshold %d)',
                        pid, len(kids), self._auto_depth1_threshold,
                    )
        self._depth1_section_ids = frozenset(effective)

        # Корни: из конфига (ancestor_ids) или по отсутствию parent
        if root_ids:
            roots = [doc for doc in all_docs if doc.id in root_ids]
            seen = {d.id for d in roots}
            # Корень из конфига может быть не проиндексирован (структурная
            # страница, провал индексации). Важно не наличие самой страницы,
            # а её роль родителя — берём её детей как корни.
            for missing in root_ids - seen:
                children = [
                    d for d in all_docs
                    if missing in d.parent_doc_ids and d.id not in seen
                ]
                if children:
                    roots.extend(children)
                    seen.update(d.id for d in children)
                    logger.info(
                        'KnowledgeMap: root %s not indexed — using %d child doc(s) as roots',
                        missing, len(children),
                    )
                else:
                    logger.warning(
                        'KnowledgeMap: root %s not indexed and has no children — skipped',
                        missing,
                    )
        else:
            roots = [
                doc for doc in all_docs
                if not doc.parent_doc_ids or not any(p in all_ids for p in doc.parent_doc_ids)
            ]

        logger.info(
            'KnowledgeMap: %d docs, %d roots, depth=%d, max_depth=%s',
            len(all_docs), len(roots), self._depth, self._max_depth,
        )

        maps: dict[str, str] = {}

        if self._prompt_strategy != 'weighted':
            # fixed: генерируем карты узлов (для описаний в промпте)
            await asyncio.gather(
                *(self._generate_for_node(root, children_map, maps, depth=0) for root in roots),
            )

        # Собираем системный промпт
        system_prompt = await self._build_system_prompt_tree(roots, children_map, maps)
        maps['_system_prompt'] = system_prompt
        logger.info(
            'KnowledgeMap: system prompt built (%d chars, %d tok)',
            len(system_prompt), self._counter.count(system_prompt),
        )

        # Сохраняем в Qdrant
        await self._save_maps(maps)

        logger.info('KnowledgeMap: generated %d maps + system prompt', len(maps) - 1)
        return maps

    async def _generate_for_node(
        self,
        doc: Document,
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
        depth: int,
    ) -> str:
        """Сгенерировать карту для узла.

        Стратегия: собираем сырую карту из doc_summary всех потомков с h-заголовками.
        Если влезает в контекст LLM — один вызов (LLM видит всю картину).
        Если нет — fallback на пакетное обобщение (матрёшка).
        """
        children = children_map.get(doc.id, [])

        # Лист, достигли max_depth, или shallow-раздел — возвращаем doc_summary
        if (not children
                or (self._max_depth is not None and depth >= self._max_depth)
                or doc.id in self._depth1_section_ids):
            summary = doc.payload.get('doc_summary', '')
            title = _node_title(doc)
            return f'{title} (id: {doc.id}): {summary}' if summary else ''

        title = _node_title(doc)
        prompt_overhead = self._counter.count(
            _MAP_PROMPT.format(section_title='', children_summaries=''),
        )
        available = self._llm_client.context_window - prompt_overhead - self._node_max_tokens

        # Собираем сырую карту: h-заголовки + doc_summary всех потомков рекурсивно
        raw_lines: list[str] = []
        self._build_raw_map(children, children_map, raw_lines, heading_level=2)
        raw_map = '\n'.join(raw_lines)
        raw_tokens = self._counter.count(raw_map)

        if raw_tokens <= available:
            # Влезает — один LLM-вызов, LLM видит всю картину
            logger.debug('KnowledgeMap: node %s — raw map %d tok, single call', doc.id, raw_tokens)
            accumulated = await self._summarize_batch(title, raw_map, max_tokens=None)
        else:
            # Не влезает — fallback на пакетное обобщение
            logger.info(
                'KnowledgeMap: node %s — raw map %d tok > %d available, using batched fallback',
                doc.id, raw_tokens, available,
            )
            # Сначала обобщаем подуровни параллельно (матрёшка)
            results = await asyncio.gather(
                *(self._generate_for_node(child, children_map, maps, depth + 1)
                  for child in children),
            )
            children_texts = [r for r in results if r]

            accumulated = ''
            batch: list[str] = []
            batch_tokens = 0

            for child_text in children_texts:
                child_tokens = self._counter.count(child_text)
                acc_tokens = self._counter.count(accumulated) if accumulated else 0

                if batch and acc_tokens + batch_tokens + child_tokens > available:
                    batch_text = (accumulated + '\n\n' + '\n\n'.join(batch)
                                  if accumulated else '\n\n'.join(batch))
                    accumulated = await self._summarize_batch(title, batch_text)
                    batch = []
                    batch_tokens = 0

                batch.append(child_text)
                batch_tokens += child_tokens

            if batch:
                batch_text = (accumulated + '\n\n' + '\n\n'.join(batch)
                              if accumulated else '\n\n'.join(batch))
                accumulated = await self._summarize_batch(title, batch_text)

        if accumulated:
            map_text = f'# {title} (id: {doc.id})\n\n{accumulated}'
            maps[doc.id] = map_text
            logger.info('KnowledgeMap: generated map for %s (%d chars)', doc.id, len(map_text))
            return map_text

        return doc.payload.get('doc_summary', '')

    def _build_raw_map(
        self,
        docs: list[Document],
        children_map: dict[str, list[Document]],
        lines: list[str],
        heading_level: int,
    ) -> None:
        """Собрать сырую карту из doc_summary с h-заголовками рекурсивно."""
        for doc in docs:
            title = _node_title(doc)
            prefix = '#' * min(heading_level, 4)
            summary = doc.payload.get('doc_summary', '')

            if summary:
                lines.append(f'{prefix} {title} (id: {doc.id})')
                lines.append(summary)
                lines.append('')
            else:
                lines.append(f'{prefix} {title} (id: {doc.id})')
                lines.append('')

            children = children_map.get(doc.id, [])
            if children:
                self._build_raw_map(children, children_map, lines, heading_level + 1)

    async def _iterative_summarize(
        self,
        title: str,
        items: list[str],
        input_budget: int,
        output_budget: int,
    ) -> str:
        """Итеративное обобщение когда полный raw_map не влезает в context_window.

        Идём по `items` (строки raw_map), наполняем батч пока влезает в
        `input_budget - accumulated`. Когда батч полный — обобщаем
        `accumulated + batch` через LLM (max_tokens=output_budget), результат
        становится новым accumulated. После всех items возвращаем accumulated.

        Гарантия: каждый отдельный LLM-вызов укладывается в input_budget.
        """
        accumulated = ''
        batch_lines: list[str] = []
        batch_tokens = 0
        for item in items:
            it_tokens = self._counter.count(item)
            acc_tokens = self._counter.count(accumulated) if accumulated else 0
            if batch_lines and acc_tokens + batch_tokens + it_tokens > input_budget:
                batch_text = (accumulated + '\n\n' + '\n'.join(batch_lines)
                              if accumulated else '\n'.join(batch_lines))
                accumulated = await self._summarize_batch(
                    title, batch_text, max_tokens=output_budget,
                )
                batch_lines = []
                batch_tokens = 0
            batch_lines.append(item)
            batch_tokens += it_tokens
        if batch_lines:
            batch_text = (accumulated + '\n\n' + '\n'.join(batch_lines)
                          if accumulated else '\n'.join(batch_lines))
            accumulated = await self._summarize_batch(
                title, batch_text, max_tokens=output_budget,
            )
        return accumulated

    async def _summarize_batch(
        self, title: str, summaries: str, max_tokens: int | None = None,
    ) -> str:
        """Обобщить пакет summary через LLM."""
        async with self._sem:
            prompt = _MAP_PROMPT.format(
                section_title=title,
                children_summaries=summaries,
            )
            try:
                result = await self._llm_client.complete(
                    [{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens if max_tokens is not None else self._node_max_tokens,
                    params=GenerationParams(enable_thinking=False),
                )
                return result.strip()
            except Exception:
                logger.exception('KnowledgeMap: LLM failed for batch of %s', title)
                return summaries

    @staticmethod
    def _group_roots_by_source(
        roots: list[Document],
    ) -> list[tuple[tuple[str, str], list[Document]]]:
        """Сгруппировать корни по (source_kind, source_name) в порядке первого появления.

        Возвращает список (key, group). Если все корни из одного источника —
        список из одной группы; вызывающий смотрит на len, чтобы решить, рендерить
        заголовок группы или нет.

        У документов без source_kind/source_name (старые индексы до ADR-0012)
        key=('', '') — попадают в общую безымянную группу.
        """
        groups: dict[tuple[str, str], list[Document]] = {}
        order: list[tuple[str, str]] = []
        for doc in roots:
            key = (
                doc.payload.get('source_kind', '') or '',
                doc.payload.get('source_name', '') or '',
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(doc)
        return [(k, groups[k]) for k in order]

    @staticmethod
    def _format_source_header(key: tuple[str, str]) -> str:
        """Заголовок группы корней. Не h-заголовок — чтобы не сдвигать уровни
        существующих корней внутри группы."""
        kind, name = key
        kind_display = {'confluence': 'Confluence', 'jira': 'Jira', 'local': 'Локальные'}.get(
            kind, kind.capitalize() or 'Без источника',
        )
        if name:
            return f'**Источник: {kind_display} «{name}»**'
        return f'**Источник: {kind_display}**'

    async def _build_system_prompt_tree(
        self,
        roots: list[Document],
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
    ) -> str:
        """Программная сборка системного промпта.

        Структура точная из Qdrant: # корень → ## подраздел → ### подподраздел.
        Описания — doc_summary или LLM-обобщение, ужатые в бюджет.
        """
        if self._prompt_strategy == 'weighted':
            return await self._build_weighted_prompt(roots, children_map, maps)

        # fixed strategy
        lines: list[str] = ['# Карта документации\n']
        groups = self._group_roots_by_source(roots)
        multi_source = len(groups) > 1
        for key, group_roots in groups:
            if multi_source:
                lines.append(self._format_source_header(key))
                lines.append('')
            for root in group_roots:
                await self._append_node_to_prompt(
                    root, children_map, maps, lines, heading_level=1, parent=None,
                )

        result = '\n'.join(lines)
        result_tokens = self._counter.count(result)
        logger.info(
            'KnowledgeMap: system prompt %d tok (fixed, %d source group(s))',
            result_tokens, len(groups),
        )
        return result

    async def _build_weighted_prompt(
        self,
        roots: list[Document],
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
    ) -> str:
        """Weighted стратегия: общий бюджет распределяется по потомкам.

        Корни получают минимум (описаны через детей).
        Средний уровень — максимум, пропорционально числу скрытых потомков.
        """
        # 1. Собираем все узлы которые попадут в промпт с числом потомков
        nodes: list[tuple[Document, int, int]] = []  # (doc, heading_level, descendant_count)
        self._collect_prompt_nodes(roots, children_map, nodes, heading_level=1)

        # 2. Считаем overhead заголовков (плейсхолдеры)
        heading_overhead = sum(
            self._counter.count(
                '#' * min(hl + 1, 4) + ' '
                + _node_title(doc)
                + f' (id: {doc.id})\n',
            )
            for doc, hl, _ in nodes
        )
        header_line = self._counter.count('# Карта документации\n\n')
        available = self._prompt_budget - heading_overhead - header_line

        if available <= 0:
            logger.warning('KnowledgeMap: no budget for descriptions (%d overhead)', heading_overhead)
            available = 100

        # 3. Распределяем бюджет
        #    Корни с потомками — 0 (описаны через детей)
        #    Корни без потомков (leaf roots) — получают бюджет как обычные узлы
        #    Остальные — пропорционально числу потомков
        budgeted_nodes = [
            (doc, hl, dc) for doc, hl, dc in nodes
            if hl > 1 or not children_map.get(doc.id) or doc.id in self._depth1_section_ids
        ]
        total_weight = sum(dc + 1 for _, _, dc in budgeted_nodes) or 1

        budgets: dict[str, int] = {}
        for doc, hl, dc in nodes:
            has_children = bool(children_map.get(doc.id))
            is_depth1 = doc.id in self._depth1_section_ids
            if hl == 1 and has_children and not is_depth1:
                budgets[doc.id] = 0  # корни с потомками — без описания (дети опишут)
            else:
                weight = dc + 1
                budgets[doc.id] = max(self._node_min_tokens, available * weight // total_weight)

        logger.info(
            'KnowledgeMap: weighted budget: %d nodes (%d non-root), %d available tok, '
            'total_weight=%d',
            len(nodes), len(budgeted_nodes), available, total_weight,
        )

        # 4. Собираем промпт с бюджетами
        lines: list[str] = ['# Карта документации\n']
        groups = self._group_roots_by_source(roots)
        multi_source = len(groups) > 1
        for key, group_roots in groups:
            if multi_source:
                lines.append(self._format_source_header(key))
                lines.append('')
            for root in group_roots:
                await self._append_node_weighted(
                    root, children_map, maps, lines,
                    heading_level=1, budgets=budgets, parent=None,
                )

        result = '\n'.join(lines)
        result_tokens = self._counter.count(result)
        logger.info(
            'KnowledgeMap: system prompt %d tok (weighted, %d source group(s))',
            result_tokens, len(groups),
        )
        return result

    def _collect_prompt_nodes(
        self,
        docs: list[Document],
        children_map: dict[str, list[Document]],
        nodes: list[tuple[Document, int, int]],
        heading_level: int,
    ) -> None:
        """Собрать узлы для промпта с числом всех потомков (рекурсивно)."""
        for doc in docs:
            dc = self._count_all_descendants(doc.id, children_map)
            nodes.append((doc, heading_level, dc))
            if doc.id in self._depth1_section_ids:
                children_count = len(children_map.get(doc.id, []))
                logger.info(
                    'depth1_section_ids: skipping children of %s (%s), '
                    '%d direct children not expanded',
                    doc.id, _node_title(doc), children_count,
                )
                continue
            if doc.id not in self._depth1_section_ids:
                children = children_map.get(doc.id, [])
                if children and heading_level < self._depth:
                    self._collect_prompt_nodes(children, children_map, nodes, heading_level + 1)

    def _count_all_descendants(
        self, doc_id: str, children_map: dict[str, list[Document]],
    ) -> int:
        """Посчитать всех потомков рекурсивно (не только прямых)."""
        children = children_map.get(doc_id, [])
        count = len(children)
        for child in children:
            count += self._count_all_descendants(child.id, children_map)
        return count

    async def _append_node_weighted(
        self,
        doc: Document,
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
        lines: list[str],
        heading_level: int,
        budgets: dict[str, int],
        parent: Document | None = None,
    ) -> None:
        """Добавить узел с описанием в рамках выделенного бюджета.

        Для средних узлов (с потомками): собираем raw map из doc_summary потомков,
        LLM обобщает с max_tokens=budget. Без промежуточных карт.
        """
        title = _node_title(doc, parent)
        prefix = '#' * min(heading_level + 1, 4)
        budget = budgets.get(doc.id, 0)

        children = children_map.get(doc.id, [])

        if budget > 0 and children:
            # Средний узел: собираем raw map потомков → LLM обобщает.
            raw_lines: list[str] = []
            self._build_raw_map(children, children_map, raw_lines, heading_level=2)
            raw_map = '\n'.join(raw_lines)
            # Guard против 400/CUDA-OOM от модели: raw_map не должен превышать
            # context_window минус output_budget. Если raw_map влезает — один
            # вызов; если нет — iterative_summarize пакетами (накопитель +
            # батч → summary → новый накопитель). См. инцидент 2026-05
            # (cudaErrorLaunchFailure на ~60К input).
            input_budget = self._llm_client.context_window - 500 - budget
            raw_tokens = self._counter.count(raw_map)
            if raw_tokens <= input_budget or input_budget <= 0:
                description = await self._summarize_batch(title, raw_map, max_tokens=budget)
            else:
                logger.info(
                    'KnowledgeMap: weighted node %s — raw map %d tok > input_budget %d, batched',
                    doc.id, raw_tokens, input_budget,
                )
                description = await self._iterative_summarize(
                    title, raw_lines, input_budget, budget,
                )
        elif budget > 0:
            # Лист: doc_summary, при необходимости сжимаем
            description = doc.payload.get('doc_summary', '')
            if description and self._counter.count(description) > budget:
                description = await self._compact_description(title, description, max_tokens=budget)
        else:
            description = ''

        if budget > 0 and description:
            desc_tokens = self._counter.count(description)
            if desc_tokens > budget:
                description = await self._compact_description(title, description, max_tokens=budget)

            lines.append(f'{prefix} {title} (id: {doc.id})')
            lines.append(f'{description}\n')
        else:
            lines.append(f'{prefix} {title} (id: {doc.id})\n')

        if doc.id not in self._depth1_section_ids:
            children = children_map.get(doc.id, [])
            if children and heading_level < self._depth:
                for child in children:
                    await self._append_node_weighted(
                        child, children_map, maps, lines,
                        heading_level + 1, budgets=budgets, parent=doc,
                    )

    async def _append_node_to_prompt(
        self,
        doc: Document,
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
        lines: list[str],
        heading_level: int,
        parent: Document | None = None,
    ) -> None:
        """Добавить узел в системный промпт."""
        title = _node_title(doc, parent)
        prefix = '#' * min(heading_level + 1, 4)

        # Описание: из карты (LLM-обобщение) или doc_summary
        map_text = maps.get(doc.id, '')
        if map_text:
            description = self._extract_description(map_text)
        else:
            description = doc.payload.get('doc_summary', '')

        # Ужать если превышает max_tokens
        if description:
            desc_tokens = self._counter.count(description)
            if desc_tokens > self._node_max_tokens:
                description = await self._compact_description(title, description)

            lines.append(f'{prefix} {title} (id: {doc.id})')
            lines.append(f'{description}\n')
        else:
            lines.append(f'{prefix} {title} (id: {doc.id})\n')

        if doc.id not in self._depth1_section_ids:
            children = children_map.get(doc.id, [])
            if children and heading_level < self._depth:
                for child in children:
                    await self._append_node_to_prompt(
                        child, children_map, maps, lines, heading_level + 1, parent=doc,
                    )

    def _append_node(
        self,
        doc: Document,
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
        lines: list[str],
        heading_level: int,
    ) -> None:
        """Рекурсивно добавить узел в системный промпт с правильным уровнем заголовка."""
        title = _node_title(doc)
        prefix = '#' * min(heading_level + 1, 4)  # ## для корней, ### для детей, #### для внуков

        # Описание узла: берём из карты (LLM-обобщение) или doc_summary
        map_text = maps.get(doc.id, '')
        if map_text:
            # Извлекаем только описание (без заголовка карты — он дублируется)
            description = self._extract_description(map_text)
        else:
            description = doc.payload.get('doc_summary', '')

        if description:
            lines.append(f'{prefix} {title} (id: {doc.id})')
            lines.append(f'{description}\n')

        # Рекурсия по потомкам (до depth уровней в промпте)
        children = children_map.get(doc.id, [])
        if children and heading_level < self._depth:
            for child in children:
                self._append_node(child, children_map, maps, lines, heading_level + 1)

    async def _compact_description(
        self, title: str, description: str, max_tokens: int | None = None,
    ) -> str:
        """LLM ужимает описание узла, сохраняя суть."""
        budget = max_tokens or self._node_max_tokens
        async with self._sem:
            prompt = (
                f'Сожми описание раздела "{title}", сохранив максимум деталей. '
                f'Несколько предложений. Отвечай на языке оригинала.\n\n{description}'
            )
            try:
                result = await self._llm_client.complete(
                    [{'role': 'user', 'content': prompt}],
                    max_tokens=budget,
                    params=GenerationParams(enable_thinking=False),
                )
                return result.strip()
            except Exception:
                logger.warning('KnowledgeMap: compact failed for %s, keeping first sentence', title)
                # Fallback: первое предложение
                for sep in ['. ', '.\n']:
                    idx = description.find(sep)
                    if idx > 0:
                        return description[:idx + 1]
                return description

    @staticmethod
    def _extract_description(map_text: str) -> str:
        """Извлечь описание из карты, убрав все markdown заголовки.

        Структура задаётся программно, а не LLM — заголовки из карты не нужны.
        """
        lines = map_text.strip().split('\n')
        result = [line for line in lines if not line.startswith('#')]
        return '\n'.join(result).strip()

    async def _compact_prompt(
        self,
        roots: list[Document],
        children_map: dict[str, list[Document]],
        maps: dict[str, str],
    ) -> str:
        """Ужать системный промпт по слоям, сохраняя структуру.

        Для каждого корня: собрать его часть дерева → если > max_tokens/2 → LLM ужимает
        до max_tokens/2, сохраняя заголовки и id.
        """
        half_budget = self._node_max_tokens // 2
        budget_per_root = max(half_budget // len(roots), 256) if roots else half_budget

        compacted_parts: list[str] = []
        for root in roots:
            lines: list[str] = []
            self._append_node(root, children_map, maps, lines, heading_level=1)
            root_text = '\n'.join(lines)
            root_tokens = self._counter.count(root_text)

            if root_tokens > budget_per_root:
                logger.info(
                    'KnowledgeMap: compacting root %s (%d tok → %d budget)',
                    root.id, root_tokens, budget_per_root,
                )
                root_text = await self._compact_layer(root_text, budget_per_root)

            compacted_parts.append(root_text)

        return '# Карта документации\n\n' + '\n\n'.join(compacted_parts)

    async def _compact_layer(self, layer_text: str, target_tokens: int) -> str:
        """LLM ужимает слой до target_tokens, сохраняя структуру."""
        async with self._sem:
            prompt = _COMPACT_PROMPT.format(layer_text=layer_text)
            try:
                result = await self._llm_client.complete(
                    [{'role': 'user', 'content': prompt}],
                    max_tokens=target_tokens,
                    params=GenerationParams(enable_thinking=False),
                )
                return result.strip()
            except Exception:
                logger.exception('KnowledgeMap: compact failed, using truncated')
                return self._counter.truncate(layer_text, target_tokens)

    # ── flat_topics strategy ──────────────────────────────────────────────

    async def _build_flat_topics_prompt(
        self, docs: list[Document],
    ) -> tuple[str, dict[str, list[str]]]:
        """Построить system prompt и membership для плоского источника.

        Возвращает кортеж (map_text, cluster_membership), где
        cluster_membership — `{cluster_id: [doc_id, ...]}` для разворота
        section_ids в retrieval-слое.

        TODO: многоуровневая рекурсия — если какая-то тема содержит много
        документов, кластеризовать её саму вторым проходом. Для >3000 доков
        также нужен батчинг + merge-проход.
        """
        if len(docs) > self._flat_topics_max_input_docs:
            raise ValueError(
                f'flat_topics: {len(docs)} docs exceeds safety limit '
                f'{self._flat_topics_max_input_docs}. Implement batching or '
                f'raise flat_topics_max_input_docs.',
            )

        target_n = self._flat_topics_target or min(max(4, math.ceil(math.sqrt(len(docs)))), 40)

        # Детерминированный порядок для стабильных прогонов
        docs_sorted = sorted(docs, key=lambda d: d.id)
        clusters = await self._cluster_docs_llm(docs_sorted, target_n)
        map_text = self._render_flat_topics_prompt(clusters, docs_sorted)
        membership = {cl['id']: [d.id for d in cl['docs']] for cl in clusters}
        return map_text, membership

    async def _cluster_docs_llm(
        self, docs: list[Document], target_n: int,
    ) -> list[dict]:
        """Двухпроходная LLM-кластеризация.

        Проход 1: все заголовки+summary без id → LLM выделяет N тем (name + \
        summary). Маленький вывод, grok справляется надёжно.

        Проход 2: для каждого документа параллельно (батчами) LLM решает \
        в какую тему он входит. Каждый батч — изолированный контекст с \
        номерами тем; grok больше не тащит тысячу id.

        Возвращает: `[{'name', 'summary', 'docs': [Document, ...]}]`
        """
        topics = await self._generate_topics(docs, target_n)
        if not topics:
            logger.warning('flat_topics: LLM returned no topics, using single fallback cluster')
            return [{
                'name': 'Все документы',
                'summary': 'Темы не сгенерированы.',
                'docs': list(docs),
            }]

        assignments = await self._assign_docs_to_topics(docs, topics)

        # Собираем результат. Индекс -1 → fallback.
        clusters: list[dict] = [
            {'name': t['name'], 'summary': t['summary'], 'docs': []}
            for t in topics
        ]
        missing: list[Document] = []
        for doc, idx in zip(docs, assignments):
            if 0 <= idx < len(clusters):
                clusters[idx]['docs'].append(doc)
            else:
                missing.append(doc)

        # Отбросить пустые темы — LLM мог сгенерировать лишнее
        non_empty = [c for c in clusters if c['docs']]
        if missing:
            logger.warning(
                'flat_topics: %d/%d docs unassigned, placing into fallback cluster',
                len(missing), len(docs),
            )
            non_empty.append({
                'name': 'Без темы',
                'summary': 'Документы, которые не удалось отнести к конкретной тематике.',
                'docs': missing,
            })

        # Назначить уникальные slug-id для ссылок из section_ids
        used_ids: set[str] = set()
        for cl in non_empty:
            base = _slugify(cl['name'])
            cid = base
            i = 2
            while cid in used_ids:
                cid = f'{base}_{i}'
                i += 1
            used_ids.add(cid)
            cl['id'] = cid

        logger.info(
            'flat_topics: %d topics → %d non-empty clusters, %d docs placed',
            len(topics), len(non_empty), sum(len(c['docs']) for c in non_empty),
        )
        return non_empty

    async def _generate_topics(
        self, docs: list[Document], target_n: int,
    ) -> list[dict]:
        """Проход 1: по всем doc_summary сгенерировать N тем без id.

        Возвращает `[{'name': str, 'summary': str}]`.
        """
        n_docs = len(docs)
        min_n = max(4, target_n - 5)
        max_n = target_n + 10

        lines = []
        for d in docs:
            title = _node_title(d)
            summary = (d.payload.get('doc_summary') or '').strip().replace('\n', ' ')
            lines.append(f'- **{title}** — {summary}' if summary else f'- **{title}**')
        catalog = '\n'.join(lines)

        prompt = f"""\
На входе — {n_docs} документов (только title + первое предложение summary, \
без идентификаторов). Твоя задача: предложить **от {min_n} до {max_n}** \
навигационных тематик (оптимум ~{target_n}) так, чтобы по ним можно было \
решить «в какой теме искать ответ на мой вопрос».

Требования к темам:
- Каждая тема покрывает осмысленную группу документов; тем не слишком \
  много и не слишком мало.
- Избегай общих меток вроде «Разное», «Прочее», «Новости». Примеры \
  хороших названий: «Автомобильный рынок», «Госполитика и законы», \
  «Происшествия и суды», «Шоубиз и культура».
- Все темы должны быть взаимно-исключающими.

Для каждой темы верни:
- `name`: короткое название, 2–6 слов, по-русски
- `summary`: 1–2 предложения о том, что в теме лежит и какие вопросы она \
  покрывает

Идентификаторы документов в ответе НЕ нужны — мы раскидаем документы по \
темам отдельным проходом.

Список документов:
{catalog}
"""

        schema = {
            'type': 'object',
            'properties': {
                'topics': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string'},
                            'summary': {'type': 'string'},
                        },
                        'required': ['name', 'summary'],
                    },
                },
            },
            'required': ['topics'],
        }

        logger.info('flat_topics pass 1: generating ~%d topics from %d docs', target_n, n_docs)
        response = await self._llm_client.complete_json(
            messages=[{'role': 'user', 'content': prompt}],
            schema=schema,
            schema_name='topics',
            max_tokens=4096,
            params=GenerationParams(enable_thinking=False),
        )
        topics = []
        for t in response.get('topics') or []:
            name = (t.get('name') or '').strip()
            summary = (t.get('summary') or '').strip()
            if name:
                topics.append({'name': name, 'summary': summary})
        logger.info('flat_topics pass 1: got %d topics', len(topics))
        return topics

    async def _assign_docs_to_topics(
        self, docs: list[Document], topics: list[dict],
    ) -> list[int]:
        """Проход 2: для каждого документа вернуть индекс темы (0..N-1) или -1.

        Батчи по ~20 документов, параллельно через `self._sem`.
        """
        topics_block = '\n'.join(
            f'{i}. **{t["name"]}** — {t["summary"]}'
            for i, t in enumerate(topics)
        )
        n_topics = len(topics)
        batch_size = self._flat_topics_assign_batch

        async def _classify_batch(batch_idx: int, batch: list[Document]) -> list[int]:
            def _fmt(d: Document) -> str:
                title = _node_title(d)
                summary = (d.payload.get('doc_summary') or '').strip().replace('\n', ' ')
                return f'**{title}** — {summary}' if summary else f'**{title}**'
            items_block = '\n'.join(
                f'[{j}] {_fmt(d)}' for j, d in enumerate(batch)
            )
            prompt = f"""\
У тебя есть список из {n_topics} тематик:

{topics_block}

Ниже — {len(batch)} документ(ов). Для каждого определи номер темы \
(целое число от 0 до {n_topics - 1}), к которой он относится. Выбирай \
ровно одну тему на документ. Если документ очевидно не подходит ни под \
одну — всё равно выбери самую близкую.

Документы:
{items_block}

Верни массив `assignments` длины {len(batch)} — i-й элемент это индекс \
темы для i-го документа.
"""
            schema = {
                'type': 'object',
                'properties': {
                    'assignments': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                    },
                },
                'required': ['assignments'],
            }
            async with self._sem:
                try:
                    response = await self._llm_client.complete_json(
                        messages=[{'role': 'user', 'content': prompt}],
                        schema=schema,
                        schema_name='topic_assignments',
                        max_tokens=1024,
                        params=GenerationParams(enable_thinking=False),
                    )
                    raw = response.get('assignments') or []
                    # Нормализуем: усечь/дополнить до размера батча
                    out: list[int] = []
                    for k in range(len(batch)):
                        val = raw[k] if k < len(raw) else -1
                        if isinstance(val, int) and 0 <= val < n_topics:
                            out.append(val)
                        else:
                            out.append(-1)
                    return out
                except Exception:
                    logger.exception(
                        'flat_topics pass 2: batch %d classification failed', batch_idx,
                    )
                    return [-1] * len(batch)

        batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
        logger.info(
            'flat_topics pass 2: classifying %d docs in %d batches of %d',
            len(docs), len(batches), batch_size,
        )
        results = await asyncio.gather(
            *(_classify_batch(i, b) for i, b in enumerate(batches)),
        )
        # Плоский список в исходном порядке
        assignments: list[int] = []
        for r in results:
            assignments.extend(r)
        return assignments

    def _render_flat_topics_prompt(
        self, clusters: list[dict], docs: list[Document],
    ) -> str:
        """Собрать итоговый markdown из кластеров.

        Формат:
            # Карта документации

            ## {cluster.name}
            {cluster.summary}

            - **{doc.title}** (id: {doc.id})
            - ...

            ## {cluster2.name}
            ...
        """
        lines: list[str] = ['# Карта документации', '']
        for cl in clusters:
            cid = cl.get('id', '')
            header = f'## {cl["name"]}'
            if cid:
                header += f' (id: {cid})'
            lines.append(header)
            if cl.get('summary'):
                lines.append(cl['summary'])
            lines.append('')
            for doc in cl['docs']:
                title = _node_title(doc)
                lines.append(f'- **{title}** (id: {doc.id})')
            lines.append('')
        return '\n'.join(lines).rstrip() + '\n'

    # ── tree loading ──────────────────────────────────────────────────────

    async def _load_all_docs(self) -> list[Document]:
        """Загрузить все документы из Qdrant."""
        all_docs = []
        offset = None
        while True:
            results = await self._client.scroll(
                collection_name=self._doc_repo.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, offset = results
            for p in points:
                payload = p.payload
                all_docs.append(Document(
                    id=payload.get('id', ''),
                    path=payload.get('path', []),
                    text='',  # не нужен текст, только summary
                    updated_at=payload.get('updated_at', ''),
                    source_type=payload.get('source_type', ''),
                    title=payload.get('title'),
                    parent_doc_ids=payload.get('parent_doc_ids', []),
                    structural=payload.get('structural', False),
                    payload={
                        k: v for k, v in payload.items()
                        if k in ('doc_summary', 'source_kind', 'source_name')
                    },
                ))
            if offset is None:
                break
        return all_docs

    async def _save_maps(
        self,
        maps: dict[str, str],
        extra_points: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Сохранить карты и произвольные extra-точки в Qdrant.

        extra_points: список `(doc_id, payload)` для записи рядом с map-точками.
        Используется flat_topics для хранения membership.
        """
        if not maps and not extra_points:
            return

        # Очистить коллекцию
        try:
            await self._client.delete_collection(self._collection)
        except Exception:
            pass
        await self.ensure_collection()

        points = []
        for i, (doc_id, map_text) in enumerate(maps.items()):
            points.append(PointStruct(
                id=i,
                vector=[0.0],  # placeholder — карты не для поиска
                payload={
                    'doc_id': doc_id,
                    'map_text': map_text,
                },
            ))
        if extra_points:
            base = len(points)
            for i, (doc_id, payload) in enumerate(extra_points):
                points.append(PointStruct(
                    id=base + i,
                    vector=[0.0],
                    payload={'doc_id': doc_id, **payload},
                ))

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info(
            'KnowledgeMap: saved %d point(s) to %s',
            len(points), self._collection,
        )
