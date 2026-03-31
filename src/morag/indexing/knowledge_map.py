"""Генерация Knowledge Map — иерархической карты документации.

Обходит дерево документов снизу вверх (BFS по уровням) и для каждого узла
генерирует карту раздела из doc_summary потомков. Матрёшка: каждый уровень
обобщает уровень ниже.

Результат сохраняется в отдельную коллекцию Qdrant.
"""
from __future__ import annotations

import asyncio
import logging
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


def _node_title(doc: Document, parent: Document | None = None) -> str:
    """Получить название узла: из поля title или fallback на path."""
    if doc.title:
        return doc.title
    if not doc.path:
        return doc.id
    full_path = doc.path[0]
    last_slash = full_path.rfind('/')
    return full_path[last_slash + 1:] if last_slash >= 0 else full_path


def _llm_params(enable_thinking: bool | None = None) -> GenerationParams:
    return GenerationParams(
        temperature=0.0, top_p=1.0, top_k=0,
        frequency_penalty=0.0, presence_penalty=0.0, seed=42,
        enable_thinking=enable_thinking,
    )


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
        context_window: int = 32768,
        enable_thinking: bool | None = None,
        concurrency: int = 4,
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
        self._context_window = context_window
        self._params = _llm_params(enable_thinking)
        self._sem = asyncio.Semaphore(concurrency)

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

        # Собрать все документы из Qdrant
        all_docs = await self._load_all_docs()
        if not all_docs:
            logger.warning('KnowledgeMap: no documents found')
            return {}

        # Построить дерево
        children_map = defaultdict(list)  # parent_id → [child_docs]
        roots = []
        all_ids = {d.id for d in all_docs}

        for doc in all_docs:
            for parent_id in doc.parent_doc_ids:
                if parent_id in all_ids:
                    children_map[parent_id].append(doc)

        # Корни: из конфига (ancestor_ids) или по отсутствию parent
        if root_ids:
            roots = [doc for doc in all_docs if doc.id in root_ids]
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

        # Лист или достигли max_depth (если задан) — возвращаем doc_summary
        if not children or (self._max_depth is not None and depth >= self._max_depth):
            summary = doc.payload.get('doc_summary', '')
            title = doc.payload.get('title', _node_title(doc))
            return f'{title} (id: {doc.id}): {summary}' if summary else ''

        title = _node_title(doc)
        prompt_overhead = self._counter.count(
            _MAP_PROMPT.format(section_title='', children_summaries=''),
        )
        available = self._context_window - prompt_overhead - self._node_max_tokens

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
                    params=self._params,
                    max_tokens=max_tokens if max_tokens is not None else self._node_max_tokens,
                )
                return result.strip()
            except Exception:
                logger.exception('KnowledgeMap: LLM failed for batch of %s', title)
                return summaries

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
        for root in roots:
            await self._append_node_to_prompt(
                root, children_map, maps, lines, heading_level=1, parent=None,
            )

        result = '\n'.join(lines)
        result_tokens = self._counter.count(result)
        logger.info('KnowledgeMap: system prompt %d tok (fixed)', result_tokens)
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
        #    Корни (heading_level=1) — минимум (50 tok)
        #    Остальные — пропорционально числу потомков
        # Корни = 0 бюджета (только заголовок, описаны через детей)
        non_root_nodes = [(doc, hl, dc) for doc, hl, dc in nodes if hl > 1]
        total_weight = sum(dc + 1 for _, _, dc in non_root_nodes) or 1

        budgets: dict[str, int] = {}
        for doc, hl, dc in nodes:
            if hl == 1:
                budgets[doc.id] = 0  # корни без описания
            else:
                weight = dc + 1
                budgets[doc.id] = max(self._node_min_tokens, available * weight // total_weight)

        logger.info(
            'KnowledgeMap: weighted budget: %d nodes (%d non-root), %d available tok, '
            'total_weight=%d',
            len(nodes), len(non_root_nodes), available, total_weight,
        )

        # 4. Собираем промпт с бюджетами
        lines: list[str] = ['# Карта документации\n']
        for root in roots:
            await self._append_node_weighted(
                root, children_map, maps, lines,
                heading_level=1, budgets=budgets, parent=None,
            )

        result = '\n'.join(lines)
        result_tokens = self._counter.count(result)
        logger.info('KnowledgeMap: system prompt %d tok (weighted)', result_tokens)
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
            # Средний узел: собираем raw map потомков → LLM обобщает
            raw_lines: list[str] = []
            self._build_raw_map(children, children_map, raw_lines, heading_level=2)
            raw_map = '\n'.join(raw_lines)
            description = await self._summarize_batch(title, raw_map, max_tokens=budget)
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

        # Потомки до depth
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
                    params=self._params,
                    max_tokens=budget,
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
                    params=self._params,
                    max_tokens=target_tokens,
                )
                return result.strip()
            except Exception:
                logger.exception('KnowledgeMap: compact failed, using truncated')
                return self._counter.truncate(layer_text, target_tokens)

    async def _load_all_docs(self) -> list[Document]:
        """Загрузить все документы из Qdrant."""
        all_docs = []
        offset = None
        while True:
            results = await self._client.scroll(
                collection_name='docs',
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
                    parent_doc_ids=payload.get('parent_doc_ids', []),
                    structural=payload.get('structural', False),
                    payload={
                        k: v for k, v in payload.items()
                        if k in ('doc_summary', 'title')
                    },
                ))
            if offset is None:
                break
        return all_docs

    async def _save_maps(self, maps: dict[str, str]) -> None:
        """Сохранить карты в Qdrant."""
        if not maps:
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

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info('KnowledgeMap: saved %d maps to %s', len(points), self._collection)
