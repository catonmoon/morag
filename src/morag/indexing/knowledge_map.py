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


# Adaptive weighted-стратегия. Глубина — фиксированная: root → дети → стоп.
_KM_MAX_DEPTH = 2
# Минимальное число токенов на одну brief-строку «- Имя (id: …) — хинт».
_KM_BRIEF_LINE_TOKENS = 30


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


def _ratio_in_words(ratio: float) -> str:
    """Описание «во сколько раз короче» по-русски. Для prompt'а LLM-сжатия."""
    if ratio < 1.6:
        return 'примерно в полтора раза'
    if ratio < 2.5:
        return 'примерно вдвое'
    if ratio < 3.5:
        return 'примерно втрое'
    if ratio < 5:
        return 'примерно вчетверо'
    return f'примерно в {ratio:.0f} раз'


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
        node_min_tokens: int = 256,
        prompt_strategy: str = 'weighted',
        prompt_budget: int = 8192,
        token_counter: TokenCounter | None = None,
        concurrency: int = 4,
        exclude_source_types: list[str] | None = None,
        flat_topics_target: int | None = None,
        flat_topics_max_input_docs: int = 3000,
        flat_topics_assign_batch: int = 5,
    ) -> None:
        self._client = client
        self._llm_client = llm_client
        self._doc_repo = doc_repo
        self._collection = collection
        self._node_min_tokens = node_min_tokens
        self._prompt_strategy = prompt_strategy
        self._prompt_budget = prompt_budget
        self._counter = token_counter or TiktokenCounter()
        self._sem = asyncio.Semaphore(concurrency)
        self._exclude_source_types = frozenset(exclude_source_types or [])
        self._flat_topics_target = flat_topics_target
        self._flat_topics_max_input_docs = flat_topics_max_input_docs
        self._flat_topics_assign_batch = flat_topics_assign_batch

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
        all_ids = {d.id for d in all_docs}

        for doc in all_docs:
            for parent_id in doc.parent_doc_ids:
                if parent_id in all_ids:
                    children_map[parent_id].append(doc)

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
            'KnowledgeMap: %d docs, %d roots, strategy=%s',
            len(all_docs), len(roots), self._prompt_strategy,
        )

        system_prompt = await self._build_weighted_prompt(roots, children_map)
        maps: dict[str, str] = {'_system_prompt': system_prompt}
        logger.info(
            'KnowledgeMap: system prompt built (%d chars, %d tok)',
            len(system_prompt), self._counter.count(system_prompt),
        )

        # Сохраняем в Qdrant
        await self._save_maps(maps)

        logger.info('KnowledgeMap: generated %d maps + system prompt', len(maps) - 1)
        return maps

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
        self, title: str, summaries: str, max_tokens: int,
    ) -> str:
        """Обобщить пакет summary через LLM. max_tokens — guard rail, не cap;
        результат, при необходимости, доужимается через _compact_until_fits."""
        async with self._sem:
            prompt = _MAP_PROMPT.format(
                section_title=title,
                children_summaries=summaries,
            )
            try:
                result = await self._llm_client.complete(
                    [{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
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

    async def _build_weighted_prompt(
        self,
        roots: list[Document],
        children_map: dict[str, list[Document]],
    ) -> str:
        """Adaptive weighted-стратегия (top-down budget propagation).

        KM нужна агенту как навигационная карта. Хотим, чтобы крупные ветки
        получали больше места и разворачивались с подсекциями, а мелкие — шли
        строкой перечня. Решение «отдельный раздел или строка» — per-child по
        бюджету, без ручных списков.

        Алгоритм:
          1. Вес узла = сумма токенов doc_summary всех потомков (+ самого).
          2. Виртуальный root распределяет `prompt_budget` по группам
             (source_kind, source_name) пропорционально весам.
          3. Внутри узла: per-child пропорциональный бюджет от веса. Если
             ребёнку достаётся ≥ node_min_tokens — он становится отдельным
             разделом (`### Header + абзац + рекурсия`), иначе уходит в
             перечень одной строкой `- Имя (id) — короткий хинт`.
          4. Глубина зафиксирована = _KM_MAX_DEPTH=2. На depth=2 узел всегда
             в collapse-режиме (один summary subtree, дети не разворачиваются).
          5. Сжатие — через _compact_until_fits: LLM рекурсивно ужимает с
             варьируемой формулировкой, никаких truncate'ов.
        """
        weights = self._compute_summary_weights(roots, children_map)

        groups = self._group_roots_by_source(roots)
        multi_source = len(groups) > 1

        # Бюджет заголовка карты + опциональных source-разделителей
        overhead = self._counter.count('# Карта документации\n\n')
        if multi_source:
            overhead += sum(
                self._counter.count(self._format_source_header(k) + '\n\n')
                for k, _ in groups
            )
        available = max(self._prompt_budget - overhead, self._node_min_tokens)

        total_w = sum(weights[r.id] for r in roots) or 1

        lines: list[str] = ['# Карта документации\n']
        for key, group_roots in groups:
            if multi_source:
                lines.append(self._format_source_header(key))
                lines.append('')
            group_w = sum(weights[r.id] for r in group_roots) or 1
            group_budget = available * group_w // total_w
            for root in group_roots:
                root_budget = max(
                    group_budget * weights[root.id] // group_w,
                    self._node_min_tokens,
                )
                logger.info(
                    'KnowledgeMap: root %s weight=%d budget=%d',
                    root.id, weights[root.id], root_budget,
                )
                text = await self._render_node(
                    root, root_budget, depth=1,
                    children_map=children_map, weights=weights,
                )
                lines.append(text)

        result = '\n'.join(lines)
        result_tokens = self._counter.count(result)
        logger.info(
            'KnowledgeMap: system prompt %d tok (adaptive, %d source group(s))',
            result_tokens, len(groups),
        )
        return result

    def _compute_summary_weights(
        self,
        roots: list[Document],
        children_map: dict[str, list[Document]],
    ) -> dict[str, int]:
        """Вес узла = tokens(doc_summary) + Σ весов потомков. Минимум 1, чтобы
        пустые узлы не делали распределение нулевым."""
        weights: dict[str, int] = {}

        def walk(doc: Document) -> int:
            if doc.id in weights:
                return weights[doc.id]
            text = doc.payload.get('doc_summary', '') or _node_title(doc)
            own = max(self._counter.count(text), 1)
            for child in children_map.get(doc.id, []):
                own += walk(child)
            weights[doc.id] = own
            return own

        for r in roots:
            walk(r)
        return weights

    async def _render_node(
        self,
        doc: Document,
        budget: int,
        depth: int,
        children_map: dict[str, list[Document]],
        weights: dict[str, int],
    ) -> str:
        """Диспетчер. На максимальной глубине / для листа — collapse. Иначе —
        expandable, где per-child решается «отдельный раздел или строка перечня»."""
        children = children_map.get(doc.id, [])
        if depth >= _KM_MAX_DEPTH or not children:
            return await self._render_collapse(doc, budget, depth, children_map)
        return await self._render_expandable(doc, budget, depth, children_map, weights)

    async def _render_expandable(
        self,
        doc: Document,
        budget: int,
        depth: int,
        children_map: dict[str, list[Document]],
        weights: dict[str, int],
    ) -> str:
        """## Header + self-summary + микс развёрнутых разделов и brief-listing.

        Per-child решение по предварительному бюджету (пропорционально весу):
          - если ≥ _KM_BRIEF_LINE_TOKENS → разворачивается в свой раздел
          - если меньше → попадает в перечень `- Title (id) — short hint`
        Точный бюджет distribute'ится между big-разделами уже после учёта
        overhead'ов заголовков и строк listing'а.
        """
        children = children_map[doc.id]

        # Header + строки listing'а — фиксированный overhead (точная оценка по токенизатору)
        header_overhead = self._counter.count(f'{"#" * min(depth + 1, 4)} {_node_title(doc)} (id: {doc.id})\n\n')

        # 1. Provisional распределение между всеми детьми по весу
        total_w = sum(weights[c.id] for c in children) or 1
        provisional = {c.id: budget * weights[c.id] // total_w for c in children}

        # 2. Per-child big/brief. Порог — node_min_tokens: если ребёнку не
        # хватает даже на «минимально содержательный» абзац, ему нет смысла
        # быть отдельным разделом, идёт в перечень одной строкой.
        section_min = self._node_min_tokens
        big_children = [c for c in children if provisional[c.id] >= section_min]
        brief_children = [c for c in children if provisional[c.id] < section_min]

        # 3. Точный overhead и self-summary budget
        brief_overhead = len(brief_children) * _KM_BRIEF_LINE_TOKENS
        big_headers_overhead = len(big_children) * header_overhead
        # self-summary — фиксированная доля либо node_min_tokens (минимум на содержательный self).
        # Если big_children пустой — всё содержимое идёт в self-summary (узел с большим self + brief listing).
        self_budget = max(budget // (len(big_children) + 2), self._node_min_tokens)

        # 4. Бюджет на тела big-разделов = всё, что осталось после overheads
        big_pool = budget - header_overhead - self_budget - brief_overhead - big_headers_overhead
        if big_pool < 0:
            big_pool = 0

        # 5. Распределяем big_pool между big_children по весу
        big_total_w = sum(weights[c.id] for c in big_children) or 1
        big_budgets = {
            c.id: max(big_pool * weights[c.id] // big_total_w, self._node_min_tokens)
            for c in big_children
        }

        # 6. Рендер. Self + big — параллельно. Brief — параллельно отдельной группой.
        title = _node_title(doc)

        async def render_big(c: Document) -> str:
            return await self._render_node(c, big_budgets[c.id], depth + 1, children_map, weights)

        async def render_brief_line(c: Document) -> str:
            ctitle = _node_title(c)
            prefix_tokens = self._counter.count(f'- {ctitle} (id: {c.id}) — ')
            # На сам hint остаётся _KM_BRIEF_LINE_TOKENS - prefix
            hint_budget = max(_KM_BRIEF_LINE_TOKENS - prefix_tokens, 6)
            text = c.payload.get('doc_summary', '') or ''
            if not text:
                return f'- {ctitle} (id: {c.id})'
            hint = await self._compact_until_fits(ctitle, text, hint_budget)
            return f'- {ctitle} (id: {c.id}) — {hint}' if hint else f'- {ctitle} (id: {c.id})'

        self_text, big_texts, brief_lines = await asyncio.gather(
            self._compact_until_fits(title, doc.payload.get('doc_summary', '') or '', self_budget),
            asyncio.gather(*(render_big(c) for c in big_children)),
            asyncio.gather(*(render_brief_line(c) for c in brief_children)),
        )

        prefix = '#' * min(depth + 1, 4)
        parts = [f'{prefix} {title} (id: {doc.id})', '']
        if self_text:
            parts.extend([self_text, ''])
        parts.extend(big_texts)
        if brief_lines:
            parts.extend(brief_lines)
            parts.append('')
        return '\n'.join(parts).rstrip() + '\n'

    async def _render_collapse(
        self,
        doc: Document,
        budget: int,
        depth: int,
        children_map: dict[str, list[Document]],
    ) -> str:
        """## Header + один summary, покрывающий узел и весь subtree.

        Используется для листьев и для узлов на максимальной глубине.
        Никаких truncate'ов — только LLM-обобщение с recompact-петлёй.
        """
        children = children_map.get(doc.id, [])
        prefix = '#' * min(depth + 1, 4)
        title = _node_title(doc)

        if not children:
            desc = await self._compact_until_fits(
                title, doc.payload.get('doc_summary', '') or '', budget,
            )
            parts = [f'{prefix} {title} (id: {doc.id})', '']
            if desc:
                parts.extend([desc, ''])
            return '\n'.join(parts)

        # raw_map = doc_summary всех потомков, LLM обобщает в budget токенов.
        raw_lines: list[str] = []
        self._build_raw_map(children, children_map, raw_lines, heading_level=2)
        raw_map = '\n'.join(raw_lines)

        # max_tokens здесь — НЕ cap на размер summary, а либеральный guard
        # против runaway-генерации. LLM может выдать больше budget; за этим
        # пойдёт _compact_until_fits, который через recompact-петлю сожмёт
        # без обрезок.
        liberal_max = max(int(budget * 1.5) + 64, 256)
        input_budget = self._llm_client.context_window - 500 - liberal_max
        raw_tokens = self._counter.count(raw_map)
        if raw_tokens <= input_budget or input_budget <= 0:
            description = await self._summarize_batch(title, raw_map, max_tokens=liberal_max)
        else:
            logger.info(
                'KnowledgeMap: collapse %s — raw %d tok > %d available, batched',
                doc.id, raw_tokens, input_budget,
            )
            description = await self._iterative_summarize(
                title, raw_lines, input_budget, liberal_max,
            )
        # _summarize_batch с liberal_max почти всегда выдаст больше budget —
        # рекурсивно сжимаем через LLM до точного укладывания.
        description = await self._compact_until_fits(title, description, budget)

        parts = [f'{prefix} {title} (id: {doc.id})', '']
        if description:
            parts.extend([description, ''])
        return '\n'.join(parts)

    async def _compact_until_fits(
        self, title: str, text: str, target_tokens: int, max_iters: int = 4,
    ) -> str:
        """Сжать text до target_tokens через LLM. Никаких truncate'ов.

        Принцип:
          - Если текст уже ≤ target — возвращаем как есть.
          - Иначе LLM сжимает с формулировкой, варьируемой по итерации
            («вдвое короче» → «всё ещё длинно» → «оставь только главное» → …).
          - max_tokens в API ставим либерально — это техническая защита от
            runaway, а не cap на результат (иначе LLM режет на середине).
          - Если LLM не уложился — повтор с более жёсткой просьбой, пока
            (а) уложимся, (б) max_iters, (в) прогресса больше нет.
          - В корнере (LLM упёрся): возвращаем последний результат как есть —
            переборщить на N токенов лучше, чем потерять смысл обрезкой.
        """
        text = (text or '').strip()
        if not text:
            return ''
        current = self._counter.count(text)
        if current <= target_tokens:
            return text

        # Идём по всем итерациям до max_iters — каждая итерация использует
        # ДРУГУЮ формулировку promtа. Прерываемся только если последовательно
        # 2 итерации подряд не дали прогресса (упёрлись).
        stuck = 0
        best_text = text
        best_tokens = current
        for iter_idx in range(max_iters):
            new_text = await self._llm_compact_to_target(
                title, text, target_tokens, current, iter_idx,
            )
            new_tokens = self._counter.count(new_text)
            if new_tokens <= target_tokens:
                return new_text
            if new_tokens < best_tokens:
                best_text, best_tokens = new_text, new_tokens
            if new_tokens >= int(current * 0.95):
                stuck += 1
                if stuck >= 2:
                    logger.warning(
                        'KnowledgeMap: compact_until_fits «%s» не сходится '
                        '(%d → %d ток, цель %d, итер %d). Возвращаю наименьший.',
                        title, current, best_tokens, target_tokens, iter_idx,
                    )
                    return best_text
                # текст не меняем — пробуем другую формулировку на тех же данных
                continue
            stuck = 0
            text, current = new_text, new_tokens
        return best_text

    async def _llm_compact_to_target(
        self,
        title: str,
        text: str,
        target_tokens: int,
        current_tokens: int,
        iter_idx: int,
    ) -> str:
        """Один LLM-вызов сжатия. Формулировка варьируется по итерации, чтобы
        LLM попробовал разные стратегии. max_tokens — не cap, а guard rail."""
        ratio = max(current_tokens / max(target_tokens, 1), 1.3)
        ratio_word = _ratio_in_words(ratio)
        if iter_idx == 0:
            instruction = (
                f'Сожми описание раздела «{title}» {ratio_word} короче, '
                f'не теряя ключевых тем и сути.'
            )
        elif iter_idx == 1:
            instruction = (
                f'Описание раздела «{title}» всё ещё слишком длинное. '
                f'Перепиши его ещё вдвое короче, оставляя ключевые темы.'
            )
        elif iter_idx == 2:
            instruction = (
                f'Описание раздела «{title}» нужно ещё сократить. '
                f'Оставь только главное — 2–3 предложения о том, что в разделе.'
            )
        else:
            instruction = (
                f'Финальная попытка: опиши раздел «{title}» одним связным '
                f'абзацем длиной примерно {target_tokens} токенов. '
                f'Любые подробности можно опустить, но смысл сохрани.'
            )

        prompt = (
            f'{instruction} Уложись примерно в {target_tokens} токенов. '
            f'Закончи законченным предложением. Без markdown-заголовков. '
            f'Отвечай на языке оригинала.\n\n{text}'
        )
        # Либеральный max_tokens: даём LLM свободу нормально закончить мысль.
        # Если он выйдет за target — следующая итерация recompact'нёт. Не cap.
        max_tokens = max(int(target_tokens * 2) + 64, 256)
        async with self._sem:
            try:
                result = await self._llm_client.complete(
                    [{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    params=GenerationParams(enable_thinking=False),
                )
                return result.strip()
            except Exception:
                logger.exception('KnowledgeMap: compact LLM call failed for «%s»', title)
                return text

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
