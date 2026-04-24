"""Section-level retrieval: найти РАЗДЕЛЫ документации для запроса.

Работает поверх `HybridSearcher.search_docs`. Алгоритм:

1. Top-N документов по doc-level эмбеддингам (полный текст документа).
2. Vote counting: каждый документ голосует за immediate in-corpus parent.
3. Adaptive descent: если ребёнок секции покрывает ≥threshold*votes — спускаемся.
4. Self-vote break: если сама страница-секция в voting_docs → kind='doc'
   (потребитель передаёт как `doc_ids`, не `section_ids`).
5. Top-K safety: добавляем top-N документов по прямому score как `doc_ids` —
   страховка от «одинокого чемпиона» проигрывающего в vote counting.

Модуль OWUI/pipelines-независимый. Возвращает `SectionResult` dataclass.
Потребитель форматирует его как угодно (markdown для LLM-агента, JSON
для API, plain text для CLI).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from morag.retrieval.searcher import HybridSearcher


@dataclass
class FindSectionConfig:
    """Параметры find_section. Все опционально — есть разумные defaults."""

    sections_limit: int = 5          # макс секций в выдаче
    doc_pool: int = 20               # сколько документов берём из search_docs
    descent_threshold: float = 0.5   # 0..1; 0 отключает descent
    top_docs: int = 3                # top-K safety docs добавляемые в doc_ids


@dataclass
class SectionEntry:
    """Запись в результате find_section.

    kind='section' → добавить в search.section_ids (рекурсивный поиск).
    kind='doc'     → добавить в search.doc_ids (точечный поиск без потомков).
    """

    section_id: str
    title: str
    kind: str  # 'section' | 'doc'
    votes: int
    summary: str


@dataclass
class ExtraDoc:
    """Top-K safety: документ с высоким score, добавлен как doc_ids."""

    doc_id: str
    title: str
    score: float
    summary: str


@dataclass
class SectionResult:
    """Готовый результат find_section.

    section_ids/doc_ids — готовые списки для передачи в search. Если оба
    пустые — find_section не смог ничего найти (смотри .error)."""

    section_ids: list[str]
    doc_ids: list[str]
    refined: list[SectionEntry] = field(default_factory=list)
    extra_docs: list[ExtraDoc] = field(default_factory=list)
    error: str | None = None


def aggregate_to_sections(
    docs: list[dict],
    valid_doc_ids: set[str] | None = None,
) -> list[tuple[str, list[dict]]]:
    """Vote counting: группируем документы по immediate in-corpus parent.

    parent_doc_ids в Confluence-docs содержит цепочку ВСЕХ предков, включая
    страницы выше настроенных ancestor_ids (out-of-corpus). Такие parent_id
    в нашей коллекции отсутствуют — идём от immediate parent вверх и берём
    первого предка в `valid_doc_ids`. Если ни один — документ сам становится
    секцией (типично для root-страниц типа «Люди»).

    Возвращает список пар (section_id, [voting_docs...]) отсортированный по
    votes desc, затем по сумме score desc.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        parents = d.get('parent_doc_ids') or []
        section_id: str | None = None
        if valid_doc_ids is not None:
            for pid in reversed(parents):
                if pid in valid_doc_ids:
                    section_id = pid
                    break
        elif parents:
            section_id = parents[-1]
        if section_id is None:
            section_id = d.get('doc_id', '')
        if not section_id:
            continue
        buckets[section_id].append(d)
    items = list(buckets.items())
    items.sort(
        key=lambda kv: (-len(kv[1]), -sum(d.get('score', 0.0) for d in kv[1])),
    )
    return items


def descend_section(
    section_id: str,
    voting_docs: list[dict],
    tree: dict[str, list[str]],
    threshold: float,
) -> tuple[str, bool]:
    """Adaptive descent: если ребёнок секции покрывает ≥threshold*votes — спускаемся.

    Returns (final_section_id, self_voted):
    - self_voted=True: сама страница-секция в voting_docs → её текст релевантен,
      передавать как `doc_ids` (без потомков). descent в ребёнка вырезал бы
      родительскую страницу из scope.
    - self_voted=False: секция-контейнер, voting_docs — её потомки.
      Передавать как `section_ids` (рекурсивно).
    """
    current = section_id
    current_docs = voting_docs
    while True:
        self_voted = any(d.get('doc_id') == current for d in current_docs)
        if self_voted:
            return current, True
        if len(current_docs) < 2:
            return current, False
        children = tree.get(current, [])
        if not children:
            return current, False
        best_child: str | None = None
        best_docs: list[dict] = []
        for child in children:
            matched = [
                d for d in current_docs
                if child in (d.get('parent_doc_ids') or [])
                or d.get('doc_id') == child
            ]
            if len(matched) > len(best_docs):
                best_child = child
                best_docs = matched
        if best_child is None or len(best_docs) < threshold * len(current_docs):
            return current, False
        current = best_child
        current_docs = best_docs


async def find_section(
    query: str,
    searcher: 'HybridSearcher',
    config: FindSectionConfig | None = None,
) -> SectionResult:
    """Найти релевантные разделы документации для запроса.

    Полный pipeline: search_docs → aggregate → descend → top-K safety → enrich.
    Возвращает `SectionResult` готовый к отображению или передаче в search.
    """
    cfg = config or FindSectionConfig()
    docs = await searcher.search_docs(query, cfg.doc_pool)
    if not docs:
        return SectionResult(section_ids=[], doc_ids=[], error='no_docs')

    valid_doc_ids = await searcher.get_indexed_doc_ids()
    aggregated = aggregate_to_sections(docs, valid_doc_ids=valid_doc_ids)
    if not aggregated:
        return SectionResult(section_ids=[], doc_ids=[], error='no_sections')

    tree, _ = await searcher.build_doc_tree()
    refined_raw: list[tuple[str, int, str]] = []  # (id, votes, kind)
    seen: set[str] = set()
    for sid, section_docs in aggregated:
        if cfg.descent_threshold > 0:
            final_sid, self_voted = descend_section(
                sid, section_docs, tree, cfg.descent_threshold,
            )
        else:
            final_sid = sid
            self_voted = any(d.get('doc_id') == sid for d in section_docs)
        if final_sid in seen:
            continue
        seen.add(final_sid)
        kind = 'doc' if self_voted else 'section'
        refined_raw.append((final_sid, len(section_docs), kind))
        if len(refined_raw) >= cfg.sections_limit:
            break

    # Top-K safety: страховка от «одинокого чемпиона» — добавляем top-N docs
    # по прямому score, которых нет в refined.
    refined_ids = {sid for sid, _, _ in refined_raw}
    extra_raw: list[dict] = []
    for d in docs:
        if len(extra_raw) >= cfg.top_docs:
            break
        did = d.get('doc_id')
        if not did or did in refined_ids:
            continue
        extra_raw.append(d)
        refined_ids.add(did)

    # Обогащение: title + doc_summary
    all_ids = [sid for sid, _, _ in refined_raw] + [d['doc_id'] for d in extra_raw]
    summaries = await searcher.fetch_doc_summaries(all_ids)

    refined: list[SectionEntry] = []
    for sid, votes, kind in refined_raw:
        title = await searcher.get_doc_title(sid)
        summary = (summaries.get(sid) or '').strip()
        refined.append(SectionEntry(
            section_id=sid, title=title, kind=kind,
            votes=votes, summary=summary,
        ))

    extra_docs: list[ExtraDoc] = [
        ExtraDoc(
            doc_id=d['doc_id'],
            title=d.get('title') or d['doc_id'],
            score=d.get('score', 0.0),
            summary=(summaries.get(d['doc_id']) or '').strip(),
        )
        for d in extra_raw
    ]

    section_ids = [e.section_id for e in refined if e.kind == 'section']
    doc_ids = [e.section_id for e in refined if e.kind == 'doc']
    doc_ids.extend(e.doc_id for e in extra_docs)

    return SectionResult(
        section_ids=section_ids, doc_ids=doc_ids,
        refined=refined, extra_docs=extra_docs,
    )
