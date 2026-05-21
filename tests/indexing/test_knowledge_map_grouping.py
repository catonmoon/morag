"""Юнит-тесты на группировку root-секций KM по источникам."""
from datetime import datetime, timezone

from morag.indexing.knowledge_map import KnowledgeMapGenerator
from morag.sources.base import Document


def _make_doc(doc_id: str, source_kind: str = '', source_name: str = '') -> Document:
    payload = {}
    if source_kind:
        payload['source_kind'] = source_kind
    if source_name:
        payload['source_name'] = source_name
    return Document(
        id=doc_id,
        path=[doc_id],
        text='',
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_type=source_kind or 'unknown',
        payload=payload,
    )


class TestGroupRootsBySource:

    def test_single_source_one_group(self):
        roots = [
            _make_doc('confluence:corp:1', 'confluence', 'corp'),
            _make_doc('confluence:corp:2', 'confluence', 'corp'),
        ]
        groups = KnowledgeMapGenerator._group_roots_by_source(roots)
        assert len(groups) == 1
        assert groups[0][0] == ('confluence', 'corp')
        assert [d.id for d in groups[0][1]] == ['confluence:corp:1', 'confluence:corp:2']

    def test_two_confluences_two_groups(self):
        roots = [
            _make_doc('confluence:corp:1', 'confluence', 'corp'),
            _make_doc('confluence:vendor:9', 'confluence', 'vendor'),
            _make_doc('confluence:corp:2', 'confluence', 'corp'),
        ]
        groups = KnowledgeMapGenerator._group_roots_by_source(roots)
        keys = [k for k, _ in groups]
        # порядок появления, не алфавит
        assert keys == [('confluence', 'corp'), ('confluence', 'vendor')]
        assert [d.id for d in groups[0][1]] == ['confluence:corp:1', 'confluence:corp:2']
        assert [d.id for d in groups[1][1]] == ['confluence:vendor:9']

    def test_mixed_kinds_grouped_separately(self):
        roots = [
            _make_doc('confluence:corp:1', 'confluence', 'corp'),
            _make_doc('jira:internal:PROJ-1', 'jira', 'internal'),
        ]
        groups = KnowledgeMapGenerator._group_roots_by_source(roots)
        assert len(groups) == 2
        assert {k for k, _ in groups} == {('confluence', 'corp'), ('jira', 'internal')}

    def test_legacy_docs_without_source_payload(self):
        """Документы без source_kind/source_name (старый индекс до ADR-0012)
        попадают в безымянную группу ('', '')."""
        roots = [_make_doc('some:legacy:id'), _make_doc('other:legacy:id')]
        groups = KnowledgeMapGenerator._group_roots_by_source(roots)
        assert len(groups) == 1
        assert groups[0][0] == ('', '')


class TestFormatSourceHeader:

    def test_confluence_named(self):
        assert KnowledgeMapGenerator._format_source_header(('confluence', 'corp')) == \
            '**Источник: Confluence «corp»**'

    def test_jira_named(self):
        assert KnowledgeMapGenerator._format_source_header(('jira', 'internal')) == \
            '**Источник: Jira «internal»**'

    def test_local_named(self):
        assert KnowledgeMapGenerator._format_source_header(('local', 'docs')) == \
            '**Источник: Локальные «docs»**'

    def test_unknown_kind_capitalized(self):
        assert KnowledgeMapGenerator._format_source_header(('webhook', 'main')) == \
            '**Источник: Webhook «main»**'

    def test_empty_kind_falls_back(self):
        # legacy doc без source_kind — заголовок всё равно валидный
        assert KnowledgeMapGenerator._format_source_header(('', '')) == \
            '**Источник: Без источника**'
