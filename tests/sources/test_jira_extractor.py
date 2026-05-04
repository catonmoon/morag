from __future__ import annotations

from datetime import datetime, timezone


from morag.sources.base import Document
from morag.sources.jira_extractor import JiraLinkExtractor


def make_doc(doc_id: str, text: str, path: list[str] | None = None) -> Document:
    return Document(
        id=doc_id,
        path=path or [doc_id],
        text=text,
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='confluence',
    )


JIRA_URL = 'https://jira.example.com'


class TestJiraLinkExtractor:
    def test_finds_single_issue(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('page', f'See [PROJ-123]({JIRA_URL}/browse/PROJ-123)')
        result = extractor.extract_from_docs([doc])
        assert 'PROJ-123' in result

    def test_finds_multiple_issues_in_one_doc(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        text = (
            f'Issue [{JIRA_URL}/browse/PROJ-1]({JIRA_URL}/browse/PROJ-1) and '
            f'[{JIRA_URL}/browse/PROJ-2]({JIRA_URL}/browse/PROJ-2)'
        )
        doc = make_doc('page', text)
        result = extractor.extract_from_docs([doc])
        assert 'PROJ-1' in result
        assert 'PROJ-2' in result

    def test_same_issue_in_two_docs(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc_a = make_doc('Engineering/Planning/Sprint', f'See {JIRA_URL}/browse/PROJ-42', path=['Engineering/Planning/Sprint'])
        doc_b = make_doc('Engineering/Team/Backlog', f'Also {JIRA_URL}/browse/PROJ-42', path=['Engineering/Team/Backlog'])
        result = extractor.extract_from_docs([doc_a, doc_b])
        assert result['PROJ-42'] == ['Engineering/Planning/Sprint', 'Engineering/Team/Backlog']

    def test_path_deduplication_same_doc(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        text = f'{JIRA_URL}/browse/PROJ-10 and again {JIRA_URL}/browse/PROJ-10'
        doc = make_doc('page', text, path=['docs/page'])
        result = extractor.extract_from_docs([doc])
        assert result['PROJ-10'] == ['docs/page']

    def test_no_issues_returns_empty(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('page', 'No Jira links here.')
        result = extractor.extract_from_docs([doc])
        assert result == {}

    def test_empty_doc_list(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        assert extractor.extract_from_docs([]) == {}

    def test_ignores_non_jira_urls(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('page', 'See https://other.com/browse/PROJ-123')
        result = extractor.extract_from_docs([doc])
        assert result == {}

    def test_uses_first_path_of_doc(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('id', f'{JIRA_URL}/browse/PROJ-5', path=['first/path', 'second/path'])
        result = extractor.extract_from_docs([doc])
        assert result['PROJ-5'] == ['first/path']

    def test_trailing_slash_in_base_url(self):
        extractor = JiraLinkExtractor('https://jira.example.com/')
        doc = make_doc('page', 'https://jira.example.com/browse/ABC-1')
        result = extractor.extract_from_docs([doc])
        assert 'ABC-1' in result

    def test_path_for_issue_is_doc_path_slash_key(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('id', f'{JIRA_URL}/browse/PROJ-7', path=['Team/Sprint'])
        result = extractor.extract_from_docs([doc])
        # extract_from_docs возвращает doc_path (не path задачи) — сборка пути в JiraSource
        assert result['PROJ-7'] == ['Team/Sprint']

    def test_issue_key_with_numbers_in_project(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc = make_doc('page', f'{JIRA_URL}/browse/PRJ2-100')
        result = extractor.extract_from_docs([doc])
        assert 'PRJ2-100' in result

    def test_docs_without_jira_do_not_affect_result(self):
        extractor = JiraLinkExtractor(JIRA_URL)
        doc_with = make_doc('a', f'{JIRA_URL}/browse/X-1')
        doc_without = make_doc('b', 'plain text')
        result = extractor.extract_from_docs([doc_with, doc_without])
        assert list(result.keys()) == ['X-1']
