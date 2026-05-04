from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from morag.config import JiraSourceConfig
from morag.sources.base import Document, Source
from morag.sources.jira import (
    JiraSource, _build_markdown, _extract_adf_text, _extract_custom_field_text,
    _extract_text, _parse_jira_date,
)


def _make_config(**kwargs) -> JiraSourceConfig:
    defaults = dict(
        kind='jira', name='test',
        url='https://jira.example.com', username='user', password='pass',
    )
    defaults.update(kwargs)
    return JiraSourceConfig(**defaults)


def _make_issue(key: str, summary: str = 'Test issue',
                updated: str = '2024-06-01T10:00:00.000+0000',
                created: str = '2024-01-01T10:00:00.000+0000',
                **extra_fields) -> dict:
    fields = {
        'summary': summary,
        'updated': updated,
        'created': created,
        'status': {'name': 'In Progress'},
        'priority': {'name': 'High'},
        'issuetype': {'name': 'Story'},
        'reporter': {'displayName': 'Jane Smith'},
        'assignee': {'displayName': 'John Doe'},
        'labels': [],
        'description': 'Issue description.',
        'subtasks': [],
        'issuelinks': [],
        'comment': {'comments': []},
    }
    fields.update(extra_fields)
    return {'key': key, 'fields': fields}


def _make_source(issue_map: dict, issues: list[dict]) -> JiraSource:
    issues_by_key = {i['key']: i for i in issues}
    with patch('morag.sources.jira.Jira') as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.issue.side_effect = lambda key, fields=None: issues_by_key.get(key)
        src = JiraSource(_make_config(), issue_map)
        src._client = mock_client
        return src


# ---------------------------------------------------------------------------
# JiraSource.__init__
# ---------------------------------------------------------------------------

class TestJiraSourceInit:
    def test_is_source(self):
        with patch('morag.sources.jira.Jira'):
            src = JiraSource(_make_config(), {})
            assert isinstance(src, Source)

    def test_password_required_at_pydantic_level(self):
        # Раньше credential-проверка жила в JiraSource.__init__. Теперь password
        # обязателен в Pydantic-схеме (JiraSourceConfig). Cloud-вариант
        # (api_token) удалён — only on-prem (см. ADR-0012).
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JiraSourceConfig(kind='jira', name='j', url='https://j.com', username='u')

    def test_no_cloud_flag(self):
        # Jira всегда on-prem — cloud=False всегда передаётся в SDK.
        with patch('morag.sources.jira.Jira') as mock_cls:
            JiraSource(_make_config(), {})
            _, kwargs = mock_cls.call_args
            assert kwargs['cloud'] is False


# ---------------------------------------------------------------------------
# JiraSource.get_metadata
# ---------------------------------------------------------------------------

class TestJiraSourceGetMetadata:
    async def test_returns_stubs(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        issue = _make_issue('PROJ-1')
        src = _make_source(issue_map, [issue])
        stubs = await src.get_metadata()
        assert len(stubs) == 1

    async def test_stub_text_is_empty(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        stubs = await src.get_metadata()
        assert stubs[0].text == ''

    async def test_stub_id_is_prefixed_issue_key(self):
        issue_map = {'PROJ-42': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-42')])
        stubs = await src.get_metadata()
        # _make_config использует name='test' → префикс jira:test:
        assert stubs[0].id == 'jira:test:PROJ-42'

    async def test_stub_source_type_is_attached_jira(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        stubs = await src.get_metadata()
        assert stubs[0].source_type == 'attached_jira'

    async def test_stub_path_includes_doc_path(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        stubs = await src.get_metadata()
        assert stubs[0].path == ['Team/Sprint/PROJ-1']

    async def test_stub_path_multiple_doc_paths(self):
        issue_map = {'PROJ-1': ['Team/Sprint', 'Team/Backlog']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        stubs = await src.get_metadata()
        assert stubs[0].path == ['Team/Sprint/PROJ-1', 'Team/Backlog/PROJ-1']

    async def test_stub_url_is_browse_link(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        stubs = await src.get_metadata()
        assert stubs[0].url == 'https://jira.example.com/browse/PROJ-1'

    async def test_empty_issue_map(self):
        src = _make_source({}, [])
        stubs = await src.get_metadata()
        assert stubs == []

    async def test_multiple_issues(self):
        issue_map = {'PROJ-1': ['Page/A'], 'PROJ-2': ['Page/B']}
        issues = [_make_issue('PROJ-1'), _make_issue('PROJ-2')]
        src = _make_source(issue_map, issues)
        stubs = await src.get_metadata()
        assert len(stubs) == 2


# ---------------------------------------------------------------------------
# JiraSource.load_one
# ---------------------------------------------------------------------------

class TestJiraSourceLoadOne:
    async def test_returns_document(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        doc = await src.load_one('PROJ-1')
        assert doc is not None
        assert isinstance(doc, Document)

    async def test_document_id_is_prefixed_issue_key(self):
        issue_map = {'PROJ-42': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-42')])
        # load_one принимает prefixed ID
        doc = await src.load_one('jira:test:PROJ-42')
        assert doc.id == 'jira:test:PROJ-42'

    async def test_document_source_type_is_attached_jira(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        doc = await src.load_one('PROJ-1')
        assert doc.source_type == 'attached_jira'

    async def test_document_text_contains_summary(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1', summary='My Task')])
        doc = await src.load_one('PROJ-1')
        assert 'My Task' in doc.text

    async def test_document_text_contains_description(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1', description='Detailed desc')])
        doc = await src.load_one('PROJ-1')
        assert 'Detailed desc' in doc.text

    async def test_document_path_is_doc_path_slash_key(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        doc = await src.load_one('PROJ-1')
        assert doc.path == ['Team/Sprint/PROJ-1']

    async def test_document_creator_from_reporter(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1', reporter={'displayName': 'Alice'})])
        doc = await src.load_one('PROJ-1')
        assert doc.creator == 'Alice'

    async def test_document_url_is_browse_link(self):
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1')])
        doc = await src.load_one('PROJ-1')
        assert doc.url == 'https://jira.example.com/browse/PROJ-1'

    async def test_returns_none_for_unknown_key(self):
        src = _make_source({}, [])
        doc = await src.load_one('UNKNOWN-1')
        assert doc is None

    async def test_document_contains_subtasks(self):
        subtasks = [{'key': 'PROJ-2', 'fields': {'summary': 'Subtask one'}}]
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1', subtasks=subtasks)])
        doc = await src.load_one('PROJ-1')
        assert 'PROJ-2' in doc.text
        assert 'Subtask one' in doc.text

    async def test_document_contains_comments(self):
        comments = [{'author': {'displayName': 'Bob'}, 'created': '2024-03-01T10:00:00.000+0000', 'body': 'Good point'}]
        issue_map = {'PROJ-1': ['Team/Sprint']}
        src = _make_source(issue_map, [_make_issue('PROJ-1', comment={'comments': comments})])
        doc = await src.load_one('PROJ-1')
        assert 'Good point' in doc.text
        assert 'Bob' in doc.text


# ---------------------------------------------------------------------------
# _parse_jira_date
# ---------------------------------------------------------------------------

class TestParseJiraDate:
    def test_iso_utc(self):
        dt = _parse_jira_date('2024-06-01T10:00:00.000+0000')
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_empty_returns_now(self):
        before = datetime.now(tz=timezone.utc)
        dt = _parse_jira_date('')
        after = datetime.now(tz=timezone.utc)
        assert before <= dt <= after

    def test_result_is_utc(self):
        dt = _parse_jira_date('2024-06-01T13:00:00.000+0300')
        assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _extract_text / _extract_adf_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_plain_string(self):
        assert _extract_text('hello') == 'hello'

    def test_empty_string(self):
        assert _extract_text('') == ''

    def test_none(self):
        assert _extract_text(None) == ''

    def test_adf_text_node(self):
        adf = {'type': 'text', 'text': 'world'}
        assert _extract_adf_text(adf) == 'world'

    def test_adf_paragraph(self):
        adf = {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'para text'}]}
        result = _extract_adf_text(adf)
        assert 'para text' in result

    def test_adf_nested(self):
        adf = {
            'type': 'doc',
            'content': [
                {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'first'}]},
                {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'second'}]},
            ],
        }
        result = _extract_adf_text(adf)
        assert 'first' in result
        assert 'second' in result


# ---------------------------------------------------------------------------
# _build_markdown
# ---------------------------------------------------------------------------

class TestBuildMarkdown:
    def test_starts_with_issue_key_and_summary(self):
        fields = _make_issue('PROJ-1', 'My Task')['fields']
        md = _build_markdown('PROJ-1', fields)
        assert md.startswith('# PROJ-1: My Task')

    def test_contains_status(self):
        fields = _make_issue('PROJ-1', status={'name': 'Done'})['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'Done' in md

    def test_contains_assignee(self):
        fields = _make_issue('PROJ-1', assignee={'displayName': 'Alice'})['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'Alice' in md

    def test_contains_description(self):
        fields = _make_issue('PROJ-1', description='Some description text')['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'Some description text' in md

    def test_contains_subtasks(self):
        subtasks = [{'key': 'PROJ-2', 'fields': {
            'summary': 'Child task',
            'status': {'name': 'In Progress'},
            'priority': {'name': 'High'},
            'issuetype': {'name': 'Sub-task'},
            'assignee': {'displayName': 'Bob'},
        }}]
        fields = _make_issue('PROJ-1', subtasks=subtasks)['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'PROJ-2' in md
        assert 'Child task' in md
        assert 'In Progress' in md
        assert 'High' in md
        assert 'Bob' in md

    def test_contains_issuelinks(self):
        links = [{'type': {'name': 'blocks', 'inward': 'is blocked by', 'outward': 'blocks'}, 'outwardIssue': {'key': 'PROJ-99', 'fields': {'summary': 'Blocked', 'status': {'name': 'Open'}, 'priority': {'name': 'High'}, 'issuetype': {'name': 'Bug'}, 'assignee': {'displayName': 'Carol'}}}}]
        fields = _make_issue('PROJ-1', issuelinks=links)['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'PROJ-99' in md
        assert 'blocks' in md
        assert 'Open' in md
        assert 'Carol' in md

    def test_contains_comments(self):
        comments = [{'author': {'displayName': 'Eve'}, 'created': '2024-01-01T00:00:00.000+0000', 'body': 'Nice work'}]
        fields = _make_issue('PROJ-1', comment={'comments': comments})['fields']
        md = _build_markdown('PROJ-1', fields)
        assert 'Nice work' in md
        assert 'Eve' in md

    def test_no_subtasks_section_when_empty(self):
        fields = _make_issue('PROJ-1', subtasks=[])['fields']
        md = _build_markdown('PROJ-1', fields)
        assert '## Подзадачи' not in md

    def test_no_comments_section_when_empty(self):
        fields = _make_issue('PROJ-1', comment={'comments': []})['fields']
        md = _build_markdown('PROJ-1', fields)
        assert '## Комментарии' not in md

    def test_custom_field_string(self):
        fields = _make_issue('PROJ-1', customfield_10100='Решение текстом')['fields']
        names = {'customfield_10100': 'Техническое решение'}
        md = _build_markdown('PROJ-1', fields, custom_field_names=names)
        assert '## Техническое решение' in md
        assert 'Решение текстом' in md

    def test_custom_field_select(self):
        fields = _make_issue('PROJ-1', customfield_10200={'value': 'Домен А'})['fields']
        names = {'customfield_10200': 'Домен Тема'}
        md = _build_markdown('PROJ-1', fields, custom_field_names=names)
        assert '## Домен Тема' in md
        assert 'Домен А' in md

    def test_custom_field_multi_select(self):
        fields = _make_issue('PROJ-1', customfield_10300=[{'value': 'A'}, {'value': 'B'}])['fields']
        names = {'customfield_10300': 'Категории'}
        md = _build_markdown('PROJ-1', fields, custom_field_names=names)
        assert '## Категории' in md
        assert 'A' in md
        assert 'B' in md

    def test_custom_field_empty_skipped(self):
        fields = _make_issue('PROJ-1', customfield_10100=None)['fields']
        names = {'customfield_10100': 'Пустое поле'}
        md = _build_markdown('PROJ-1', fields, custom_field_names=names)
        assert '## Пустое поле' not in md

    def test_custom_field_adf(self):
        adf = {'type': 'doc', 'content': [
            {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'ADF текст'}]},
        ]}
        fields = _make_issue('PROJ-1', customfield_10100=adf)['fields']
        names = {'customfield_10100': 'Описание решения'}
        md = _build_markdown('PROJ-1', fields, custom_field_names=names)
        assert '## Описание решения' in md
        assert 'ADF текст' in md

    def test_no_custom_fields_section_when_none(self):
        fields = _make_issue('PROJ-1')['fields']
        md = _build_markdown('PROJ-1', fields)
        # Без custom_field_names не должно быть лишних секций
        sections = [line for line in md.split('\n') if line.startswith('## ')]
        standard = {'## Описание', '## Подзадачи', '## Задачи эпика',
                    '## Связанные задачи', '## Комментарии'}
        for section in sections:
            assert section in standard


# ---------------------------------------------------------------------------
# _extract_custom_field_text
# ---------------------------------------------------------------------------

class TestExtractCustomFieldText:
    def test_string(self):
        assert _extract_custom_field_text('hello') == 'hello'

    def test_none(self):
        assert _extract_custom_field_text(None) == ''

    def test_number(self):
        assert _extract_custom_field_text(42) == '42'

    def test_select_dict(self):
        assert _extract_custom_field_text({'value': 'Option A'}) == 'Option A'

    def test_name_dict(self):
        assert _extract_custom_field_text({'name': 'Category'}) == 'Category'

    def test_adf_dict(self):
        adf = {'type': 'doc', 'content': [
            {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'hi'}]},
        ]}
        assert 'hi' in _extract_custom_field_text(adf)

    def test_multi_select_list(self):
        result = _extract_custom_field_text([{'value': 'A'}, {'value': 'B'}])
        assert 'A' in result
        assert 'B' in result

    def test_empty_list(self):
        assert _extract_custom_field_text([]) == ''
