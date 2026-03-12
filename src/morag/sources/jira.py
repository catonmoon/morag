from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from atlassian import Jira

from morag.config import JiraConfig
from morag.sources.base import Document, Source

logger = logging.getLogger(__name__)

# Поля задачи, запрашиваемые при загрузке метаданных (стаб)
_STUB_FIELDS = 'summary,updated'

# Поля задачи, запрашиваемые при полной загрузке
_FULL_FIELDS = (
    'summary,description,status,priority,issuetype,assignee,reporter,'
    'created,updated,labels,comment,subtasks,issuelinks'
)


class JiraSource(Source):
    """Источник задач Jira, обнаруженных в других проиндексированных документах.

    Принимает issue_map — маппинг {issue_key: [doc_path, ...]} — построенный
    JiraLinkExtractor по уже проиндексированным документам.

    Путь каждой задачи формируется как ['{doc_path}/{issue_key}', ...] — по одному
    пути на каждый документ, в котором задача упоминается. Это позволяет находить
    задачи в поиске в контексте документа, где они были упомянуты.

    Поддерживает on-premise (username + password) и Cloud (username + api_token).
    """

    @property
    def source_type(self) -> str:
        return 'attached_jira'

    def __init__(
        self,
        config: JiraConfig,
        issue_map: dict[str, list[str]],
        parent_ids_map: dict[str, list[str]] | None = None,
    ) -> None:
        credential = config.api_token or config.password
        if not credential:
            raise ValueError('Jira config requires either api_token or password')

        self._client = Jira(
            url=config.url,
            username=config.username,
            password=credential,
            cloud=config.api_token is not None,
            timeout=config.timeout,
            backoff_and_retry=config.max_retries > 0,
            max_backoff_retries=config.max_retries,
        )
        self._base_url = config.url.rstrip('/')
        self._issue_map = issue_map         # {issue_key: [doc_path, ...]}
        self._parent_ids_map = parent_ids_map or {}  # {issue_key: [doc_id, ...]}
        self._timeout = config.timeout
        self._custom_fields = config.custom_fields  # ['customfield_10100', ...]
        self._field_names: dict[str, str] = {}  # кеш: field_id → display_name

    async def _ensure_field_names(self) -> None:
        """Загрузить маппинг field_id → display_name (один раз)."""
        if self._field_names or not self._custom_fields:
            return
        try:
            all_fields = await asyncio.to_thread(self._client.get_all_fields)
            self._field_names = {f['id']: f['name'] for f in all_fields}
            logger.info(
                'Loaded %d field name(s) from Jira, custom fields: %s',
                len(self._field_names),
                ', '.join(
                    f'{fid} -> {self._field_names.get(fid, "?")}'
                    for fid in self._custom_fields
                ),
            )
        except Exception:
            logger.exception('Failed to load Jira field names')

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы задач: только key, summary, updated_at, без описания и комментариев."""
        await self._ensure_field_names()
        stubs: list[Document] = []
        for issue_key, doc_paths in self._issue_map.items():
            try:
                stub = await self._fetch_stub(issue_key, doc_paths)
                if stub is not None:
                    stubs.append(stub)
            except Exception:
                logger.exception('Failed to get metadata for Jira issue %s', issue_key)
        logger.info('Fetched metadata for %d Jira issue(s)', len(stubs))
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить одну задачу Jira целиком по ключу (doc_id == issue_key)."""
        doc_paths = self._issue_map.get(doc_id)
        if doc_paths is None:
            logger.warning('Jira issue %s not found in issue_map', doc_id)
            return None
        try:
            return await self._fetch_full(doc_id, doc_paths)
        except Exception:
            logger.exception('Failed to load Jira issue %s', doc_id)
            return None

    async def _fetch_stub(self, issue_key: str, doc_paths: list[str]) -> Document | None:
        """Получить стаб задачи: только метаданные без тела."""
        issue = await asyncio.to_thread(
            self._client.issue, issue_key, fields=_STUB_FIELDS,
        )
        fields = issue.get('fields', {})
        updated_at = _parse_jira_date(fields.get('updated', ''))
        path = [f'{dp}/{issue_key}' for dp in doc_paths]

        return Document(
            id=issue_key,
            path=path,
            text='',
            updated_at=updated_at,
            source_type='attached_jira',
            size=0,
            url=f'{self._base_url}/browse/{issue_key}',
            parent_doc_ids=self._parent_ids_map.get(issue_key, []),
        )

    async def _fetch_full(self, issue_key: str, doc_paths: list[str]) -> Document | None:
        """Получить полную задачу и сконвертировать в Document."""
        fields_to_request = _FULL_FIELDS
        if self._custom_fields:
            fields_to_request = _FULL_FIELDS + ',' + ','.join(self._custom_fields)
        issue = await asyncio.to_thread(
            self._client.issue, issue_key, fields=fields_to_request,
        )
        fields = issue.get('fields', {})

        summary = fields.get('summary', '')
        updated_at = _parse_jira_date(fields.get('updated', ''))
        created_at = _parse_jira_date(fields.get('created', ''))

        reporter = _get_display_name(fields.get('reporter'))
        assignee = _get_display_name(fields.get('assignee'))

        # Для эпиков дополнительно загружаем дочерние задачи
        epic_issues: list[dict] = []
        if fields.get('issuetype', {}).get('name', '').lower() == 'epic':
            epic_issues = await self._fetch_epic_issues(issue_key)

        custom_field_names = {fid: self._field_names.get(fid, fid) for fid in self._custom_fields}
        text = _build_markdown(
            issue_key, fields,
            epic_issues=epic_issues,
            custom_field_names=custom_field_names,
        )
        path = [f'{dp}/{issue_key}' for dp in doc_paths]

        return Document(
            id=issue_key,
            path=path,
            text=text,
            updated_at=updated_at,
            source_type='attached_jira',
            size=len(text.encode('utf-8')),
            url=f'{self._base_url}/browse/{issue_key}',
            creator=reporter,
            created_at=created_at,
            parent_doc_ids=self._parent_ids_map.get(issue_key, []),
            payload={'summary': summary, 'assignee': assignee},
        )

    async def _fetch_epic_issues(self, epic_key: str) -> list[dict]:
        """Получить список задач эпика через JQL (один запрос, без рекурсии)."""
        _EPIC_ISSUE_FIELDS = 'summary,status,priority,issuetype,assignee'
        try:
            result = await asyncio.to_thread(
                self._client.jql,
                f'"Epic Link" = {epic_key} ORDER BY created ASC',
                fields=_EPIC_ISSUE_FIELDS,
                limit=200,
            )
            issues = result.get('issues', [])
            logger.debug('Epic %s: found %d child issue(s)', epic_key, len(issues))
            return issues
        except Exception:
            logger.warning('Failed to fetch epic issues for %s, skipping', epic_key, exc_info=True)
            return []


def _build_markdown(
    issue_key: str,
    fields: dict,
    epic_issues: list[dict] | None = None,
    custom_field_names: dict[str, str] | None = None,
) -> str:
    """Сформировать markdown-представление задачи Jira."""
    lines: list[str] = []

    summary = fields.get('summary', '')
    lines.append(f'# {issue_key}: {summary}')
    lines.append('')

    # Метаданные задачи
    status = fields.get('status', {}).get('name', '')
    priority = fields.get('priority', {}).get('name', '')
    issue_type = fields.get('issuetype', {}).get('name', '')
    reporter = _get_display_name(fields.get('reporter'))
    assignee = _get_display_name(fields.get('assignee'))
    created = fields.get('created', '')
    updated = fields.get('updated', '')
    labels = fields.get('labels', [])

    if issue_type:
        lines.append(f'**Тип:** {issue_type}')
    if status:
        lines.append(f'**Статус:** {status}')
    if priority:
        lines.append(f'**Приоритет:** {priority}')
    if reporter:
        lines.append(f'**Автор:** {reporter}')
    if assignee:
        lines.append(f'**Исполнитель:** {assignee}')
    if created:
        lines.append(f'**Создано:** {created[:10]}')
    if updated:
        lines.append(f'**Обновлено:** {updated[:10]}')
    if labels:
        lines.append(f'**Метки:** {", ".join(labels)}')

    # Описание
    description = fields.get('description')
    if description:
        desc_text = _extract_text(description)
        if desc_text:
            lines.append('')
            lines.append('## Описание')
            lines.append('')
            lines.append(desc_text)

    # Кастомные поля
    if custom_field_names:
        for field_id, display_name in custom_field_names.items():
            value = fields.get(field_id)
            if not value:
                continue
            text = _extract_custom_field_text(value)
            if text:
                lines.append('')
                lines.append(f'## {display_name}')
                lines.append('')
                lines.append(text)

    # Подзадачи — все поля, доступные в ответе родительской задачи без доп. запросов
    subtasks = fields.get('subtasks', [])
    if subtasks:
        lines.append('')
        lines.append('## Подзадачи')
        lines.append('')
        for subtask in subtasks:
            key = subtask.get('key', '')
            sf = subtask.get('fields', {})
            sub_summary = sf.get('summary', '')
            sub_status = sf.get('status', {}).get('name', '')
            sub_priority = sf.get('priority', {}).get('name', '')
            sub_type = sf.get('issuetype', {}).get('name', '')
            sub_assignee = _get_display_name(sf.get('assignee'))

            meta_parts = [p for p in [sub_type, sub_status, sub_priority, sub_assignee] if p]
            meta = f' ({", ".join(meta_parts)})' if meta_parts else ''
            lines.append(f'- **{key}**{meta}: {sub_summary}')

    # Задачи эпика (только для issue типа Epic, один доп. JQL-запрос)
    if epic_issues:
        lines.append('')
        lines.append('## Задачи эпика')
        lines.append('')
        for ei in epic_issues:
            ei_key = ei.get('key', '')
            ef = ei.get('fields', {})
            ei_summary = ef.get('summary', '')
            ei_status = ef.get('status', {}).get('name', '')
            ei_priority = ef.get('priority', {}).get('name', '')
            ei_type = ef.get('issuetype', {}).get('name', '')
            ei_assignee = _get_display_name(ef.get('assignee'))

            meta_parts = [p for p in [ei_type, ei_status, ei_priority, ei_assignee] if p]
            meta = f' ({", ".join(meta_parts)})' if meta_parts else ''
            lines.append(f'- **{ei_key}**{meta}: {ei_summary}')

    # Связанные задачи (только перечисление, без рекурсии)
    issuelinks = fields.get('issuelinks', [])
    if issuelinks:
        lines.append('')
        lines.append('## Связанные задачи')
        lines.append('')
        for link in issuelinks:
            link_type = link.get('type', {})
            for direction, label in (
                ('inwardIssue', link_type.get('inward', link_type.get('name', ''))),
                ('outwardIssue', link_type.get('outward', link_type.get('name', ''))),
            ):
                linked = link.get(direction)
                if not linked:
                    continue
                lf = linked.get('fields', {})
                l_key = linked.get('key', '')
                l_summary = lf.get('summary', '')
                l_status = lf.get('status', {}).get('name', '')
                l_priority = lf.get('priority', {}).get('name', '')
                l_type = lf.get('issuetype', {}).get('name', '')
                l_assignee = _get_display_name(lf.get('assignee'))

                meta_parts = [p for p in [l_type, l_status, l_priority, l_assignee] if p]
                meta = f' ({", ".join(meta_parts)})' if meta_parts else ''
                lines.append(f'- **{l_key}**{meta} [{label}]: {l_summary}')

    # Комментарии
    comments = fields.get('comment', {}).get('comments', [])
    if comments:
        lines.append('')
        lines.append('## Комментарии')
        for comment in comments:
            author = _get_display_name(comment.get('author'))
            created = comment.get('created', '')[:10]
            body = _extract_text(comment.get('body', ''))
            if body:
                lines.append('')
                lines.append(f'### {author} ({created})')
                lines.append('')
                lines.append(body)

    return '\n'.join(lines)


def _extract_custom_field_text(value) -> str:
    """Извлечь текст из значения кастомного поля Jira.

    Кастомные поля могут быть: строкой, числом, словарём (select/ADF),
    списком (multi-select), None.
    """
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        # Select-поле: {'value': 'Option A'} или {'name': 'Name'}
        if 'value' in value:
            return value['value']
        if 'name' in value:
            return value['name']
        # ADF (Atlassian Document Format)
        if 'type' in value and 'content' in value:
            return _extract_adf_text(value)
        return ''
    if isinstance(value, list):
        parts = [_extract_custom_field_text(item) for item in value]
        return ', '.join(p for p in parts if p)
    return ''


def _get_display_name(user: dict | None) -> str | None:
    """Извлечь отображаемое имя пользователя из объекта Jira."""
    if not user:
        return None
    return user.get('displayName') or user.get('name') or None


def _extract_text(value: str | dict | None) -> str:
    """Извлечь текст из поля Jira: plain text или Atlassian Document Format (ADF)."""
    if not value:
        return ''
    if isinstance(value, str):
        return value
    # ADF (Atlassian Document Format) — рекурсивно извлекаем text-узлы
    if isinstance(value, dict):
        return _extract_adf_text(value)
    return ''


def _extract_adf_text(node: dict) -> str:
    """Рекурсивно извлечь текст из ADF-узла."""
    node_type = node.get('type', '')
    parts: list[str] = []

    # Листовой текстовый узел
    if node_type == 'text':
        return node.get('text', '')

    # Обрабатываем дочерние узлы
    for child in node.get('content', []):
        child_text = _extract_adf_text(child)
        if child_text:
            parts.append(child_text)

    # Добавляем разрывы между блочными элементами
    separator = '\n\n' if node_type in ('paragraph', 'heading', 'bulletList', 'orderedList', 'blockquote') else ' '
    return separator.join(parts)


def _parse_jira_date(date_str: str) -> datetime:
    """Парсить дату из Jira API в datetime с UTC."""
    if not date_str:
        return datetime.now(tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc)
    except ValueError:
        logger.warning('Cannot parse Jira date: %r', date_str)
        return datetime.now(tz=timezone.utc)
