#!/usr/bin/env python3
"""Превью Jira-задачи: загрузить задачу и вывести markdown в консоль.

Использование:
    python scripts/jira_preview.py PROJ-123
    python scripts/jira_preview.py https://jira.example.com/browse/PROJ-123
    python scripts/jira_preview.py PROJ-123 --config config.yml
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

# Добавляем src в путь, чтобы можно было запускать из корня проекта
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from morag.config import load_config
from morag.sources.jira import JiraSource


def parse_issue_key(arg: str) -> str:
    """Извлечь ключ задачи из URL или вернуть как есть."""
    # Ссылка вида https://jira.example.com/browse/PROJ-123
    match = re.search(r'/browse/([A-Z][A-Z0-9]*-\d+)', arg)
    if match:
        return match.group(1)
    # Уже ключ вида PROJ-123
    if re.fullmatch(r'[A-Z][A-Z0-9]*-\d+', arg):
        return arg
    raise ValueError(f'Не удалось распознать ключ задачи: {arg!r}')


async def main() -> None:
    parser = argparse.ArgumentParser(description='Загрузить Jira-задачу и вывести markdown')
    parser.add_argument('issue', help='Ключ задачи (PROJ-123) или ссылка (https://jira.../browse/PROJ-123)')
    parser.add_argument('--config', default='config.yml', metavar='PATH', help='Путь к конфигу (по умолчанию: config.yml)')
    parser.add_argument('--raw', action='store_true', help='Показать сырой JSON-ответ от Jira API')
    args = parser.parse_args()

    issue_key = parse_issue_key(args.issue)

    config = load_config(args.config)
    if not config.sources.jira:
        print('ERROR: секция sources.jira не задана в конфиге', file=sys.stderr)
        sys.exit(1)

    # Создаём JiraSource с минимальным issue_map — один ключ, путь-заглушка
    issue_map = {issue_key: ['<preview>']}
    source = JiraSource(config.sources.jira, issue_map)

    print(f'Загружаю {issue_key} из {config.sources.jira.url} ...\n')

    if args.raw:
        import json
        from atlassian import Jira
        credential = config.sources.jira.api_token or config.sources.jira.password
        client = Jira(
            url=config.sources.jira.url,
            username=config.sources.jira.username,
            password=credential,
            cloud=config.sources.jira.api_token is not None,
        )
        raw = client.issue(issue_key, fields='summary,description,status,priority,issuetype,assignee,reporter,created,updated,labels,comment,subtasks,issuelinks')
        print(json.dumps(raw, ensure_ascii=False, indent=2))
        return

    doc = await source.load_one(issue_key)
    if doc is None:
        print(f'ERROR: не удалось загрузить задачу {issue_key}', file=sys.stderr)
        sys.exit(1)

    print('=' * 72)
    print(f'id:         {doc.id}')
    print(f'path:       {doc.path}')
    print(f'url:        {doc.url}')
    print(f'source:     {doc.source_type}')
    print(f'creator:    {doc.creator}')
    print(f'created_at: {doc.created_at}')
    print(f'updated_at: {doc.updated_at}')
    print(f'size:       {doc.size} bytes')
    print('=' * 72)
    print()
    print(doc.text)


if __name__ == '__main__':
    asyncio.run(main())
