from __future__ import annotations

import logging
import re

from morag.sources.base import Document

logger = logging.getLogger(__name__)


class JiraLinkExtractor:
    """Извлекает ключи Jira-задач из текста документов по URL-префиксу.

    Ищет все href вида `{jira_base_url}/browse/PROJ-123` в markdown-тексте
    и строит маппинг {issue_key: [path_1, path_2, ...]} — список путей документов,
    в которых эта задача упоминается.
    """

    def __init__(self, jira_base_url: str) -> None:
        base = jira_base_url.rstrip('/')
        browse_prefix = re.escape(f'{base}/browse/')
        # Ищем URL вида {jira_url}/browse/PROJ-123 в любом месте текста
        # [A-Z][A-Z0-9]* — ключ проекта: 1+ заглавных букв/цифр; [A-Z0-9]* допускает однобуквенный ключ
        self._pattern = re.compile(browse_prefix + r'([A-Z][A-Z0-9]*-\d+)')

    def extract_from_docs(self, docs: list[Document]) -> dict[str, list[str]]:
        """Просканировать документы и вернуть маппинг {issue_key: [doc_path, ...]}.

        Для каждого документа ищет упоминания Jira-задач по URL.
        Один ключ может встречаться в нескольких документах — все пути сохраняются.
        """
        issue_map: dict[str, list[str]] = {}

        for doc in docs:
            keys = self._extract_keys(doc.text)
            if not keys:
                continue

            # Используем первый путь документа как базовый для построения пути задачи
            doc_path = doc.path[0] if doc.path else doc.id
            logger.debug('Document %s: found %d Jira key(s): %s', doc.id, len(keys), sorted(keys))

            for key in keys:
                if key not in issue_map:
                    issue_map[key] = []
                if doc_path not in issue_map[key]:
                    issue_map[key].append(doc_path)

        logger.info('JiraLinkExtractor: found %d unique issue(s) across %d document(s)', len(issue_map), len(docs))
        return issue_map

    def _extract_keys(self, text: str) -> set[str]:
        """Извлечь все уникальные ключи Jira из текста документа."""
        return set(self._pattern.findall(text))
