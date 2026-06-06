from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source

_FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n?', re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Извлечь YAML-front-matter (простые `key: value`) и тело без него.

    Поддерживается минимальный subset: плоские строки `key: value` в блоке `---...---`
    в начале файла. Достаточно для `title`/`url` аудио-транскриптов. Без front-matter —
    возвращает ({}, text).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip()
    return meta, text[m.end():]


class MarkdownSource(Source):
    """Источник локальных Markdown-файлов.

    Рекурсивно сканирует директорию и возвращает Document для каждого *.md файла.
    parent_doc_ids ссылается на структурные документы директорий (DirectorySource).

    kind='local' (соответствует discriminator config.LocalSourceConfig). name —
    из config (передаётся через LocalDocumentSource). Document.id форматируется
    через self.make_id() — `local:<name>:<relative-path>`.
    """

    @property
    def source_type(self) -> str:
        return 'markdown'

    def __init__(self, root: Path | str, name: str = 'default') -> None:
        self._root = Path(root).resolve()
        self._kind = 'local'
        self._name = name

    async def get_metadata(self) -> list[Document]:
        all_md_files = sorted(self._root.rglob('*.md'))

        stubs: list[Document] = []
        for path in all_md_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        external = self._strip_prefix(doc_id)
        return self._load_file(self._root / external)

    def _strip_prefix(self, doc_id: str) -> str:
        """Извлечь external-id (relative path) из prefixed doc_id."""
        prefix = f'{self._kind}:{self._name}:'
        return doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Parent — структурный документ директории. ID тоже prefixed."""
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [self.make_id(str(parent_dir.relative_to(self._root)) + '/')]

    def _get_file_metadata(self, path: Path) -> Document | None:
        try:
            stat = path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            external = str(path.relative_to(self._root))
            doc_id = self.make_id(external)
            return Document(
                id=doc_id,
                path=[external],  # path остаётся "human-readable", без prefix
                text='',
                updated_at=updated_at,
                source_type='markdown',
                title=path.stem,
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except OSError:
            return None

    def _load_file(self, path: Path) -> Document | None:
        try:
            stat = path.stat()
            raw = path.read_text(encoding='utf-8')
            meta, text = _parse_frontmatter(raw)
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            external = str(path.relative_to(self._root))
            doc_id = self.make_id(external)
            return Document(
                id=doc_id,
                path=[external],
                text=text,
                updated_at=updated_at,
                source_type='markdown',
                title=meta.get('title', path.stem),
                size=stat.st_size,
                # front-matter `url` (напр. оригинальный mp3) переопределяет file:// —
                # нужно для deep-link цитат на источник.
                url=meta.get('url', path.as_uri()),
                parent_doc_ids=self._parent_doc_ids(path),
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except OSError:
            return None
