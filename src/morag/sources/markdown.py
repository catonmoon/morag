from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n?', re.DOTALL)


def _coerce_value(v: str):
    """Front-matter значение → типизированное: JSON-массив/объект (`["a","b"]`) → list/dict;
    целое → int; иначе строка (обрамляющие кавычки снимаются)."""
    if v[:1] in ('[', '{'):
        try:
            return json.loads(v)
        except ValueError:
            pass
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':  # снять обрамляющие кавычки
        return v[1:-1]
    if v.lstrip('-').isdigit():
        return int(v)
    return v


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Извлечь YAML-front-matter (простые `key: value`) и тело без него.

    Минимальный subset: плоские `key: value` в блоке `---...---`. Значения коэрсятся
    (`_coerce_value`): JSON-списки/объекты → list/dict, целые → int, иначе строка. ЛЮБОЙ ключ едет
    дальше generic — ядро не знает про доменные поля (`season`/`speakers`/…). Без front-matter — ({}, text).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            meta[key.strip()] = _coerce_value(val.strip())
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

    def __init__(self, root: Path | str, name: str = 'default',
                 annotations_suffix: str | None = None) -> None:
        self._root = Path(root).resolve()
        self._kind = 'local'
        self._name = name
        # ADR-0027: сайдкар `<stem><suffix>` рядом с файлом. None — не ищется вовсе.
        self._annotations_suffix = annotations_suffix

    # --- аннотации (ADR-0027) -------------------------------------------------------------------

    def _sidecar(self, path: Path) -> Path | None:
        if not self._annotations_suffix:
            return None
        side = path.with_name(path.stem + self._annotations_suffix)
        return side if side.is_file() else None

    def _updated_at(self, path: Path, mtime: float) -> datetime:
        """mtime документа, но не раньше mtime сайдкара.

        ⚠️ «Изменился ли документ» pipeline решает по updated_at; без этого правка сайдкара не
        переиндексирует документ. Считается ОДИНАКОВО для заглушки get_metadata и для load_one —
        разойдись они, документ переиндексировался бы на каждом прогоне.
        """
        side = self._sidecar(path)
        if side is not None:
            try:
                mtime = max(mtime, side.stat().st_mtime)
            except OSError:
                pass
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    def _load_annotations(self, path: Path) -> list[dict]:
        """Элементы сайдкара как есть (`{kind, …}`); битый файл — предупреждение, документ без них."""
        side = self._sidecar(path)
        if side is None:
            return []
        try:
            data = json.loads(side.read_text(encoding='utf-8'))
            items = data.get('items') if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError('нет списка items')
        except (OSError, ValueError) as e:
            logger.warning('Annotations sidecar %s ignored: %s', side, e)
            return []
        return [it for it in items if isinstance(it, dict) and it.get('kind')]

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
            updated_at = self._updated_at(path, stat.st_mtime)
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
            updated_at = self._updated_at(path, stat.st_mtime)
            external = str(path.relative_to(self._root))
            doc_id = self.make_id(external)
            return Document(
                id=doc_id,
                path=[external],
                text=text,
                updated_at=updated_at,
                annotations=self._load_annotations(path),
                source_type='markdown',
                title=meta.get('title', path.stem),
                size=stat.st_size,
                # front-matter `url` (напр. оригинальный mp3) переопределяет file:// —
                # нужно для deep-link цитат на источник.
                url=meta.get('url', path.as_uri()),
                parent_doc_ids=self._parent_doc_ids(path),
                # ВЕСЬ front-matter (кроме title/url → поля Document) едет в payload GENERIC:
                # date/duration_sec/season/episode/speakers + любые будущие доменные поля. Типы уже
                # коэрснуты в _parse_frontmatter. Ядро НЕ знает про конкретные поля (см. ADR/CLAUDE.md).
                payload={'source_name': self._name, 'source_kind': self._kind,
                         **{k: v for k, v in meta.items()
                            if k not in ('title', 'url', 'source_name', 'source_kind')
                            and v not in (None, '')}},
            )
        except OSError:
            return None
