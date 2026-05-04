#!/usr/bin/env python3
"""Migrate config.yml from old (pre-ADR-0012) schema to new schema_version=1.

Преобразования:
- sources.local_documents.{path}     → sources: [{kind: local, name: 'docs', path: ...}]
- sources.confluence.{...}           → sources: [{kind: confluence, name: 'main', ...}]
- sources.jira.{...}                 → sources: [{kind: jira, name: 'main', ..., только password}]
- llm.{...}                          → llms: [{name: 'main', ...}]
- llm_vision.{...}                   → llms: [{name: 'vision', capabilities: [text, vision], ...}]
- indexing.llm = 'main', indexing.vision = 'vision' (auto-wired)
- schema_version: 1 (новое)

Usage:
    python scripts/migrate_config.py < config.yml > config.yml.new
    # затем посмотреть и заменить
    diff config.yml config.yml.new
    mv config.yml.new config.yml

Output записывает на stdout. Stderr — diagnostics. Старый файл НЕ перезаписывает.
"""
from __future__ import annotations

import sys
from typing import Any

import yaml


def migrate(old: dict[str, Any]) -> dict[str, Any]:
    """Старый dict → новый dict. Pure function, легко тестируется."""
    new: dict[str, Any] = {'schema_version': 1}

    # ---- sources: dict с named-секциями → list[Source] ----
    new['sources'] = _migrate_sources(old.get('sources') or {})

    # ---- llms: top-level llm + llm_vision → list[LLMInstance] ----
    new['llms'] = _migrate_llms(old.get('llm'), old.get('llm_vision'))

    # ---- остальные секции переносятся как есть, но llm/vision добавляются в indexing ----
    for key in ('qdrant', 'pdf'):
        if key in old:
            new[key] = old[key]

    if 'indexing' in old:
        new['indexing'] = _migrate_indexing(old['indexing'])

    return new


def _migrate_sources(old_sources: dict[str, Any]) -> list[dict[str, Any]]:
    """SourcesConfig (3 named-секции) → list of Source items."""
    sources: list[dict[str, Any]] = []

    if 'local_documents' in old_sources:
        ld = old_sources['local_documents']
        if isinstance(ld, dict) and 'path' in ld:
            sources.append({'kind': 'local', 'name': 'docs', 'path': ld['path']})
            print('  → local source: name=docs, path=' + ld['path'], file=sys.stderr)

    if 'confluence' in old_sources:
        cf = dict(old_sources['confluence'])
        cf.update({'kind': 'confluence', 'name': 'main'})
        # Reorder: kind/name первые
        ordered = {'kind': cf.pop('kind'), 'name': cf.pop('name'), **cf}
        sources.append(ordered)
        print('  → confluence source: name=main, url=' + cf.get('url', '?'), file=sys.stderr)

    if 'jira' in old_sources:
        jr = dict(old_sources['jira'])
        # Только on-prem — api_token больше не поддерживается
        if 'api_token' in jr and 'password' not in jr:
            print('  ⚠ jira: api_token detected, but new schema только on-prem (password). '
                  'Manually convert auth to password.', file=sys.stderr)
        jr.pop('api_token', None)
        jr.update({'kind': 'jira', 'name': 'main'})
        ordered = {'kind': jr.pop('kind'), 'name': jr.pop('name'), **jr}
        sources.append(ordered)
        print('  → jira source: name=main, url=' + jr.get('url', '?'), file=sys.stderr)

    if not sources:
        print('  ⚠ No sources found in old config — добавь хотя бы один source вручную',
              file=sys.stderr)

    return sources


def _migrate_llms(old_llm: dict | None, old_vision: dict | None) -> list[dict[str, Any]]:
    """Top-level llm + llm_vision → list[LLMInstance]."""
    llms: list[dict[str, Any]] = []

    if old_llm:
        entry = {'name': 'main', **old_llm}
        # capabilities default to text — explicit для clarity
        entry['capabilities'] = ['text']
        llms.append(entry)
        print('  → llm: name=main, model=' + old_llm.get('model', '?'), file=sys.stderr)

    if old_vision:
        entry = {'name': 'vision', **old_vision}
        entry['capabilities'] = ['text', 'vision']
        llms.append(entry)
        print('  → llm: name=vision, model=' + old_vision.get('model', '?')
              + ', capabilities=[text,vision]', file=sys.stderr)

    if not llms:
        print('  ⚠ No llms found — нужен хотя бы один llm в новой schema', file=sys.stderr)

    return llms


def _migrate_indexing(old_indexing: dict[str, Any]) -> dict[str, Any]:
    """Старая indexing-секция + автоматически добавленные llm/vision references."""
    new = dict(old_indexing)

    # Автомаппинг ролей: indexing.llm → 'main', indexing.vision → 'vision'
    # (имена соответствуют тому что мы назначили в _migrate_llms).
    new['llm'] = 'main'
    new['vision'] = 'vision'
    print('  → indexing.llm = "main", indexing.vision = "vision" (auto-wired)',
          file=sys.stderr)

    return new


def main() -> None:
    old = yaml.safe_load(sys.stdin) or {}
    if not isinstance(old, dict):
        print('Error: input is not a dict', file=sys.stderr)
        sys.exit(1)

    if old.get('schema_version') == 1:
        print('  ⚠ Input уже schema_version=1 — миграция не требуется', file=sys.stderr)
        sys.exit(0)

    print('Migrating config.yml → schema_version=1...', file=sys.stderr)
    new = migrate(old)
    print('Done. Saving to stdout.', file=sys.stderr)

    yaml.safe_dump(new, sys.stdout, allow_unicode=True, sort_keys=False)


if __name__ == '__main__':
    main()
