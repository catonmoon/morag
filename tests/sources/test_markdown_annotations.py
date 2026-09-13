"""Сайдкар аннотаций рядом с markdown-документом (ADR-0027)."""
import json
import os

from morag.sources.markdown import MarkdownSource

PFX = 'local:default:'


def _corpus(tmp_path, sidecar: str | None = '{"version": "annotations-v1", "items": []}'):
    (tmp_path / 'talk.md').write_text('---\ntitle: Очереди\n---\n\n[Ведущий] <!-- t:0.0 --> Привет.\n',
                                       encoding='utf-8')
    if sidecar is not None:
        (tmp_path / 'talk.annotations.json').write_text(sidecar, encoding='utf-8')
    return tmp_path


async def test_sidecar_is_ignored_without_suffix(tmp_path):
    """Секции конфига нет → сайдкар не читается: поведение прежнее байт в байт."""
    root = _corpus(tmp_path, '{"items": [{"kind": "boundary", "at": 84.0}]}')
    doc = await MarkdownSource(root).load_one(f'{PFX}talk.md')
    assert doc.annotations == []


async def test_sidecar_items_arrive_as_is(tmp_path):
    items = [{'kind': 'boundary', 'at': 84.0},
             {'kind': 'screen', 't0': 84.0, 't1': 333.0, 'text': 'Producer → Kafka'},
             {'kind': 'ref', 'at': 140.5, 'to': 84.0, 'quote': 'вот здесь'}]
    root = _corpus(tmp_path, json.dumps({'version': 'annotations-v1', 'items': items}))
    src = MarkdownSource(root, annotations_suffix='.annotations.json')
    doc = await src.load_one(f'{PFX}talk.md')
    assert doc.annotations == items
    # и НЕ в payload: payload копируется в каждый чанк
    assert 'annotations' not in doc.payload


async def test_items_without_kind_are_dropped(tmp_path):
    root = _corpus(tmp_path, json.dumps({'items': [{'at': 1.0}, 'мусор', {'kind': 'boundary', 'at': 2.0}]}))
    doc = await MarkdownSource(root, annotations_suffix='.annotations.json').load_one(f'{PFX}talk.md')
    assert doc.annotations == [{'kind': 'boundary', 'at': 2.0}]


async def test_broken_sidecar_does_not_drop_document(tmp_path):
    root = _corpus(tmp_path, '{"items": [')
    doc = await MarkdownSource(root, annotations_suffix='.annotations.json').load_one(f'{PFX}talk.md')
    assert doc is not None and doc.annotations == []


async def test_missing_sidecar_is_fine(tmp_path):
    root = _corpus(tmp_path, sidecar=None)
    src = MarkdownSource(root, annotations_suffix='.annotations.json')
    doc = await src.load_one(f'{PFX}talk.md')
    assert doc.annotations == []


async def test_sidecar_mtime_moves_updated_at_in_both_places(tmp_path):
    """Правка сайдкара обязана переиндексировать документ: updated_at = max(md, сайдкар) —
    и у заглушки get_metadata, и у load_one, иначе документ переиндексируется вечно."""
    root = _corpus(tmp_path)
    md, side = root / 'talk.md', root / 'talk.annotations.json'
    os.utime(md, (1_700_000_000, 1_700_000_000))
    os.utime(side, (1_700_000_500, 1_700_000_500))
    src = MarkdownSource(root, annotations_suffix='.annotations.json')
    stub = (await src.get_metadata())[0]
    doc = await src.load_one(stub.id)
    assert stub.updated_at == doc.updated_at
    assert stub.updated_at.timestamp() == 1_700_000_500
    # без суффикса mtime сайдкара не учитывается вовсе
    plain = (await MarkdownSource(root).get_metadata())[0]
    assert plain.updated_at.timestamp() == 1_700_000_000
