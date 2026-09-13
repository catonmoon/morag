"""Поле чанка из аннотаций (ADR-0027, этап D): выбор по отрезку, процессор, лексика."""
from datetime import datetime, timezone

from morag.indexing.annotations import annotation_text, item_text, select_annotations
from morag.indexing.bm25 import bm25_text
from morag.indexing.processors import AnnotationsProcessor, SparseEmbeddingProcessor
from morag.indexing.token_counter import TiktokenCounter
from morag.sources.base import Chunk
from tests.indexing.test_processors import FakeEmbedder, FakeSparseEmbedder, make_document

counter = TiktokenCounter()
ITEMS = [
    {'kind': 'boundary', 'at': 84.0},
    {'kind': 'screen', 't0': 0.0, 't1': 84.0, 'sub': 'slide', 'title': 'Титул', 'text': 'Очереди сообщений'},
    {'kind': 'screen', 't0': 84.0, 't1': 333.0, 'sub': 'slide', 'title': 'Брокер', 'text': 'Producer → Kafka → Consumer'},
    {'kind': 'screen', 't0': 333.0, 't1': 400.0, 'sub': 'app', 'title': 'Окно', 'text': 'psql'},
    {'kind': 'screen', 't0': 200.0, 't1': 200.0, 'sub': 'browser', 'text': 'точка выборки'},
    {'kind': 'ref', 'at': 140.5, 'to': 84.0, 'quote': 'вот здесь', 'text': 'стрелка «ack»'},
    {'kind': 'chapter', 't0': 0.0, 't1': 1000.0, 'text': 'глава — другой род, в поле не идёт'},
]


def sel(start, end, **kw):
    return [it.get('title') or it.get('quote') or it.get('text') for it in
            select_annotations(ITEMS, 'screen', start_sec=start, end_sec=end, **kw)]


def test_switch_inside_chunk_and_ref_inside_are_taken():
    # чанк 80-150: смена на 84 внутри; титульный (0-84) покрывает начало лишь 4 с → не берём
    assert sel(80.0, 150.0) == ['Брокер', 'вот здесь']


def test_screen_still_shown_at_chunk_start_is_taken():
    # продолжение длинного слайда: чанк 250-330 внутри отрезка 84-333
    assert sel(250.0, 330.0) == ['Брокер']


def test_tail_of_previous_screen_is_not_leaked():
    # граница притянута на 3 с раньше смены (330 при смене 333): хвост «Брокера» 3 с < 5 с → нет
    assert sel(330.0, 400.0) == ['Окно']


def test_short_previous_screen_uses_half_length():
    items = [{'kind': 'screen', 't0': 100.0, 't1': 106.0, 'text': 'короткий'},
             {'kind': 'screen', 't0': 106.0, 't1': 200.0, 'text': 'долгий'}]
    got = select_annotations(items, 'screen', start_sec=103.0, end_sec=150.0)
    assert [it['text'] for it in got] == ['короткий', 'долгий']   # 3 с ≥ половины 6 с


def test_point_item_only_inside():
    assert 'точка выборки' in sel(190.0, 210.0)
    assert 'точка выборки' not in sel(210.0, 300.0)


def test_open_ended_chunk_takes_everything_after():
    assert sel(300.0, None) == ['Брокер', 'Окно']


def test_other_kinds_and_missing_anchor_ignored():
    assert 'глава — другой род, в поле не идёт' not in sel(0.0, 1000.0)
    assert select_annotations([{'kind': 'screen', 'text': 'без якоря'}], 'screen', start_sec=0.0, end_sec=10.0) == []


def test_char_anchor_for_documents_without_time():
    items = [{'kind': 'figure', 'o0': 100, 'o1': 200, 'text': 'Рис. 1'}, {'kind': 'ref', 'offset': 150, 'quote': 'см. рис. 1'}]
    got = select_annotations(items, 'figure', char_start=120, char_end=300)
    assert [item_text(it) for it in got] == ['Рис. 1', '«см. рис. 1»']


def test_max_tokens_cuts_tail_items_and_first_oversized():
    long = [{'kind': 'screen', 't0': 0.0, 't1': 10.0, 'text': 'слово ' * 300},
            {'kind': 'screen', 't0': 10.0, 't1': 20.0, 'text': 'второй'}]
    got = select_annotations(long, 'screen', start_sec=0.0, end_sec=20.0, counter=counter, max_tokens=50)
    assert len(got) == 1 and counter.count(got[0]['text']) <= 50
    assert long[0]['text'].startswith('слово ' * 300)                # исходник не тронут


def test_item_text_shapes():
    assert item_text({'kind': 'ref', 'quote': 'вот здесь', 'text': 'стрелка'}) == '«вот здесь» — стрелка'
    assert item_text({'kind': 'screen', 'title': 'Брокер', 'text': 'Kafka'}) == 'Брокер\nKafka'
    assert annotation_text([]) == ''


# --- процессор ------------------------------------------------------------------------------------

def _doc_and_chunk(start=80.0, end=150.0):
    doc = make_document()
    doc.annotations = ITEMS
    chunk = Chunk(doc_id=doc.id, path=['talks/alpha.md'], order=0, total=1, text='[A] речь',
                  updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    chunk.payload.update({'start_sec': start, 'end_sec': end, 'char_offset': 0})
    return doc, chunk


async def test_processor_writes_field_and_named_vector():
    doc, chunk = _doc_and_chunk()
    proc = AnnotationsProcessor('screen', counter, max_tokens=400, embedder=FakeEmbedder())
    await proc.process(chunk, doc)
    assert [it['kind'] for it in chunk.payload['screen']] == ['screen', 'ref']
    assert 'screen' in chunk.vectors and 'full' not in chunk.vectors


async def test_processor_leaves_chunk_without_items_untouched():
    doc, chunk = _doc_and_chunk(start=900.0, end=950.0)
    await AnnotationsProcessor('screen', counter, embedder=FakeEmbedder()).process(chunk, doc)
    assert 'screen' not in chunk.payload and chunk.vectors == {}


async def test_processor_without_embedder_writes_only_field():
    doc, chunk = _doc_and_chunk()
    await AnnotationsProcessor('screen', counter).process(chunk, doc)
    assert chunk.payload['screen'] and chunk.vectors == {}


async def test_processor_batch_embeds_only_chunks_with_items():
    doc, a = _doc_and_chunk()
    _, b = _doc_and_chunk(start=900.0, end=950.0)
    await AnnotationsProcessor('screen', counter, embedder=FakeEmbedder()).process_batch([a, b], doc)
    assert 'screen' in a.vectors and 'screen' not in b.vectors and 'screen' not in b.payload


# --- лексика: sparse и BM25 собирают один и тот же текст ---------------------------------------------

def test_sparse_text_mixes_annotations_when_asked():
    doc, chunk = _doc_and_chunk()
    chunk.payload['screen'] = [ITEMS[2], ITEMS[5]]
    with_ann = SparseEmbeddingProcessor(FakeSparseEmbedder(), annotation_field='screen')._sparse_text(chunk, doc)
    without = SparseEmbeddingProcessor(FakeSparseEmbedder())._sparse_text(chunk, doc)
    assert without == '[A] речь'
    assert with_ann == '[A] речь\nБрокер\nProducer → Kafka → Consumer\n«вот здесь» — стрелка «ack»'
    assert bm25_text(chunk.payload | {'text': chunk.text}, annotation_field='screen') == with_ann
    assert bm25_text({'text': '[A] речь'}, annotation_field='screen') == '[A] речь'
