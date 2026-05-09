import json
from pathlib import Path

from morag.indexing.status_reporter import FileStatusReporter, NullStatusReporter


class TestNullStatusReporter:
    def test_all_methods_are_noop(self):
        r = NullStatusReporter()
        r.start_phase('phase', 10)
        r.document_done('doc1')
        r.finish('completed')
        r.finish('failed', error='boom')


class TestFileStatusReporter:
    def test_creates_file_on_init(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        FileStatusReporter(path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data['state'] == 'idle'
        assert data['processed'] == 0
        assert data['total'] == 0

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / 'a' / 'b' / 'c' / 'state.json'
        FileStatusReporter(path)
        assert path.exists()

    def test_start_phase_updates_state(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)

        r.start_phase('indexing_local', 42)
        data = json.loads(path.read_text())

        assert data['state'] == 'running'
        assert data['phase'] == 'indexing_local'
        assert data['processed'] == 0
        assert data['total'] == 42

    def test_document_done_increments_processed(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p', 3)

        r.document_done('doc1')
        r.document_done('doc2')
        data = json.loads(path.read_text())

        assert data['processed'] == 2
        # Skip-path (only document_done без document_start): in-flight остаётся пустым.
        assert data['current_docs'] == []

    def test_start_phase_resets_processed_counter(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p1', 5)
        r.document_done('a')
        r.document_done('b')

        r.start_phase('p2', 10)
        data = json.loads(path.read_text())

        assert data['phase'] == 'p2'
        assert data['processed'] == 0
        assert data['total'] == 10
        assert data['current_docs'] == []

    def test_in_flight_lifecycle(self, tmp_path: Path):
        """document_start → set_chunks → chunk_done × N → document_done."""
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p', 3)

        r.document_start('doc1', title='Foo', url='http://example/foo')
        data = json.loads(path.read_text())
        assert len(data['current_docs']) == 1
        entry = data['current_docs'][0]
        assert entry['doc_id'] == 'doc1'
        assert entry['title'] == 'Foo'
        assert entry['url'] == 'http://example/foo'
        assert entry['chunks_done'] == 0
        assert entry['chunks_total'] is None

        r.document_set_chunks('doc1', 5)
        r.document_chunk_done('doc1')
        r.document_chunk_done('doc1')
        data = json.loads(path.read_text())
        assert data['current_docs'][0]['chunks_total'] == 5
        assert data['current_docs'][0]['chunks_done'] == 2

        r.document_done('doc1')
        data = json.loads(path.read_text())
        assert data['current_docs'] == []
        assert data['processed'] == 1

    def test_parallel_in_flight(self, tmp_path: Path):
        """Несколько документов одновременно in-flight (concurrency)."""
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p', 10)

        r.document_start('a', title='A')
        r.document_start('b', title='B')
        r.document_start('c', title='C')
        data = json.loads(path.read_text())
        assert {d['doc_id'] for d in data['current_docs']} == {'a', 'b', 'c'}

        r.document_done('b')
        data = json.loads(path.read_text())
        assert {d['doc_id'] for d in data['current_docs']} == {'a', 'c'}
        assert data['processed'] == 1

    def test_finish_sets_state_and_error(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)

        r.finish('failed', error='Qdrant down')
        data = json.loads(path.read_text())

        assert data['state'] == 'failed'
        assert data['error'] == 'Qdrant down'

    def test_atomic_write_no_partial_state_visible(self, tmp_path: Path):
        """После каждой операции файл должен быть валидным JSON.

        Проверка через многократные перезаписи: если бы запись была не-атомарной,
        существовал бы кадр когда tmp есть, а target ещё пустой.
        """
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p', 100)
        for i in range(50):
            r.document_done(f'doc{i}')
            # На каждой итерации файл целиком парсится как JSON.
            data = json.loads(path.read_text())
            assert data['processed'] == i + 1

    def test_no_tmp_file_left_after_writes(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        r.start_phase('p', 1)
        r.document_done('x')
        r.finish('completed')

        siblings = list(tmp_path.iterdir())
        assert siblings == [path], f'expected only {path.name}, got {[p.name for p in siblings]}'

    def test_started_at_stable_across_writes(self, tmp_path: Path):
        path = tmp_path / 'state.json'
        r = FileStatusReporter(path)
        first = json.loads(path.read_text())['started_at']

        r.start_phase('p', 1)
        r.document_done('a')
        r.finish('completed')

        last = json.loads(path.read_text())['started_at']
        assert first == last, 'started_at must be stable across the lifetime of a reporter'
