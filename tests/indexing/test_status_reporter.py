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
        assert data['current_doc_id'] == 'doc2'

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
        assert data['current_doc_id'] is None

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
