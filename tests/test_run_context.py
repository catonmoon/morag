"""Тесты RunContext — глобальный счётчик прогонов."""
import json
from pathlib import Path

from morag.run_context import RunContext


class TestBegin:

    def test_first_run_starts_at_1(self, tmp_path: Path):
        ctx = RunContext.begin(state_path=tmp_path / 'counter.json')
        assert ctx.run_number == 1

    def test_subsequent_runs_increment(self, tmp_path: Path):
        path = tmp_path / 'counter.json'
        ctx1 = RunContext.begin(state_path=path)
        ctx2 = RunContext.begin(state_path=path)
        ctx3 = RunContext.begin(state_path=path)
        assert (ctx1.run_number, ctx2.run_number, ctx3.run_number) == (1, 2, 3)

    def test_indexed_at_is_iso_utc(self, tmp_path: Path):
        ctx = RunContext.begin(state_path=tmp_path / 'counter.json')
        # Корректно парсится как ISO-datetime
        from datetime import datetime
        dt = datetime.fromisoformat(ctx.indexed_at)
        assert dt.tzinfo is not None  # timezone-aware

    def test_state_file_persists_across_processes(self, tmp_path: Path):
        path = tmp_path / 'counter.json'
        RunContext.begin(state_path=path)
        # Симулируем «новый процесс» — повторно читаем файл
        data = json.loads(path.read_text())
        assert data['current_run'] == 1
        assert 'last_bumped_at' in data

    def test_state_file_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / 'a' / 'b' / 'c' / 'counter.json'
        ctx = RunContext.begin(state_path=nested)
        assert ctx.run_number == 1
        assert nested.exists()


class TestRecovery:

    def test_recovers_from_qdrant_when_file_missing(self, tmp_path: Path):
        path = tmp_path / 'counter.json'
        # Симулируем потерю файла, но в Qdrant max=42
        ctx = RunContext.begin(
            state_path=path,
            recover_from_qdrant=lambda: 42,
        )
        # Recovered = 42, bumped = 43
        assert ctx.run_number == 43

    def test_no_recovery_callback_starts_from_0(self, tmp_path: Path):
        ctx = RunContext.begin(state_path=tmp_path / 'counter.json')
        assert ctx.run_number == 1

    def test_recovery_callback_failure_falls_back_to_0(self, tmp_path: Path):
        def boom():
            raise RuntimeError('Qdrant down')

        ctx = RunContext.begin(
            state_path=tmp_path / 'counter.json',
            recover_from_qdrant=boom,
        )
        assert ctx.run_number == 1  # 0 + 1

    def test_recovery_only_when_file_missing(self, tmp_path: Path):
        """Если state-file есть — recovery не вызывается, читаем counter из файла."""
        path = tmp_path / 'counter.json'
        # Сначала создаём счётчик
        RunContext.begin(state_path=path)  # → 1
        RunContext.begin(state_path=path)  # → 2

        # Теперь recovery callback не должен вызываться
        called = [False]

        def shouldnt_be_called():
            called[0] = True
            return 999

        ctx = RunContext.begin(state_path=path, recover_from_qdrant=shouldnt_be_called)
        assert ctx.run_number == 3
        assert called[0] is False


class TestCorruption:

    def test_corrupted_json_treated_as_missing(self, tmp_path: Path):
        path = tmp_path / 'counter.json'
        path.write_text('not valid json {{{')
        ctx = RunContext.begin(
            state_path=path,
            recover_from_qdrant=lambda: 100,
        )
        # Файл повреждён → recovery → 100 + 1
        assert ctx.run_number == 101

    def test_invalid_current_run_type_treated_as_missing(self, tmp_path: Path):
        path = tmp_path / 'counter.json'
        path.write_text('{"current_run": "not a number"}')
        ctx = RunContext.begin(state_path=path)  # без recovery → 0 + 1
        assert ctx.run_number == 1
