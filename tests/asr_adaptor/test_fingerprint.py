"""Отпечаток установки: что уезжает в артефакт вместе с расшифровкой.

Он нужен, когда долгие прогоны уходят на вторую машину, а разработка остаётся на первой: разъезд
установок замечают через месяц по испорченному корпусу, и тогда уже не отличить «конвейер стал
хуже» от «прогон был на другой машине». Отпечаток внутри артефакта отвечает на это без чьей-либо
памяти — но ровно поэтому он обязан быть безопасным: артефакт уезжает вместе с корпусом.
"""
import pytest

from fingerprint import one_line, stack_fingerprint


@pytest.fixture(autouse=True)
def _fresh():
    """Отпечаток кэширован на процесс — тесту нужен пересчёт."""
    stack_fingerprint.cache_clear()
    yield
    stack_fingerprint.cache_clear()


def test_names_the_machine_and_the_interpreter():
    fp = stack_fingerprint()
    assert fp.get('host') and fp.get('python') and fp.get('platform')


def test_carries_versions_that_change_behaviour():
    """`wordfreq` — таблица частот гейта редкости: другая версия даёт другой глоссарий."""
    pkgs = stack_fingerprint().get('packages') or {}
    assert 'wordfreq' in pkgs and 'transformers' in pkgs


def test_endpoint_address_never_leaks():
    """⚠️ Несущее свойство: артефакт уезжает вместе с корпусом. Адрес — только хэшем."""
    from config import CFG
    fp = stack_fingerprint()
    flat = repr(fp)
    if CFG.llm_base_url:
        assert CFG.llm_base_url not in flat
        assert len(fp['llm_endpoint']) == 8, 'хэш короткий: сравнить можно, прочитать нельзя'
    for secret in (CFG.llm_key, CFG.asr_key, CFG.campp_key, CFG.diarizer_key):
        if secret:
            assert secret not in flat


def test_marks_uncommitted_engine():
    """Прогон закоммиченного кода и прогон рабочего дерева обязаны различаться в отпечатке."""
    morag = stack_fingerprint().get('morag')
    if morag is None:
        pytest.skip('не git-чекаут')
    assert morag.split('+')[0], 'коммит должен быть назван'


def test_one_line_survives_a_failed_fingerprint():
    """Отпечаток не имеет права ронять ни расшифровку, ни приёмку партии."""
    assert one_line({}) == '—'
    assert one_line({'error': 'RuntimeError'}) == '—'


def test_one_line_is_comparable_between_machines():
    line = one_line(stack_fingerprint())
    assert line != '—' and 'wordfreq' in line
