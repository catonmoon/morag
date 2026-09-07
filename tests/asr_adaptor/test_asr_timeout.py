"""Таймаут пасса-1 растёт с длиной записи.

Куплено ночным прогоном курса QA: шесть записей от 172 до 328 минут сорвались на `Read timed out`
ПОСЛЕ того, как конвейер честно отработал диаризацию, — фиксированный потолок в 300 с не оставлял
пассу-1 шанса. ⚠️ Отказ выглядел как сетевой, хотя причина была в длине файла.
"""
import wave

import pytest

from audio_clients import _ASR_TIMEOUT_MIN, _asr_timeout


def _wav(path, seconds: float):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b'\0\0' * int(16000 * seconds))
    return str(path)


def test_short_chunk_keeps_the_floor(tmp_path):
    """Куски пасса-2 идут по 28 с: там время уходит на очередь, а не на счёт."""
    assert _asr_timeout(_wav(tmp_path / 'c.wav', 28)) == _ASR_TIMEOUT_MIN


def test_long_record_gets_more_than_the_floor(tmp_path):
    """Запись на 328 минут — та самая, что сорвалась ночью.

    ⓘ Пол держится до сорока минут звука: короче него делитель даёт меньше 300 с.
    """
    assert _asr_timeout(_wav(tmp_path / 'l.wav', 328 * 60)) > _ASR_TIMEOUT_MIN


@pytest.mark.parametrize('minutes', [172, 199, 236, 328])
def test_timeouts_cover_measured_pass1_speed(tmp_path, minutes):
    """Пасс-1 замерен на 31-34× реального времени; потолок обязан покрывать его с запасом."""
    got = _asr_timeout(_wav(tmp_path / f'{minutes}.wav', minutes * 60))
    need = minutes * 60 / 31.2          # худшая измеренная скорость
    assert got > need * 2, f'{minutes} мин: потолок {got} с при нужных {need:.0f} с'


def test_unreadable_file_falls_back_to_the_floor(tmp_path):
    """Не прочли заголовок — ведём себя как раньше, а не роняем расшифровку."""
    bad = tmp_path / 'broken.wav'
    bad.write_bytes(b'not a wav')
    assert _asr_timeout(str(bad)) == _ASR_TIMEOUT_MIN
