"""Реестр голосов: приклейка короткого кластера к чужому голосу должна быть ЗАМЕТНОЙ.

Порог косинуса (`ASR_MATCH_THRESHOLD`) на это не влияет вовсе — решает `ASR_MIN_GUEST_MIN`:
не совпавший с реестром голос заводится отдельным человеком, только если наговорил достаточно,
иначе приклеивается к ближайшему НЕЗАВИСИМО от косинуса. Для подкаста это защита от обрывков,
для записи с короткими репликами из зала — гарантированная порча: замерено на 13-минутном
митапе, где шесть диаризованных кластеров свелись в один Speaker_N.

Ошибка молчаливая — расшифровка выглядит удавшейся, просто один «человек» задаёт себе вопрос
и сам отвечает. Поэтому проверяем именно предупреждения.
"""
import importlib
import logging

import numpy as np
import pytest


def _reload(monkeypatch, **env):
    """registry читает ASR_MIN_GUEST_MIN при импорте — перечитываем модуль под новым окружением."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import stages.registry as registry
    return importlib.reload(registry)


def _voice(seed: int) -> list:
    """L2-нормированный вектор: разные seed → разные люди (косинус около нуля)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_short_stranger_folds_and_warns(tmp_path, monkeypatch, caplog):
    """Митап: докладчик говорит 700 с, вопрос из зала — 15 с, голоса РАЗНЫЕ."""
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN='2.0')
    reg = tmp_path / 'registry.json'

    cents = {'SPEAKER_00': _voice(1), 'SPEAKER_01': _voice(2)}
    air = {'SPEAKER_00': 701.0, 'SPEAKER_01': 15.0}

    with caplog.at_level(logging.WARNING, logger='asr'):
        mapping = registry.assign(cents, air, 'ep1', str(reg))

    assert len(set(mapping.values())) == 1, 'ожидаем воспроизведение поломки: оба свелись в один'
    text = ' '.join(r.getMessage() for r in caplog.records)
    assert 'ASR_MIN_GUEST_MIN' in text, 'предупреждение обязано назвать виновную настройку'
    assert 'ОДИН' in text, 'схлопывание записи в один голос должно быть названо прямо'
    assert 'speaker_map' in text, 'предупреждение должно сказать, где посмотреть глазами'


def test_lower_threshold_keeps_them_apart(tmp_path, monkeypatch, caplog):
    """С ASR_MIN_GUEST_MIN=0.25 (15 с) тот же вход даёт двух людей и не жалуется."""
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN='0.25')
    reg = tmp_path / 'registry.json'

    cents = {'SPEAKER_00': _voice(1), 'SPEAKER_01': _voice(2)}
    air = {'SPEAKER_00': 701.0, 'SPEAKER_01': 15.0}

    with caplog.at_level(logging.WARNING, logger='asr'):
        mapping = registry.assign(cents, air, 'ep1', str(reg))

    assert len(set(mapping.values())) == 2, 'короткий голос должен стать отдельным человеком'
    assert not [r for r in caplog.records if 'ОДИН' in r.getMessage()]


def test_same_voice_is_matched_not_folded(tmp_path, monkeypatch, caplog):
    """Тот же голос во второй записи узнаётся по косинусу — это не приклейка, жалоб быть не должно."""
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN='2.0')
    reg = tmp_path / 'registry.json'
    voice = _voice(1)

    registry.assign({'A': voice}, {'A': 700.0}, 'ep1', str(reg))
    with caplog.at_level(logging.WARNING, logger='asr'):
        mapping = registry.assign({'B': voice}, {'B': 30.0}, 'ep2', str(reg))

    assert mapping['B'] == 'Speaker_0'
    assert not caplog.records, 'узнавание по голосу не должно выглядеть как проблема'


@pytest.mark.parametrize('minutes,expect_people', [('2.0', 1), ('0.25', 2)])
def test_setting_actually_decides(tmp_path, monkeypatch, minutes, expect_people):
    """Одна и та же запись, разный ASR_MIN_GUEST_MIN — разное число людей."""
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN=minutes)
    mapping = registry.assign(
        {'A': _voice(1), 'B': _voice(2)}, {'A': 600.0, 'B': 20.0},
        'ep1', str(tmp_path / 'r.json'))
    assert len(set(mapping.values())) == expect_people


# --- решение по каждому кластеру: то, что рисуется на карте голосов ----------------------------

def test_the_decision_for_every_cluster_can_be_collected(tmp_path, monkeypatch):
    """Косинус и ветка решения считаются и так — просто не выходили наружу.

    ⚠️⚠️ Центроида в отчёте нет и быть не может: это биометрический признак живого человека, а
    лента событий сохраняется в файл. Наружу — только решение и близость.
    """
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN='2.0')
    reg = tmp_path / 'registry.json'

    # первый прогон заводит голос, второй его узнаёт и приклеивает короткого незнакомца
    registry.assign({'SPEAKER_00': _voice(1)}, {'SPEAKER_00': 700.0}, 'ep1', str(reg))
    out: list[dict] = []
    mapping = registry.assign({'A': _voice(1), 'B': _voice(2)}, {'A': 700.0, 'B': 15.0},
                              'ep2', str(reg), out=out)

    assert len(out) == 2 and [d['cluster'] for d in out] == ['A', 'B'], 'по убыванию эфира'
    known, stranger = out
    assert known['action'] == 'matched' and known['cos'] > 0.99 and known['label'] == mapping['A']
    assert stranger['action'] == 'folded' and stranger['cos'] < 0.5
    assert all('centroid' not in d and 'cent' not in d for d in out), 'биометрии в отчёте нет'
    assert all(isinstance(v, (str, int, float)) for d in out for v in d.values())


def test_collecting_decisions_does_not_change_the_mapping(tmp_path, monkeypatch):
    registry = _reload(monkeypatch, ASR_MIN_GUEST_MIN='0.25')
    cents, air = {'A': _voice(3), 'B': _voice(4)}, {'A': 700.0, 'B': 300.0}

    plain = registry.assign(cents, air, 'ep1', str(tmp_path / 'a.json'))
    noisy = registry.assign(cents, air, 'ep1', str(tmp_path / 'b.json'), out=[])
    assert plain == noisy
