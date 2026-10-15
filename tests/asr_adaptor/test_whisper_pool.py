"""Пул whisper: адреса из `ASR_WHISPER_INSTANCES` совпадают с портами `stack.sh`, кусок — свободному."""
from __future__ import annotations

import threading

import audio_clients
import config


def test_instances_give_the_ports_stack_sh_starts(monkeypatch):
    monkeypatch.delenv('ASR_BACKEND_URLS', raising=False)
    monkeypatch.delenv('ASR_BACKEND_URL', raising=False)
    monkeypatch.setenv('ASR_WHISPER_INSTANCES', '3')
    assert [u.split('/')[2] for u in config._asr_urls()] == ['127.0.0.1:8123', '127.0.0.1:8133', '127.0.0.1:8143']
    monkeypatch.setenv('ASR_WHISPER_INSTANCES', '1')
    assert len(config._asr_urls()) == 1


def test_explicit_list_wins_and_slots_follow_the_pool(monkeypatch):
    monkeypatch.setenv('ASR_BACKEND_URLS', 'http://a/v1/audio/transcriptions, http://b/v1/audio/transcriptions')
    monkeypatch.delenv('ASR_WHISPER_SLOTS', raising=False)
    cfg = config.Config()
    assert cfg.asr_urls == ['http://a/v1/audio/transcriptions', 'http://b/v1/audio/transcriptions']
    assert cfg.whisper_slots == 2, 'слот на процесс пула, если слоты не заданы явно'


def test_request_goes_to_the_least_busy_instance(monkeypatch):
    monkeypatch.setattr(audio_clients.CFG, 'asr_urls', ['u1', 'u2'])
    monkeypatch.setattr(audio_clients, '_INFLIGHT', {})
    with audio_clients._whisper_url() as first:
        with audio_clients._whisper_url() as second:
            assert {first, second} == {'u1', 'u2'}
        with audio_clients._whisper_url() as third:
            assert third == second, 'освободившийся процесс — первым'
    assert audio_clients._INFLIGHT == {'u1': 0, 'u2': 0}


def test_pool_counts_survive_threads(monkeypatch):
    monkeypatch.setattr(audio_clients.CFG, 'asr_urls', ['u1', 'u2', 'u3'])
    monkeypatch.setattr(audio_clients, '_INFLIGHT', {})

    def hit():
        for _ in range(200):
            with audio_clients._whisper_url():
                pass

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(audio_clients._INFLIGHT.values()) == {0}
