"""Конфиг asr-adaptor: env-driven настройки + сборка morag `LLMClient`.

Аудио (диаризация/ASR/CAM++) — на Маке по HTTP (Apple-Silicon-bound). LLM (облако) — Grok НАПРЯМУЮ
через xAI (`api.x.ai/v1`). Дефолт `grok-4.20-0309-non-reasoning` — быстрый non-reasoning SKU
(~1.5с/вызов vs ~8.5с у grok-4.3, качество коррекции идентично — замерено 2026-06). Для него
reasoning-флаг НЕ шлём (`ASR_LLM_ENABLE_THINKING` пусто → None; модель реджектит `reasoningEffort` → 400).
grok-4.3 (reasoning) — fallback: `ASR_LLM_MODEL=grok-4.3` + `ASR_LLM_ENABLE_THINKING=false` (reasoning
OFF обязателен, иначе loop на structured). Ключ — env `OR_KEY` (имя историческое, держит xAI-ключ).
Все значения берутся из env при импорте (env выставляется в launchd-plist / docker-compose).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from morag.llm.client import LLMClient


def _env(key: str, default: str = '') -> str:
    return os.environ.get(key, default)


def _flag(key: str, default: str = '1') -> bool:
    return _env(key, default) not in ('0', 'false', 'no', '')


def _enable_thinking() -> bool | None:
    """Reasoning-флаг для LLMClient (env `ASR_LLM_ENABLE_THINKING`). Пусто/none → None: НЕ слать
    reasoning-параметр (для non-reasoning моделей типа grok-4.20-non-reasoning он невалиден → 400).
    false → reasoning OFF (для reasoning-моделей типа grok-4.3, иначе loop на structured). true → ON."""
    v = _env('ASR_LLM_ENABLE_THINKING').strip().lower()
    if v in ('', 'none', 'null'):
        return None
    return v in ('1', 'true', 'yes', 'on')


@dataclass
class Config:
    # --- аудио-бэкенды на Маке (loopback при native-деплое; Caddy-URL при Docker/remote) ---
    diarizer_url: str = field(default_factory=lambda: _env('ASR_DIARIZER_URL', 'http://127.0.0.1:8090/diarize'))
    diarizer_key: str = field(default_factory=lambda: _env('ASR_DIARIZER_KEY'))
    asr_url: str = field(default_factory=lambda: _env('ASR_BACKEND_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions'))
    asr_key: str = field(default_factory=lambda: _env('ASR_BACKEND_KEY'))
    asr_model: str = field(default_factory=lambda: _env('ASR_MODEL', 'whisper-podlodka-turbo'))
    campp_url: str = field(default_factory=lambda: _env('ASR_CAMPP_URL', 'http://127.0.0.1:8126/embed-centroids'))
    campp_key: str = field(default_factory=lambda: _env('ASR_CAMPP_KEY'))

    # --- LLM: облако = Grok через xAI. Дефолт grok-4.20-0309-non-reasoning (быстрый non-reasoning,
    #     ~1.5с/вызов, качество коррекции = grok-4.3 — замерено). reasoning-флаг для него НЕ шлём (None).
    #     grok-4.3/OpenRouter — fallback через ASR_LLM_MODEL/ASR_LLM_BASE_URL + ASR_LLM_ENABLE_THINKING=false.
    #     Mac-native без облака: oMLX-local qwen3.6-35b (ASR_LLM_BASE_URL=http://127.0.0.1:8000/v1). ---
    llm_base_url: str = field(default_factory=lambda: _env('ASR_LLM_BASE_URL', 'https://api.x.ai/v1'))
    llm_model: str = field(default_factory=lambda: _env('ASR_LLM_MODEL', 'grok-4.20-0309-non-reasoning'))
    llm_key: str = field(default_factory=lambda: _env('OR_KEY'))  # имя env историческое; держит xAI-ключ

    # --- реестр спикеров (локальный JSON, см. CLAUDE.md — НЕ Qdrant) ---
    registry_path: str = field(default_factory=lambda: _env(
        'ASR_REGISTRY_PATH', os.path.expanduser('~/diar-test/speaker_registry.json')))
    match_threshold: float = field(default_factory=lambda: float(_env('ASR_MATCH_THRESHOLD', '0.55')))
    max_centroids: int = field(default_factory=lambda: int(_env('ASR_MAX_CENTROIDS', '8')))

    # --- прочее ---
    mode: str = field(default_factory=lambda: _env('ASR_MODE', 'async'))  # async | sync
    whisper_tokenizer: str = field(default_factory=lambda: _env('ASR_WHISPER_TOKENIZER', 'openai/whisper-large-v3'))
    prompt_budget: int = field(default_factory=lambda: int(_env('ASR_PROMPT_BUDGET', '200')))
    # авто-наминг Speaker_N → имя (интро-LLM + реестр). off → транскрипт остаётся в Speaker_N
    enable_naming: bool = field(default_factory=lambda: _flag('ASR_ENABLE_NAMING'))

    # --- покрытие звука расшифровкой (потери речи, см. stages/coverage.py) ---
    # hole_min_s — с какой длины считать непокрытый промежуток дырой (и добирать его чанками);
    # coverage_warn_s — расхождение «длительность реплики vs распознанное», с которого пишем WARN.
    hole_min_s: float = field(default_factory=lambda: float(_env('ASR_HOLE_MIN_S', '5')))
    coverage_warn_s: float = field(default_factory=lambda: float(_env('ASR_COVERAGE_WARN_S', '5')))
    recover_gaps: bool = field(default_factory=lambda: _flag('ASR_RECOVER_GAPS'))
    retry_empty: bool = field(default_factory=lambda: _flag('ASR_RETRY_EMPTY'))

    # --- пословное выравнивание (stages/align.py; торч ставится отдельно, см. requirements-align.txt) ---
    enable_align: bool = field(default_factory=lambda: _flag('ASR_ENABLE_ALIGN'))
    align_device: str = field(default_factory=lambda: _env('ASR_ALIGN_DEVICE'))  # '' → mps|cuda|cpu

    def build_llm(self) -> LLMClient:
        """morag LLMClient (облако = Grok). enable_thinking из ASR_LLM_ENABLE_THINKING (дефолт None —
        не слать reasoning-флаг; non-reasoning модель его реджектит. grok-4.3-fallback → =false)."""
        return LLMClient(base_url=self.llm_base_url, model=self.llm_model, api_key=self.llm_key,
                         enable_thinking=_enable_thinking(), timeout=180, max_retries=4)


CFG = Config()
