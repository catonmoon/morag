#!/bin/bash
# Запуск транскрайб-бэкенда на Маке. ffmpeg в PATH обязателен (mlx_whisper зовёт его).
# venv должен иметь: mlx-whisper, fastapi, uvicorn, python-multipart.
#
# Пути настраиваются окружением (дефолты — историческая раскладка):
#   TRANSCRIBE_VENV       venv с mlx-whisper           ($ASR_STACK_HOME/venvs/whisper)
#   TRANSCRIBE_MODELS_DIR каталог с MLX-весами         ($ASR_STACK_HOME/models) — читает app.py
#   TRANSCRIBE_PORT       порт                         (8123)
#   TRANSCRIBE_HOST       адрес прослушивания          (loopback; исторический дефолт — 0.0.0.0)
#
# ⚠️ Дефолт host сменён с 0.0.0.0 на loopback: адаптер и так ходит сюда по 127.0.0.1, а бэкенд
# на ноутбуке в чужой сети — открытый ASR за одним Bearer. Прежнее — TRANSCRIBE_HOST=0.0.0.0.
set -e
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")"
PORT="${TRANSCRIBE_PORT:-8123}"
STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
VENV="${TRANSCRIBE_VENV:-$STACK_HOME/venvs/whisper}"
exec "$VENV/bin/uvicorn" app:app --host "${TRANSCRIBE_HOST:-127.0.0.1}" --port "$PORT"
