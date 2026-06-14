#!/bin/bash
# Запуск транскрайб-бэкенда на Маке. ffmpeg в PATH обязателен (mlx_whisper зовёт его).
# venv-whisper должен иметь: mlx-whisper, fastapi, uvicorn, python-multipart.
set -e
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")"
PORT="${TRANSCRIBE_PORT:-8123}"
exec ~/diar-test/.venv-whisper/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT"
