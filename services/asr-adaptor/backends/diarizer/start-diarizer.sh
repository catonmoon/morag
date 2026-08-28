#!/usr/bin/env bash
# Запуск сервиса диаризации, паттерн как у start-omlx.sh.
set -euo pipefail

STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
SERVICE_DIR="${SERVICE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"

LOG_DIR="${LOG_DIR:-$STACK_HOME/logs}"
RUN_DIR="${RUN_DIR:-$STACK_HOME/run}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/diarizer.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/diarizer.pid}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "diarizer already running (pid=$(cat "$PID_FILE"))."
  exit 0
fi

# Подгружаем секреты из ~/.diarizer.env (DIARIZER_API_KEY, HF_TOKEN)
if [[ -f "$HOME/.diarizer.env" ]]; then
  set -a
  source "$HOME/.diarizer.env"
  set +a
fi

if [[ -z "${DIARIZER_API_KEY:-}" ]]; then
  echo "ERROR: DIARIZER_API_KEY not set (put it in ~/.diarizer.env)"
  exit 1
fi

echo "Starting diarizer service:"
echo "  host:   $HOST"
echo "  port:   $PORT"
echo "  log:    $LOG_FILE"

cd "$SERVICE_DIR"
# venv отдельной переменной: на машине с раскладкой deploy/mac он лежит не внутри SERVICE_DIR
VENV="${DIARIZER_VENV:-$STACK_HOME/venvs/diarizer}"
nohup "$VENV/bin/uvicorn" app:app \
  --host "$HOST" --port "$PORT" \
  --log-level info \
  >>"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started (pid=$(cat "$PID_FILE"))."
