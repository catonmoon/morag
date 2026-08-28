#!/usr/bin/env bash
# Запуск CAM++ embed-бэкенда (:8126). Паттерн как у start-diarizer.sh — pid/лог в файлы.
#
# Окружение (дефолты — историческая раскладка):
#   CAMPP_VENV     venv с sherpa-onnx      (~/diar-test/.venv-qwenasr)
#   CAMPP_MODEL    путь к .onnx CAM++      (~/llm-stack/services/diarizer-onnx/models/…advanced.onnx) — читает app.py
#   CAMPP_API_KEY  Bearer (обязателен)     — из ~/.asr-stack.env или окружения
#   HOST/PORT      где слушать             (127.0.0.1:8126)
set -euo pipefail

SERVICE_DIR="${CAMPP_SERVICE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV="${CAMPP_VENV:-$HOME/diar-test/.venv-qwenasr}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8126}"

LOG_DIR="${LOG_DIR:-$HOME/llm-stack/logs}"
RUN_DIR="${RUN_DIR:-$HOME/llm-stack/run}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/campp.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/campp.pid}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "campp already running (pid=$(cat "$PID_FILE"))."
  exit 0
fi

if [[ -z "${CAMPP_API_KEY:-}" ]]; then
  echo "ERROR: CAMPP_API_KEY not set" >&2
  exit 1
fi

cd "$SERVICE_DIR"
nohup "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT" --log-level info \
  >>"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started campp (pid=$(cat "$PID_FILE"), $HOST:$PORT)."
