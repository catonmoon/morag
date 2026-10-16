#!/usr/bin/env bash
# Поднять / погасить / проверить Mac-стек транскрибации — без launchd.
#
#   ./stack.sh up       поднять четыре сервиса (диаризатор грузит модель ~90 с)
#   ./stack.sh down     погасить всё
#   ./stack.sh status   кто слушает порты
#   ./stack.sh health   опросить /health адаптера (он сам пингует downstream)
#   ./stack.sh logs     хвосты логов
#
# Почему не launchd с KeepAlive, как на стационарном инстансе: там памяти вдоволь и стек живёт
# всегда. На ноутбуке четыре резидентных сервиса с моделями съедают память впустую, а
# транскрибация нужна раз в неделю. Поднял → прогнал → погасил.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ADAPTOR_DIR="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"
[[ -f "$ENV_FILE" ]] || { echo "нет $ENV_FILE — сначала ./install.sh" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a

# Профиль корпуса проекта (ADR-0023): доменное — описание материала, постоянные термины, потолок
# голосов — живёт в git репозитория проекта, а секреты остаются в env-файле машины. Читается ПОСЛЕ
# него и потому перекрывает: профиль отвечает за «что за материал», env-файл — за «чем и куда ходить».
#   ASR_PROFILE=<репозиторий-проекта>/<корпус>/asr-profile.env ./stack.sh up
if [[ -n "${ASR_PROFILE:-}" ]]; then
  [[ -f "$ASR_PROFILE" ]] || { echo "ASR_PROFILE указывает в никуда: $ASR_PROFILE" >&2; exit 1; }
  set -a; source "$ASR_PROFILE"; set +a
fi

STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
VENVS="$STACK_HOME/venvs"
LOGS="$STACK_HOME/logs"; RUN="$STACK_HOME/run"
mkdir -p "$LOGS" "$RUN"

# Бэкенды ждут ключи под своими именами; в env они же лежат под ASR_*_KEY.
export TRANSCRIBE_API_KEY="${TRANSCRIBE_API_KEY:-${ASR_BACKEND_KEY:-}}"
export CAMPP_API_KEY="${CAMPP_API_KEY:-${ASR_CAMPP_KEY:-}}"
export DIARIZER_API_KEY="${DIARIZER_API_KEY:-${ASR_DIARIZER_KEY:-}}"
export ASR_REGISTRY_PATH="${ASR_REGISTRY_PATH:-$STACK_HOME/state/speaker_registry.json}"

# Диаризатор и выравниватель считают локально; наружу ходят только LLM-стадии адаптера.
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m  ✗ %s\033[0m\n' "$*"; }

PORT_ASR=8082 PORT_DIAR=8090 PORT_WHISPER=8123 PORT_CAMPP=8126
# Пул whisper (`ASR_WHISPER_INSTANCES`, по умолчанию 1): процессы на 8123, 8133, 8143… — у каждого
# своя модель в памяти (turbo ~1.6 ГБ, со вторым ухом в кэше ~4.6 ГБ). Адаптер выводит те же адреса
# из той же переменной и отдаёт кусок самому свободному. Потолок — 8 (порты до 8193).
WHISPER_N="${ASR_WHISPER_INSTANCES:-1}"
(( WHISPER_N >= 1 && WHISPER_N <= 8 )) || { echo "ASR_WHISPER_INSTANCES=$WHISPER_N: нужно 1..8" >&2; exit 1; }
whisper_port() { echo $(( PORT_WHISPER + 10 * $1 )); }
whisper_name() { (( $1 == 0 )) && echo whisper || echo "whisper$1"; }

# Полосы (`ASR_LANES="cpu=cpu,1,http://127.0.0.1:8223;…"`, см. config.py): на каждый ЛОКАЛЬНЫЙ адрес
# полосы — свой процесс того же бэкенда (`lane-<имя>[k]`, порт из адреса); удалённый адрес — просто
# адрес. Так модель другой школы (GigaAM на CPU) слушает в своём процессе и не стоит в очереди whisper.
lane_entries() {               # печатает строки «имя порт» по локальным адресам всех полос
  local spec name rest url host port k
  IFS=';' read -ra _SPECS <<< "${ASR_LANES:-}"
  for spec in "${_SPECS[@]}"; do
    spec="${spec// /}"; [[ -z "$spec" ]] && continue
    name="${spec%%=*}"; rest="${spec#*=}"
    IFS=',' read -ra _P <<< "$rest"
    k=0
    for url in "${_P[@]:2}"; do
      host="${url#*://}"; host="${host%%/*}"; port="${host##*:}"; host="${host%%:*}"
      if [[ "$host" == 127.0.0.1 || "$host" == localhost ]]; then
        echo "lane-$name$([[ $k -gt 0 ]] && echo "$k") $port"
      fi
      k=$((k+1))
    done
  done
}

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

start_one() {                   # start_one <имя> <порт> <лог> <команда...>
  local name="$1" port="$2" log="$3"; shift 3
  if listening "$port"; then ok "$name уже слушает :$port"; return 0; fi
  say "$name → :$port"
  ( "$@" >>"$log" 2>&1 & echo $! > "$RUN/$name.pid" )
  sleep 1
  ok "$name запущен (pid $(cat "$RUN/$name.pid")), лог $log"
}

wait_port() {                   # wait_port <порт> <секунд> <имя>
  local port="$1" limit="$2" name="$3" i=0
  while (( i < limit )); do listening "$port" && { ok "$name отвечает"; return 0; }; sleep 2; i=$((i+2)); done
  bad "$name не поднялся за ${limit}с — смотри лог"
  return 1
}

cmd_up() {
  # Диаризатор первым: он дольше всех грузит модель.
  DIARIZER_DEVICE="${DIARIZER_DEVICE:-auto}" \
  start_one diarizer "$PORT_DIAR" "$LOGS/diarizer.log" \
    "$VENVS/diarizer/bin/uvicorn" --app-dir "$ADAPTOR_DIR/backends/diarizer" app:app \
    --host 127.0.0.1 --port "$PORT_DIAR" --log-level info

  CAMPP_MODEL="${CAMPP_MODEL:-$STACK_HOME/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx}" \
  start_one campp "$PORT_CAMPP" "$LOGS/campp.log" \
    "$VENVS/campp/bin/uvicorn" --app-dir "$ADAPTOR_DIR/backends/campp" app:app \
    --host 127.0.0.1 --port "$PORT_CAMPP" --log-level info

  local k
  for (( k = 0; k < WHISPER_N; k++ )); do
    PATH="/opt/homebrew/bin:$PATH" \
    TRANSCRIBE_MODELS_DIR="${TRANSCRIBE_MODELS_DIR:-$STACK_HOME/models}" \
    start_one "$(whisper_name $k)" "$(whisper_port $k)" "$LOGS/$(whisper_name $k).log" \
      "$VENVS/whisper/bin/uvicorn" --app-dir "$ADAPTOR_DIR/backends/transcribe" app:app \
      --host 127.0.0.1 --port "$(whisper_port $k)" --log-level info
  done

  # Бэкенды полос — тот же код, свой порт; модели грузятся по первому запросу (прогрев — /warmup).
  local lane_name lane_port
  while read -r lane_name lane_port; do
    [[ -z "$lane_name" ]] && continue
    PATH="/opt/homebrew/bin:$PATH" \
    TRANSCRIBE_MODELS_DIR="${TRANSCRIBE_MODELS_DIR:-$STACK_HOME/models}" \
    start_one "$lane_name" "$lane_port" "$LOGS/$lane_name.log" \
      "$VENVS/whisper/bin/uvicorn" --app-dir "$ADAPTOR_DIR/backends/transcribe" app:app \
      --host 127.0.0.1 --port "$lane_port" --log-level info
  done < <(lane_entries)

  # Адаптер последним — он на старте пингует downstream.
  start_one adaptor "$PORT_ASR" "$LOGS/adaptor.log" \
    "$VENVS/adaptor/bin/uvicorn" --app-dir "$ADAPTOR_DIR" app:app \
    --host 127.0.0.1 --port "$PORT_ASR" --log-level info

  say "ждём готовности (диаризатор грузит модель до ~90 с)"
  wait_port "$PORT_CAMPP"   30 campp    || true
  for (( k = 0; k < WHISPER_N; k++ )); do wait_port "$(whisper_port $k)" 60 "$(whisper_name $k)" || true; done
  while read -r lane_name lane_port; do
    [[ -n "$lane_name" ]] && { wait_port "$lane_port" 60 "$lane_name" || true; }
  done < <(lane_entries)
  wait_port "$PORT_ASR"     60 adaptor  || true
  wait_port "$PORT_DIAR"   180 diarizer || true
  cmd_health
}

cmd_down() {
  # Пул whisper гасится целиком, даже если ASR_WHISPER_INSTANCES с тех пор уменьшили.
  local k
  # Полосы — по pid-файлам `lane-*`: гасятся и те, которых в ASR_LANES уже нет.
  local lanes=(); for f in "$RUN"/lane-*.pid; do [[ -f "$f" ]] && lanes+=("$(basename "$f" .pid)"); done
  for name in adaptor whisper whisper{1..7} "${lanes[@]}" campp diarizer; do
    [[ $name == whisper? && ! -f "$RUN/$name.pid" ]] && continue
    local pid_file="$RUN/$name.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      kill "$(cat "$pid_file")" && ok "$name остановлен"
    else
      ok "$name не запущен"
    fi
    rm -f "$pid_file"
  done
  # uvicorn мог родить воркеров — добираем по порту.
  for p in $PORT_ASR $(for k in {0..7}; do whisper_port $k; done) $(lane_entries | awk '{print $2}') $PORT_CAMPP $PORT_DIAR; do
    local pids; pids="$(lsof -nP -iTCP:"$p" -sTCP:LISTEN -t 2>/dev/null || true)"
    [[ -n "$pids" ]] && { echo "$pids" | xargs kill 2>/dev/null || true; ok "порт $p освобождён"; }
  done
}

cmd_status() {
  local pairs=("адаптер:$PORT_ASR" "диаризатор:$PORT_DIAR" "CAM++:$PORT_CAMPP") k
  for (( k = 0; k < WHISPER_N; k++ )); do pairs+=("$(whisper_name $k):$(whisper_port $k)"); done
  while read -r lane_name lane_port; do [[ -n "$lane_name" ]] && pairs+=("$lane_name:$lane_port"); done < <(lane_entries)
  for pair in "${pairs[@]}"; do
    local name="${pair%%:*}" port="${pair##*:}"
    if listening "$port"; then
      local rss; rss="$(ps -o rss= -p "$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | head -1)" 2>/dev/null | tr -d ' ')"
      ok "$name :$port — $(( ${rss:-0} / 1024 )) МБ"
    else bad "$name :$port — не слушает"; fi
  done
}

cmd_health() {
  say "health адаптера"
  curl -s -m 20 "http://127.0.0.1:$PORT_ASR/health" | python3 -m json.tool 2>/dev/null \
    || bad "адаптер не ответил"
}

cmd_logs() { tail -n 25 "$LOGS"/*.log; }

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  health) cmd_health ;;
  logs) cmd_logs ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
