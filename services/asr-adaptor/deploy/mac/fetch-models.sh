#!/usr/bin/env bash
# Скачивает модели ASR-стека из сети.
#
#   ./fetch-models.sh                 # всё, чего не хватает
#   ./fetch-models.sh whisper         # только распознавание
#   ./fetch-models.sh campp           # только эмбеддер голосов
#   ./fetch-models.sh pyannote        # только диаризация (нужен HF_TOKEN)
#   FORCE=1 ./fetch-models.sh         # перекачать даже то, что уже есть
#
# Идемпотентно: файл нужного размера повторно не качается.
#
# Что откуда:
#   whisper  → chukanov/whisper-podlodka-turbo-mlx   (публичный, MLX-конверсия bond005)
#   CAM++    → релизы sherpa-onnx на GitHub          (публичный)
#   pyannote → pyannote/*                            (ЧАСТЬЮ gated, нужен HF_TOKEN)
#
# Реестра голосов здесь нет и быть не может: это не модель, а состояние корпуса. Новый корпус
# начинает с пустого реестра — он заведётся сам при первом прогоне.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"
# ⚠️ Вместе с путями отсюда подхватится и HTTPS_PROXY, если он там задан: профиль пишется под
# LLM-стадии, а качать через него модели обычно медленнее. Не устраивает — запускайте как
# `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY ./fetch-models.sh`.
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
MODELS="$STACK_HOME/models"
# Кэш pyannote — СОБСТВЕННЫЙ дефолт библиотеки (PYANNOTE_CACHE), а НЕ кэш huggingface_hub.
# Скачанное в ~/.cache/huggingface/hub библиотека не увидит, и стек будет выглядеть «без моделей».
PYANNOTE_CACHE="${PYANNOTE_CACHE:-$HOME/.cache/torch/pyannote}"

WHAT="${1:-all}"
FORCE="${FORCE:-}"

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m  ✗ %s\033[0m\n' "$*"; }

mkdir -p "$MODELS"

# размер файла в байтах, 0 если нет (BSD stat — на macOS другой синтаксис, чем в GNU)
size_of() { [[ -f "$1" ]] && stat -f%z "$1" 2>/dev/null || echo 0; }

# человекочитаемый размер: у модели файлы от 268 Б до 1.5 ГиБ, и «0 МиБ» для конфига — не ответ
human() {
  local b="$1"
  if   [[ "$b" -ge 1048576 ]]; then printf '%d МиБ' $((b / 1048576))
  elif [[ "$b" -ge 1024 ]];    then printf '%d КиБ' $((b / 1024))
  else printf '%d Б' "$b"; fi
}

# fetch_file <url> <куда> <человекочитаемое имя>
# Качает во временный файл рядом и переставляет на место: оборванная закачка не должна оставлять
# после себя огрызок, который на следующем запуске сойдёт за готовый файл.
fetch_file() {
  local url="$1" dest="$2" label="$3"
  local remote local_size
  local_size="$(size_of "$dest")"
  remote="$(curl -sIL -m 60 "$url" 2>/dev/null | grep -i '^content-length:' | tail -1 \
            | tr -d '\r' | awk '{print $2}')"
  remote="${remote:-0}"

  if [[ -z "$FORCE" && "$local_size" -gt 0 ]]; then
    if [[ "$remote" -eq 0 || "$local_size" -eq "$remote" ]]; then
      ok "$label — уже на месте ($(human "$local_size"))"
      return 0
    fi
    warn "$label — размер разошёлся (локально $local_size, в сети $remote), качаю заново"
  fi

  say "$label"
  if curl -fL --retry 3 --retry-delay 2 -m 3600 --progress-bar -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
    ok "$label — $(human "$(size_of "$dest")")"
  else
    rm -f "$dest.part"
    bad "$label — не скачалось: $url"
    return 1
  fi
}

# --- whisper: MLX-конверсия, публичная ---------------------------------------
fetch_whisper() {
  local repo="${WHISPER_HF_REPO:-chukanov/whisper-podlodka-turbo-mlx}"
  local dir="$MODELS/whisper-podlodka-turbo"
  mkdir -p "$dir"
  say "whisper: $repo (~1.5 ГиБ)"
  # Список явный, а не «весь репозиторий»: так видно, что именно нужно бэкенду, и случайный
  # новый файл в репозитории не утянется молча.
  local f
  for f in weights.safetensors config.json generation_config.json preprocessor_config.json \
           tokenizer.json tokenizer_config.json vocab.json added_tokens.json \
           special_tokens_map.json normalizer.json; do
    fetch_file "https://huggingface.co/$repo/resolve/main/$f" "$dir/$f" "  $f" || return 1
  done
  ok "whisper готов: $dir"
}

# --- CAM++: ONNX-сборка 3D-Speaker от проекта sherpa-onnx --------------------
fetch_campp() {
  local name="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  # ⚠️ «recongition» в URL — опечатка апстрима в имени релиза, так и должно быть.
  local url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$name"
  say "CAM++ (~27 МиБ)"
  if ! fetch_file "$url" "$MODELS/$name" "  $name"; then
    warn "GitHub отдаёт файл с release-assets.githubusercontent.com — некоторые сети его режут."
    warn "Проверить: curl -sIL '$url' | tail -3. Помогает другая сеть или прокси."
    return 1
  fi
}

# --- pyannote: часть репозиториев gated --------------------------------------
fetch_pyannote() {
  say "pyannote → $PYANNOTE_CACHE"
  # Качаем не curl'ом: библиотека читает кэш huggingface_hub (blobs/snapshots/refs), а не папку
  # с файлами. Собирать эту раскладку руками — способ получить кэш, который никто не прочитает.
  local py="" cand
  for cand in "$STACK_HOME/venvs/diarizer/bin/python" "$STACK_HOME/venvs/adaptor/bin/python" \
              "$STACK_HOME/venvs/whisper/bin/python" \
              "$HOME/asr-stack/venvs/diarizer/bin/python" \
              /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import huggingface_hub" 2>/dev/null; then
      py="$cand"; break
    fi
  done
  if [[ -z "$py" ]]; then
    warn "не нашёл python с huggingface_hub. Варианты: сначала ./install.sh (он создаст venv-ы)"
    warn "и повторить, либо поставить пакет вручную: pip3 install huggingface_hub"
    return 1
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN пуст. Свободно качается только эмбеддер (wespeaker-voxceleb-resnet34-LM);"
    warn "segmentation-3.0 и speaker-diarization-3.1 — gated, без токена НЕ скачаются."
    warn "Что делать: завести токен на huggingface.co/settings/tokens, принять условия на"
    warn "  huggingface.co/pyannote/segmentation-3.0 и /pyannote/speaker-diarization-3.1,"
    warn "  вписать HF_TOKEN в $ENV_FILE и повторить."
  fi

  PYANNOTE_CACHE="$PYANNOTE_CACHE" FORCE="${FORCE:-}" "$py" - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

cache = os.environ["PYANNOTE_CACHE"]
token = os.environ.get("HF_TOKEN") or None
# speaker-diarization-3.1 — это конфиг пайплайна; веса лежат в двух других репозиториях,
# и их он тянет по именам из своего config.yaml.
REPOS = [
    ("pyannote/speaker-diarization-3.1", True,  ["config.yaml"]),
    ("pyannote/segmentation-3.0",        True,  ["config.yaml", "pytorch_model.bin"]),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", False, ["config.yaml", "pytorch_model.bin"]),
]
failed = []
for repo, gated, files in REPOS:
    try:
        snapshot_download(repo_id=repo, cache_dir=cache, token=token, allow_patterns=files)
        print(f"\033[32m  ✓ {repo}\033[0m")
    except Exception as e:
        first = str(e).strip().splitlines()[0][:120]
        print(f"\033[31m  ✗ {repo}: {first}\033[0m")
        failed.append((repo, gated))

if failed:
    print()
    for repo, gated in failed:
        if gated and not token:
            print(f"\033[33m  ! {repo} требует HF_TOKEN и принятых условий на huggingface.co/{repo}\033[0m")
        elif gated:
            print(f"\033[33m  ! {repo}: токен есть — примите условия на huggingface.co/{repo}\033[0m")
    sys.exit(1)
PY
}

rc=0
case "$WHAT" in
  all)      fetch_whisper || rc=1; fetch_campp || rc=1; fetch_pyannote || rc=1 ;;
  whisper)  fetch_whisper || rc=1 ;;
  campp)    fetch_campp || rc=1 ;;
  pyannote) fetch_pyannote || rc=1 ;;
  *)        sed -n '2,18p' "$0"; exit 2 ;;
esac

echo
if [[ $rc -eq 0 ]]; then
  say "модели на месте"
  echo "  дальше: $HERE/install.sh --check   (должен показать зелёное по моделям)"
else
  say "часть моделей не скачалась — см. сообщения выше"
fi
# ⚠️ MMS_FA для пословного выравнивания здесь не качаем: это бандл torchaudio, он приезжает сам
# при первом выравнивании (~1.3 ГБ в ~/.cache/torch/hub/checkpoints/model.pt).
exit $rc
