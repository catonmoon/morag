#!/usr/bin/env bash
# Забирает с машины-донора то, чего НЕТ в git: веса моделей и реестр голосов.
#
#   ./fetch-assets.sh user@host              # всё
#   ./fetch-assets.sh user@host registry     # только реестр (перед прогоном — он меняется)
#
# Почему копированием, а не скачиванием:
#   whisper-podlodka-turbo — своя MLX-конверсия (собиралась из HF-весов bond005), в сети её нет;
#   pyannote 3.1          — gated, требует HF-токена и принятия условий на сайте;
#   CAM++ onnx            — есть в сети, но 28 МБ рядом с остальным дешевле искать в браузере;
#   speaker_registry.json — ЕДИНСТВЕННОЕ настоящее состояние транскрибации, его негде взять.
#
# ⚠️ Реестр — не просто файл, а сквозная нумерация голосов. Копия на второй машине означает две
# расходящиеся истории: прогон здесь и прогон там дадут РАЗНЫХ людей под одним Speaker_N. Держать
# один рабочий экземпляр: забрал перед прогоном → прогнал → вернул (push-registry).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }
STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"

SRC="${1:-}"; WHAT="${2:-all}"
[[ -n "$SRC" ]] || { sed -n '2,8p' "$0"; exit 2; }

# Раскладка донора — историческая. Переопределяется переменными.
D_WHISPER="${DONOR_WHISPER:-.lmstudio/models/mlx-community/whisper-podlodka-turbo-mlx}"
D_CAMPP="${DONOR_CAMPP:-llm-stack/services/diarizer-onnx/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx}"
D_PYANNOTE="${DONOR_PYANNOTE:-.cache/torch/pyannote}"
D_REGISTRY="${DONOR_REGISTRY:-diar-test/speaker_registry.json}"

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

mkdir -p "$STACK_HOME"/{models,state} "$HOME/.cache/torch/pyannote"

# ⚠️ rsync на macOS — openrsync: он НЕ понимает --info=progress2 и падает с usage-подсказкой,
# которую легко принять за успех (ловили: «скопировалось» при пустом каталоге). Флагов минимум.
fetch() { rsync -a "$SRC:$1" "$2"; }

if [[ "$WHAT" == "all" || "$WHAT" == "models" ]]; then
  say "whisper-podlodka-turbo (MLX, ~1.6 ГБ)"
  fetch "$D_WHISPER/" "$STACK_HOME/models/whisper-podlodka-turbo/"
  ok "$(du -sh "$STACK_HOME/models/whisper-podlodka-turbo" | cut -f1)"

  say "CAM++ onnx"
  fetch "$D_CAMPP" "$STACK_HOME/models/"
  ok "готово"

  say "pyannote (gated — потому и копируем)"
  fetch "$D_PYANNOTE/" "$HOME/.cache/torch/pyannote/"
  ok "готово"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "registry" ]]; then
  say "реестр голосов"
  fetch "$D_REGISTRY" "$STACK_HOME/state/speaker_registry.json"
  python3 - "$STACK_HOME/state/speaker_registry.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
named = sum(1 for v in d['speakers'].values() if v.get('name'))
print(f"  ✓ next_id={d['next_id']}, голосов {len(d['speakers'])}, с именами {named}")
PY
fi

say "сверка контрольных сумм с донором"
for f in "models/whisper-podlodka-turbo/weights.safetensors" "state/speaker_registry.json"; do
  [[ -f "$STACK_HOME/$f" ]] || continue
  local_md5="$(md5 -q "$STACK_HOME/$f")"
  case "$f" in
    models/*) remote="$D_WHISPER/weights.safetensors" ;;
    state/*)  remote="$D_REGISTRY" ;;
  esac
  remote_md5="$(ssh "$SRC" "md5 -q '$remote'" 2>/dev/null || echo "?")"
  if [[ "$local_md5" == "$remote_md5" ]]; then ok "$f совпал"
  else printf '\033[31m  ✗ %s РАЗОШЁЛСЯ: %s ≠ %s\033[0m\n' "$f" "$local_md5" "$remote_md5"; fi
done
