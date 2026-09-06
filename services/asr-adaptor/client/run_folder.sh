#!/usr/bin/env bash
# Расшифровывает ПАПКУ с записями: аудио и видео вперемешку, жанр знать заранее не нужно.
#
#   ASR_BASE=http://127.0.0.1:8082 ./run_folder.sh ~/Records
#   ASR_BASE=... ./run_folder.sh ~/Records --dry        # только показать, что будет сделано
#
# Результат кладётся РЯДОМ с исходником: meeting.mp4 → meeting.md и meeting.json.
# Пословные тайм-коды лежат ВНУТРИ json (`x_enriched.words`), отдельным файлом их достаёт уже
# доменный шаг (в подкасте это align.sh) — здесь он не нужен.
# Идемпотентно: запись с готовым `.json` пропускается, так что папку можно гонять повторно и
# доливать новые файлы. Последовательно: реестр голосов общий, параллельный прогон разъедет
# нумерацию Speaker_N.
#
# Чем это отличается от run_corpus.sh: тот гоняет ПРОНУМЕРОВАННЫЕ выпуски по URL-шаблону
# (подкаст), а этот — произвольные файлы, у которых есть только имя.
#
# Настройки корпуса (описание материала, термины, свой реестр голосов) живут в окружении адаптера,
# а не здесь — см. deploy/mac/README.md, раздел про профили. А знание об ОДНОЙ записи (верные
# написания её терминов и фамилий) кладётся файлом рядом: `<имя>.hints.json`.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:-}"
DRY="${2:-}"
[[ -d "$DIR" ]] || { sed -n '2,16p' "$0"; exit 2; }
: "${ASR_BASE:?set ASR_BASE, напр. http://127.0.0.1:8082}"

# Расширения, которые умеет transcribe_one.sh (видео он сам разожмёт в дорожку через ffmpeg).
EXTS="mp3 m4a wav aac ogg opus flac mp4 mkv mov webm avi m4v"

# ⚠️ Без mapfile/readarray: macOS поставляет bash 3.2, где их нет вовсе («mapfile: command not
# found»), а `#!/usr/bin/env bash` на большинстве маков берёт именно его. Читаем циклом с
# разделителем \0 — имена записей бывают с пробелами («Планёрка 12 марта.mp4»).
FILES=()
while IFS= read -r -d '' f; do
  FILES+=("$f")
done < <(
  for e in $EXTS; do
    find "$DIR" -type f -iname "*.${e}" ! -name "._*" -print0 2>/dev/null
  done | sort -z
)

[[ ${#FILES[@]} -gt 0 ]] || { echo "в $DIR не нашлось записей (искали: $EXTS)"; exit 0; }

echo "=== папка: $DIR"
echo "=== записей найдено: ${#FILES[@]}"

todo=() skip=()
for f in "${FILES[@]}"; do
  base="${f%.*}"
  if [[ -s "$base.json" ]]; then skip+=("$f"); else todo+=("$f"); fi
done

echo "=== уже расшифровано: ${#skip[@]} | к работе: ${#todo[@]}"
if [[ "$DRY" == "--dry" ]]; then
  for f in "${todo[@]}"; do echo "  будет расшифровано: $(basename "$f")"; done
  exit 0
fi
[[ ${#todo[@]} -gt 0 ]] || { echo "всё готово, работы нет"; exit 0; }

failed=0
for f in "${todo[@]}"; do
  name="$(basename "$f")"
  dir="$(dirname "$f")"
  stem="${name%.*}"
  echo "--- $name @ $(date +%H:%M:%S)"

  # transcribe_one.sh кладёт результат как ep<N>.{md,json} в OUT_DIR и требует номер. Даём ему
  # временный каталог и номер 1, а потом переносим под ИМЯ файла: у записей нет нумерации, и
  # «ep1.md» рядом с «Планёрка 12 марта.mp4» ничего не сказал бы.
  # Подсказки записи — файл рядом со звуком: `Планёрка.mp4` → `Планёрка.hints.json`. Расширение
  # выбрано так, чтобы не спутать клиента: сам он ищет только аудио, а готовность проверяет по
  # `<имя>.json` — `<имя>.hints.json` для него ни запись, ни артефакт.
  hints="$dir/$stem.hints.json"
  [ -s "$hints" ] && echo "    подсказки: $(basename "$hints")"

  work="$(mktemp -d)"
  if OUT_DIR="$work/out" CACHE_DIR="$work/cache" TITLE_TEMPLATE="$stem" SEASON=0 SEASON_PREFIX="" \
     HINTS_FILE="$hints" \
     bash "$HERE/transcribe_one.sh" 1 "$f"; then
    for ext in md json; do
      [[ -f "$work/out/ep1.$ext" ]] && mv "$work/out/ep1.$ext" "$dir/$stem.$ext"
    done
    echo "    → $stem.md"
  else
    echo "    FAILED: $name (продолжаю)"
    failed=$((failed + 1))
  fi
  rm -rf "$work"
done

echo "=== готово @ $(date +%H:%M:%S): расшифровано $(( ${#todo[@]} - failed )), сорвалось $failed"
[[ $failed -eq 0 ]]
