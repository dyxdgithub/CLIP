#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CSV_DIR="${CSV_DIR:-$REPO_ROOT/my_code/data/OpenImages}"
IMAGE_ROOT="${IMAGE_ROOT:-$REPO_ROOT/my_code/data/OpenImages}"
IMAGE_SOURCE="${IMAGE_SOURCE:-local}"
ONLINE_CACHE_DIR="${ONLINE_CACHE_DIR:-}"
ONLINE_TIMEOUT="${ONLINE_TIMEOUT:-15}"
ONLINE_RETRIES="${ONLINE_RETRIES:-2}"
ONLINE_MAX_IMAGE_MB="${ONLINE_MAX_IMAGE_MB:-32}"
MODEL_NAME="${MODEL_NAME:-ViT-B/32}"
MODEL_SLUG="${MODEL_NAME//\//-}"
DEVICE="${DEVICE:-cuda:2}"
PRECISION="${PRECISION:-auto}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

ALLOW_TRUNCATED_IMAGES="${ALLOW_TRUNCATED_IMAGES:-1}"
SKIP_BAD_IMAGES="${SKIP_BAD_IMAGES:-1}"
SKIP_MISSING_IMAGES="${SKIP_MISSING_IMAGES:-0}"
IMAGE_ARGS=()
if [[ "$IMAGE_SOURCE" != "local" && "$IMAGE_SOURCE" != "online" ]]; then
  echo "IMAGE_SOURCE must be local or online" >&2
  exit 2
fi
if [[ -n "$ONLINE_CACHE_DIR" ]]; then
  IMAGE_ARGS+=(--online-cache-dir "$ONLINE_CACHE_DIR")
fi
if [[ "$ALLOW_TRUNCATED_IMAGES" == "1" ]]; then
  IMAGE_ARGS+=(--allow-truncated-images)
elif [[ "$ALLOW_TRUNCATED_IMAGES" != "0" ]]; then
  echo "ALLOW_TRUNCATED_IMAGES must be 0 or 1" >&2
  exit 2
fi
if [[ "$SKIP_BAD_IMAGES" == "1" ]]; then
  IMAGE_ARGS+=(--skip-bad-images)
elif [[ "$SKIP_BAD_IMAGES" != "0" ]]; then
  echo "SKIP_BAD_IMAGES must be 0 or 1" >&2
  exit 2
fi
if [[ "$SKIP_MISSING_IMAGES" == "1" ]]; then
  IMAGE_ARGS+=(--skip-missing-images)
elif [[ "$SKIP_MISSING_IMAGES" != "0" ]]; then
  echo "SKIP_MISSING_IMAGES must be 0 or 1" >&2
  exit 2
fi

OUTPUT_ARGS=()
OUTPUT_LABEL="checkpoint directory"
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "$OUTPUT_PATH")
  OUTPUT_LABEL="$OUTPUT_PATH"
else
  EXPLICIT_OUTPUT=0
  for ARG in "$@"; do
    if [[ "$ARG" == --resume || "$ARG" == --resume=* ||
          "$ARG" == --output-dir || "$ARG" == --output-dir=* ]]; then
      EXPLICIT_OUTPUT=1
      break
    fi
  done
  if [[ "$EXPLICIT_OUTPUT" == "0" ]]; then
    OUTPUT_PATH="$OUTPUT_ROOT/clip_openimages_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)_$$"
    OUTPUT_ARGS=(--output-dir "$OUTPUT_PATH")
    OUTPUT_LABEL="$OUTPUT_PATH"
  fi
fi

printf 'Open Images training: CSV=%s images=%s source=%s model=%s device=%s batch=%s output=%s\n' \
  "$CSV_DIR" "$IMAGE_ROOT" "$IMAGE_SOURCE" "$MODEL_NAME" "$DEVICE" "$BATCH_SIZE" "$OUTPUT_LABEL"

"$PYTHON_BIN" my_code/train/finetune_clip_openimages.py \
  --csv-dir "$CSV_DIR" \
  --image-root "$IMAGE_ROOT" \
  --image-source "$IMAGE_SOURCE" \
  --online-timeout "$ONLINE_TIMEOUT" \
  --online-retries "$ONLINE_RETRIES" \
  --online-max-image-mb "$ONLINE_MAX_IMAGE_MB" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LEARNING_RATE" \
  --seed "$SEED" \
  "${IMAGE_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
