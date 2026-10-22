#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-ViT-B/32}"
MODEL_SLUG="${MODEL_NAME//\//-}"
DEVICE="${DEVICE:-cuda:3}"
PRECISION="${PRECISION:-auto}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
OUTPUT_ARGS=()
OUTPUT_LABEL="checkpoint directory"
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "$OUTPUT_PATH")
  OUTPUT_LABEL="$OUTPUT_PATH"
else
  RESUMING=0
  for ARG in "$@"; do
    if [[ "$ARG" == --resume || "$ARG" == --resume=* ]]; then
      RESUMING=1
      break
    fi
  done
  if [[ "$RESUMING" == "0" ]]; then
    OUTPUT_PATH="$SCRIPT_DIR/output/clip_flickr30k_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)_$$"
    OUTPUT_ARGS=(--output-dir "$OUTPUT_PATH")
    OUTPUT_LABEL="$OUTPUT_PATH"
  fi
fi

printf 'Flickr30K training: model=%s device=%s batch=%s output=%s\n' \
  "$MODEL_NAME" "$DEVICE" "$BATCH_SIZE" "$OUTPUT_LABEL"

"$PYTHON_BIN" my_code/train/finetune_clip_flickr30k.py \
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
  "${OUTPUT_ARGS[@]}" \
  "$@"
