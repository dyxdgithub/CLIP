#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-ViT-B/32}"
MODEL_SLUG="${MODEL_NAME//\//-}"
DEVICE="${DEVICE:-cuda:1}"
PRECISION="${PRECISION:-auto}"
OBJECTIVE="${OBJECTIVE:-sct}"
SCT_LAMBDA="${SCT_LAMBDA:-1.0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

if [[ "$OBJECTIVE" != "sct" && "$OBJECTIVE" != "hn" ]]; then
  echo "OBJECTIVE must be sct or hn" >&2
  exit 2
fi

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
    OUTPUT_PATH="$OUTPUT_ROOT/clip_flickr30k_${OBJECTIVE}_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)_$$"
    OUTPUT_ARGS=(--output-dir "$OUTPUT_PATH")
    OUTPUT_LABEL="$OUTPUT_PATH"
  fi
fi

printf 'Flickr30K %s: model=%s device=%s batch=%s lambda=%s output=%s HF_ENDPOINT=%s\n' \
  "$OBJECTIVE" "$MODEL_NAME" "$DEVICE" "$BATCH_SIZE" "$SCT_LAMBDA" "$OUTPUT_LABEL" "$HF_ENDPOINT"

"$PYTHON_BIN" my_code/train/finetune_clip_flickr30k_sct.py \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --objective "$OBJECTIVE" \
  --sct-lambda "$SCT_LAMBDA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LEARNING_RATE" \
  --seed "$SEED" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
