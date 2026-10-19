#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_ROOT="${IMAGE_ROOT:-/media/xyycyc/Elements/dyx/TinyCLIP/my_code/data/OpenImage/meta/Hierarchy/n2/images_all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-ViT-B/32}"
MODEL_SLUG="${MODEL_NAME//\//-}"
IMAGE_LAYOUT="${IMAGE_LAYOUT:-relative}"
ALLOW_TRUNCATED_IMAGES="${ALLOW_TRUNCATED_IMAGES:-1}"
TRUNCATED_IMAGE_ARGS=()
if [[ "$ALLOW_TRUNCATED_IMAGES" == "1" ]]; then
  TRUNCATED_IMAGE_ARGS+=(--allow-truncated-images)
elif [[ "$ALLOW_TRUNCATED_IMAGES" != "0" ]]; then
  echo "ALLOW_TRUNCATED_IMAGES must be 0 or 1" >&2
  exit 2
fi
SKIP_BAD_IMAGES="${SKIP_BAD_IMAGES:-1}"
BAD_IMAGE_ARGS=()
if [[ "$SKIP_BAD_IMAGES" == "1" ]]; then
  BAD_IMAGE_ARGS+=(--skip-bad-images)
elif [[ "$SKIP_BAD_IMAGES" != "0" ]]; then
  echo "SKIP_BAD_IMAGES must be 0 or 1" >&2
  exit 2
fi
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/output/clip_retrieval_$MODEL_SLUG}"
DEVICE="${DEVICE:-cuda:3}"
PRECISION="${PRECISION:-auto}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/output/clip_retrieval_${MODEL_SLUG}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)_$$}"

printf 'Training configuration: device=%s batch_size=%s eval_batch_size=%s output=%s\n' \
  "$DEVICE" "$BATCH_SIZE" "$EVAL_BATCH_SIZE" "$OUTPUT_PATH"

"$PYTHON_BIN" my_code/train/finetune_clip_retrieval.py \
  --split-csv my_code/data/Image_IDs_n2_all_downloaded_split_8_1_1.csv \
  --caption-csv "my_code/data/Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv" \
  --image-root "$IMAGE_ROOT" \
  --image-layout "$IMAGE_LAYOUT" \
  "${TRUNCATED_IMAGE_ARGS[@]}" \
  "${BAD_IMAGE_ARGS[@]}" \
  --output-dir "$OUTPUT_PATH" \
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
  "$@"
