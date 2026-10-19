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
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/output/clip_retrieval_$MODEL_SLUG}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-auto}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-42}"

"$PYTHON_BIN" my_code/train/finetune_clip_retrieval.py \
  --split-csv my_code/data/Image_IDs_n2_all_downloaded_split_8_1_1.csv \
  --caption-csv "my_code/data/Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv" \
  --image-root "$IMAGE_ROOT" \
  --image-layout "$IMAGE_LAYOUT" \
  --output-dir "$OUTPUT_PATH" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LEARNING_RATE" \
  --seed "$SEED" \
  "$@"
