#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# First argument is the checkpoint; remaining arguments override these defaults.
CHECKPOINT="${1:?Usage: bash my_code/test/run_test_clip_retrieval.sh /path/to/best.pt [options]}"
shift
PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_ROOT="${IMAGE_ROOT:-/media/xyycyc/Elements/dyx/TinyCLIP/my_code/data/OpenImage/meta/Hierarchy/n2/images_all}"
DEVICE="${DEVICE:-cuda:3}"
PRECISION="${PRECISION:-auto}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"$PYTHON_BIN" my_code/test/test_clip_retrieval.py \
  --checkpoint "$CHECKPOINT" \
  --image-root "$IMAGE_ROOT" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --split "$EVAL_SPLIT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "$@"
