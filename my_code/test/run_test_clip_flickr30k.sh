#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash my_code/test/run_test_clip_flickr30k.sh /path/to/best.pt [options]" >&2
  echo "   or: bash my_code/test/run_test_clip_flickr30k.sh pretrained ViT-B/32 [--config /path/to/config.json] [options]" >&2
  exit 2
fi
if [[ "$1" == "pretrained" ]]; then
  shift
  MODEL_NAME="${1:?Specify a CLIP model name after pretrained, e.g. ViT-B/32}"
  shift
  SOURCE_ARGS=(--pretrained "$MODEL_NAME")
else
  CHECKPOINT="$1"
  shift
  SOURCE_ARGS=(--checkpoint "$CHECKPOINT")
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:3}"
PRECISION="${PRECISION:-auto}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"$PYTHON_BIN" my_code/test/test_clip_flickr30k.py \
  "${SOURCE_ARGS[@]}" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --split "$EVAL_SPLIT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "$@"
