#!/usr/bin/env bash
# TinyCLIP harness startup smoke check.
# This is a library, not a long-running service, so "start" verifies that the
# version module is readable without importing PyTorch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c "exec(open('src/open_clip/version.py').read()); print('TinyCLIP', __version__, 'ready for library use')"
