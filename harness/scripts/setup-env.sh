#!/usr/bin/env bash
# TinyCLIP harness setup script.
# The core library has no external services, so this script validates the
# Python runtime and documents the required local installation commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Setting up TinyCLIP harness ==="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON_BIN' was not found on PATH." >&2
    exit 1
fi

"$PYTHON_BIN" --version
echo "Project root: $PROJECT_ROOT"
echo "No external services are required for the core model library."
echo "Install dependencies when needed:"
echo "  $PYTHON_BIN -m pip install -r requirements-training.txt"
echo "  $PYTHON_BIN -m pip install -e ."

echo "=== Harness environment ready ==="
