#!/usr/bin/env bash
# TinyCLIP harness teardown script.
# No long-running services are started, so this only removes generated
# harness artifacts and does not touch model checkpoints or source files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Tearing down TinyCLIP harness ==="

rm -f "$PROJECT_ROOT"/harness/trace/*.json 2>/dev/null || true
rm -f "$PROJECT_ROOT"/harness/state/*.json 2>/dev/null || true

echo "=== Harness environment cleaned up ==="
