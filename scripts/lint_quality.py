#!/usr/bin/env python3
"""Report TinyCLIP quality findings.

By default this script returns zero so the current codebase passes on day one.
Pass --strict to fail when any finding is present.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

MAX_FILE_LINES = 1000

SECRET_NAMES = r"(?:HF_TOKEN|API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET|PASSWORD|TOKEN)"
SECRET_PATTERN = re.compile(
    rf"(?i)\b{SECRET_NAMES}\b"
    r"[^=\n]{0,60}=\s*['\"][A-Za-z0-9_\-./+]{16,}['\"]"
)

PRINT_PATTERN = re.compile(r"(?m)^\s*print\s*\(")
TODO_PATTERN = re.compile(r"(?im)^\s*#.*\b(TODO|FIXME)\b")


def relative_path(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] == "scripts":
            continue
        files.append(path)
    return sorted(files)


def collect_findings():
    secrets: list[tuple[str, int]] = []
    oversized: list[tuple[str, int]] = []
    print_counts: dict[str, int] = {}
    todo_counts: dict[str, int] = {}

    for path in iter_python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        lines = text.splitlines()
        line_count = len(lines)

        if line_count > MAX_FILE_LINES:
            oversized.append((relative_path(path).as_posix(), line_count))

        for idx, line in enumerate(lines, start=1):
            if SECRET_PATTERN.search(line):
                secrets.append((relative_path(path).as_posix(), idx))

        prints = len(PRINT_PATTERN.findall(text))
        if prints:
            print_counts[relative_path(path).as_posix()] = prints

        todos = len(TODO_PATTERN.findall(text))
        if todos:
            todo_counts[relative_path(path).as_posix()] = todos

    return secrets, oversized, print_counts, todo_counts


def main() -> int:
    strict = "--strict" in sys.argv
    secrets, oversized, print_counts, todo_counts = collect_findings()
    finding_count = (
        len(secrets)
        + len(oversized)
        + sum(print_counts.values())
        + sum(todo_counts.values())
    )

    if secrets:
        print(f"Secrets ({len(secrets)}):")
        for filename, line in secrets:
            print(
                f"  {filename}:{line}: hardcoded credential or token. "
                f"Move it to an environment variable and rotate any leaked value."
            )
        print()

    if oversized:
        print(f"Oversized files ({len(oversized)}):")
        for filename, line_count in oversized:
            print(
                f"  {filename}: {line_count} lines (recommended max {MAX_FILE_LINES}). "
                f"Split by responsibility."
            )
        print()

    if print_counts:
        print(f"Files with debug print() calls ({len(print_counts)}):")
        for filename, count in sorted(print_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {filename}: {count} print() call(s)")
        print()

    if todo_counts:
        print(f"Files with TODO/FIXME markers ({len(todo_counts)}):")
        for filename, count in sorted(todo_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {filename}: {count} marker(s)")
        print()

    if finding_count == 0:
        print("No quality findings.")
        return 0

    print(
        f"Quality findings: {finding_count}. "
        f"{'Failing in strict mode.' if strict else 'Run with --strict to fail CI after cleanup.'}"
    )
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
