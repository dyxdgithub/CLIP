#!/usr/bin/env python3
"""Enforce TinyCLIP package dependency boundaries.

The dependency direction is:
    L0 open_clip
    L1 data
    L2 training
    L3 my_code
    L4 top-level scripts

Higher layers may import lower layers. A package may import itself. Peer
packages and lower-to-higher imports are forbidden.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LAYER_NAMES = {
    0: "open_clip",
    1: "data",
    2: "training",
    3: "my_code",
    4: "top-level",
    5: "tooling",
}

PACKAGE_ROOTS = {
    "open_clip": 0,
    "data": 1,
    "training": 2,
    "my_code": 3,
}

TOP_LEVEL_ENTRYPOINTS = {
    "inference.py",
    "measure_throughput.py",
    "setup.py",
}

SKIP_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


def relative_path(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def layer_for_file(path: Path) -> tuple[int | None, str | None]:
    rel = relative_path(path)
    text = rel.as_posix()

    if text in TOP_LEVEL_ENTRYPOINTS:
        return 4, None

    parts = rel.parts
    if parts[0] == "src" and len(parts) >= 2:
        root = parts[1]
        if root in PACKAGE_ROOTS:
            return PACKAGE_ROOTS[root], root

    if parts[0] == "my_code":
        return PACKAGE_ROOTS["my_code"], "my_code"

    if parts[0] == "scripts":
        return 5, None

    return None, None


def module_root(import_name: str) -> str | None:
    first = import_name.split(".", 1)[0]
    return first if first in PACKAGE_ROOTS else None


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def collect_imports(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"{relative_path(path)}:{exc.lineno}: syntax error: {exc.msg}")
        return []

    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.append((node.module, node.lineno))
    return imports


def main() -> int:
    violations: list[str] = []
    covered = 0
    unknown: list[str] = []

    for path in iter_python_files():
        importer_layer, importer_root = layer_for_file(path)
        if importer_layer is None:
            unknown.append(relative_path(path).as_posix())
            continue
        covered += 1

        for import_name, line in collect_imports(path):
            target_root = module_root(import_name)
            if target_root is None:
                continue
            target_layer = PACKAGE_ROOTS[target_root]

            same_package = importer_root == target_root
            if target_layer < importer_layer or same_package:
                continue

            target_display = LAYER_NAMES[target_layer]
            importer_display = LAYER_NAMES[importer_layer]
            violations.append(
                f"{relative_path(path)}:{line} imports {import_name} "
                f"(layer {importer_layer} {importer_display} -> "
                f"layer {target_layer} {target_display}).\n"
                f"  Layer {importer_layer} packages may only import layers "
                f"below them, or their own package.\n"
                f"  Fix options:\n"
                f"  1. Move the dependency on {target_display} to a higher layer.\n"
                f"  2. Pass the needed value as a parameter instead of importing it.\n"
                f"  3. Define an interface in {importer_display} and implement it "
                f"in {target_display}.\n"
            )

    if unknown:
        print("Unmapped Python files (not covered by dependency linter):")
        for item in unknown:
            print(f"  - {item}")
        print("Add these paths to scripts/lint_deps.py to close the coverage gap.\n")

    if violations:
        print(f"Found {len(violations)} dependency violation(s):\n")
        print("\n".join(violations))
        return 1

    print(f"Dependency checks passed: {covered} Python files covered.")
    if unknown:
        print(f"WARNING: {len(unknown)} Python file(s) are outside the layer map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
