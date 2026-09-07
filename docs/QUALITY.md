# Quality Standards

## 1 Dependency Direction

The architecture boundary is mechanically checked by
[scripts/lint_deps.py](). The rules are:

- `open_clip` is layer 0 and must have no repository-local imports above it.
- `training` is layer 2 and may import `open_clip` plus its own submodules.
- `my_code` is layer 3 and may consume lower layers only.

An agent-actionable violation includes the importing file, imported package,
layer transition, and possible fixes.

> Enforced by: [scripts/lint_deps.py:31-112]().

## 2 Code Quality Checks

[scripts/lint_quality.py]() reports non-fatal findings for:

- Hardcoded credentials or API keys.
- Python files above the recommended 1000-line threshold.
- Debug `print()` calls.
- TODO/FIXME markers.

Pass `--strict` to turn these reports into a non-zero exit status after the
known items are cleaned up.

> Enforced by: [scripts/lint_quality.py:12-112]().

## 3 File Size Guidance

Keep modules focused. If a module approaches 1000 lines, split by
responsibility rather than by arbitrary chunks. Current large files are
tracked in [docs/exec-plans/tech-debt-tracker.md]().

## 4 Secrets

```python
# Bad - committed token.
os.environ["HF_TOKEN"] = "hf_..."

# Good - read from the environment.
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
```

Secrets belong in environment variables or a local untracked file, never in
source. The current `HF_TOKEN` assignment in `my_code/test/dataset.py:10` is
listed as technical debt and should be removed/rotated.

## 5 Testing

There is currently no `tests/` directory. Before a functional change to
`open_clip`, add focused tests for:

- Model config loading and unknown config handling.
- Checkpoint prefix normalization.
- Tokenizer truncation and padding.
- `weight_inherit` shape remapping for a small synthetic state dict.

## 6 Enforcement

```bash
make lint-arch     # dependency + quality
make build         # syntax check
make test          # pytest, once tests/ exists
```
