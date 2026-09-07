# Tech Debt Tracker

## 1 Known Quality Findings

| # | Location | Finding | Priority | Remediation |
|---|----------|---------|----------|-------------|
| 1 | `my_code/test/dataset.py:10` | Hardcoded Hugging Face token assigned to `HF_TOKEN`. | P0 | Remove the value, read `HF_TOKEN` from the environment, and rotate the leaked token. |
| 2 | `src/open_clip/model.py` | 1438-line module combines architecture, wrappers, and checkpoint conversion. | P2 | Split checkpoint conversion and DDP wrappers into separate modules. |
| 3 | `src/training/train.py` | 664-line training loop. | P2 | Extract logging, masking, and checkpoint helpers. |
| 4 | `src/training/data.py` | 503-line dataset module. | P3 | Extract synthetic and CSV builders if they grow further. |
| 5 | `Makefile` test target | Points at `tests/`, which does not exist. | P1 | Add focused pytest tests or update the target. |
| 6 | `src/open_clip/transform.py` | Duplicate `ResizeMaxSize` class definition. | P3 | Remove the duplicate and confirm intended resize behavior. |

## 2 Why These Are Not Linter Failures

The harness follows the "day-one pass" rule: existing issues are recorded
here and reported as warnings by `scripts/lint_quality.py`. Once these items
are fixed, the linter can be run with `--strict` in CI.
