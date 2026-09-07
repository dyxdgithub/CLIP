# Development Setup

## 1 Prerequisites

- Python 3.8+ (the package metadata states Python 3.6+, but PyTorch releases
  should guide the effective minimum).
- A CUDA-compatible PyTorch build for training, distillation, or throughput
  measurement. CPU works for simple inference.
- No external services are required for the core model library.

## 2 Quick Start

```bash
python -m pip install -U pip
python -m pip install -r requirements-training.txt
python -m pip install -e .
```

For inference-only use, `requirements.txt` is sufficient.

## 3 Build Commands

| Command | Description | Notes |
|---------|-------------|-------|
| `make build` | Syntax-compile Python files. | Does not import PyTorch, so it is fast. |
| `make lint-arch` | Run dependency and quality linters. | Dependency violations fail the build. |
| `make lint-deps` | Run only the dependency linter. | |
| `make lint-quality` | Run only the quality linter. | Current findings are warnings. |
| `make test` | Run `pytest` from `tests/`. | The directory is not present in this snapshot. |

## 4 Test Commands

The Makefile currently points at a `tests/` directory that is not present in
this repository snapshot. Before running `make test`, add pytest tests under
`tests/` or update the target.

```bash
python -m pytest -x -s -v tests
```

Useful smoke checks that do not require a trained checkpoint:

```bash
make build
make lint-arch
PYTHONPATH=src python -c "from open_clip.factory import list_models; print(len(list_models()), 'model configs')"
```

## 5 Inference And Evaluation

Inference examples:

```bash
python inference.py
python measure_throughput.py --model-name TinyCLIP-ViT-8M-16-Text-3M --device cpu
```

Training and evaluation need `src/` on `PYTHONPATH`. The documented
zero-shot evaluation pattern is:

```bash
PYTHONPATH=src python -m torch.distributed.launch \
  --use_env --nproc_per_node 8 src/training/main_for_test.py \
  --imagenet-val ./ImageNet \
  --model TinyCLIP-ViT-8M-16-Text-3M \
  --eval \
  --resume ./checkpoints/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt
```

PowerShell equivalent:

```powershell
$env:PYTHONPATH = "src"
python -m torch.distributed.launch --use_env --nproc_per_node 8 src/training/main_for_test.py --imagenet-val ./ImageNet --model TinyCLIP-ViT-8M-16-Text-3M --eval --resume ./checkpoints/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt
```

## 6 Project Structure

```text
.
├── src/
│   ├── open_clip/       L0 model library
│   ├── data/            L1 dataset tooling
│   └── training/        L2 training pipeline
├── my_code/             L3 local experiments
├── docs/                Architecture and development docs
├── scripts/             Agent linters
├── harness/             Runtime contract and environment scripts
├── inference.py         L4 inference entry point
└── measure_throughput.py L4 throughput benchmark
```

## 7 Configuration

| Config File | Location | Purpose |
|-------------|----------|---------|
| Model architecture | `src/open_clip/model_configs/*.json` | Declarative vision/text architecture. |
| Pretrained registry | `src/open_clip/pretrained.py` | URL and Hugging Face metadata for checkpoints. |
| Runtime contract | `harness/config/environment.json` | Harness startup and environment metadata. |

## 8 Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PYTHONPATH` | - | For training/eval | Must include `src` so `training.*` imports resolve. |
| `EVAL_FREQ` | `1000` | No | Evaluation frequency in training steps. |
| `SAVE_FREQ` | `1000` | No | Checkpoint frequency in training steps. |
| `EVAL_EMB` | - | No | Cached zero-shot classifier path. |
| `HF_TOKEN` | - | Optional | Used by `my_code/test/dataset.py` for gated datasets. |

> Sources: [src/training/train.py:157-158](),
> [src/training/zero_shot.py:124](),
> [my_code/test/dataset.py:10]().
