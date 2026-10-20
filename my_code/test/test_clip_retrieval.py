"""Evaluate a finetune_clip_retrieval.py checkpoint without retraining.

Reads config.json beside the checkpoint by default. Explicit command-line
arguments override saved settings. Only the test split is evaluated by default.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "my_code" / "data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Training best.pt or last.pt.")
    parser.add_argument("--config", type=Path, help="Training config JSON; default: checkpoint directory/config.json.")
    parser.add_argument("--split", choices=("test", "validation", "both"), default="test",
                        help="Split to evaluate. Default: %(default)s.")
    parser.add_argument("--split-csv", type=Path, help="Override training split CSV location; contents must match.")
    parser.add_argument("--caption-csv", type=Path, help="Override training caption CSV location; contents must match.")
    parser.add_argument("--image-root", type=Path, help="Override image root saved in training config.")
    parser.add_argument("--image-layout", choices=("relative", "flat"), help="Override saved image layout.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), help="Override saved precision.")
    parser.add_argument("--eval-batch-size", type=int, help="Override encoding batch size saved in config.")
    parser.add_argument("--score-batch-size", type=int, help="Override retrieval scoring block size.")
    parser.add_argument("--num-workers", type=int, help="Override image loader worker count.")
    parser.add_argument("--seed", type=int, help="Override training seed (also controls evaluation caption sampling).")
    parser.add_argument("--output-dir", type=Path,
                        help="Empty output directory; default: a unique directory under my_code/test/output.")
    for name in ("skip-file-check", "allow-truncated-images", "skip-bad-images"):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--" + name, dest=name.replace("-", "_"), action="store_true",
                           help="Enable {}; default: inherit training config.".format(name))
        group.add_argument("--no-" + name, dest=name.replace("-", "_"), action="store_false",
                           help="Disable {}.".format(name))
        parser.set_defaults(**{name.replace("-", "_"): None})
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    explicit_config = args.config is not None
    args.config = (args.config or args.checkpoint.parent / "config.json").expanduser().resolve()
    if explicit_config and not args.config.is_file():
        parser.error("Config does not exist: {}".format(args.config))
    saved = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    defaults = {
        "split_csv": DATA_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv",
        "caption_csv": DATA_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv",
        "image_root": None, "image_layout": "relative", "precision": "auto",
        "eval_batch_size": 16, "score_batch_size": 128, "num_workers": 4, "seed": 42,
        "skip_file_check": False, "allow_truncated_images": False, "skip_bad_images": False,
    }
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))
    if args.image_root is None:
        parser.error("Specify --image-root when the training config has no image_root")
    for key in ("split_csv", "caption_csv", "image_root"):
        setattr(args, key, Path(getattr(args, key)).expanduser().resolve())
    if args.eval_batch_size <= 0 or args.score_batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch sizes must be positive; num-workers must be nonnegative")
    if args.output_dir is None:
        run_name = "{}_{}_{}".format(args.checkpoint.parent.name, args.checkpoint.stem,
                                     datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        args.output_dir = Path(__file__).resolve().parent / "output" / ("clip_retrieval_" + run_name)
    args.output_dir = args.output_dir.expanduser().resolve()
    return args, saved


def main(argv=None):
    args, saved_config = parse_args(argv)
    # Keep --help usable without importing PyTorch or loading model weights.
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from my_code.train import finetune_clip_retrieval as training
    from clip.clip import _transform

    SummaryWriter = training.get_summary_writer()
    for name in ("split_csv", "caption_csv"):
        if not getattr(args, name).is_file():
            raise FileNotFoundError("Use --{} to locate {}".format(name.replace("_", "-"), getattr(args, name)))
    if not args.image_root.is_dir():
        raise NotADirectoryError("Use --image-root to locate images: {}".format(args.image_root))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty --output-dir: {}".format(args.output_dir))

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model", "identity"}.issubset(payload):
        raise ValueError("Expected a checkpoint saved by finetune_clip_retrieval.py")
    identity = payload["identity"]
    for name in ("split_csv", "caption_csv"):
        actual_hash = training.sha256_file(getattr(args, name))
        if identity.get(name + "_sha256") != actual_hash:
            raise ValueError("{} contents differ from the checkpoint's training data".format(name))
    checkpoint_info = {
        "epoch": int(payload.get("epoch", -1)) + 1,
        "global_step": int(payload.get("global_step", 0)),
        "best_score": payload.get("best_score"), "identity": identity,
    }
    state_dict = payload["model"]
    del payload  # Release saved optimizer state on CPU before building the model.
    device = training.choose_device(args.device)
    precision = training.choose_precision(args.precision, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # build_model casts weights to fp16; reload after float() to retain trained
    # fp32 weights. Architecture and resolution come from the saved weights.
    model = training.build_model(dict(state_dict)).float()
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.requires_grad_(False)
    model.eval().to(device)
    preprocess = _transform(model.visual.input_resolution)

    by_split, captions = training.read_data(args.split_csv, args.caption_csv)
    split_names = ("validation", "test") if args.split == "both" else (args.split,)
    selected = {name: by_split[name] for name in split_names}
    training.assign_image_paths(selected, args.image_root, args.image_layout,
                                not args.skip_file_check and not args.skip_bad_images)
    # Compact the text pool while preserving caption equality and positive pairs.
    indices = sorted({index for samples in selected.values() for sample in samples
                      for index in sample.caption_indices})
    remap = {index: position for position, index in enumerate(indices)}
    for samples in selected.values():
        for sample in samples:
            sample.caption_indices = tuple(remap[index] for index in sample.caption_indices)
    caption_tokens = training.tokenize_captions([captions[index] for index in indices])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "training_config": saved_config, "checkpoint_info": checkpoint_info,
        "checkpoint_sha256": training.sha256_file(args.checkpoint),
        "resolved_device": str(device), "resolved_precision": precision,
        "split_sizes": {key: len(value) for key, value in selected.items()},
        "caption_count": len(indices),
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Checkpoint: {}; epoch: {}; step: {}; device: {}".format(
        args.checkpoint, checkpoint_info["epoch"], checkpoint_info["global_step"], device), flush=True)
    results = {}
    with SummaryWriter(log_dir=str(args.output_dir / "tensorboard")) as writer:
        writer.add_text("evaluation/config", json.dumps(config, ensure_ascii=False, indent=2),
                        checkpoint_info["global_step"])
        for split, samples in selected.items():
            metrics = training.evaluate(
                model, samples, caption_tokens, preprocess, args, device, precision, split,
                args.output_dir / "bad_images.jsonl", set(),
            )
            training.append_log(
                args.output_dir / "metrics.jsonl",
                {"epoch": checkpoint_info["epoch"], "step": checkpoint_info["global_step"], **metrics},
                writer,
            )
            results.update(metrics)
            (args.output_dir / "metrics.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("Test outputs: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
