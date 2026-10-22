"""Evaluate a Flickr30K CLIP fine-tuning checkpoint on Hugging Face data.

Loads the exact pinned dataset revision saved with the checkpoint. By default
only its test split is evaluated; validation can be selected for diagnostics.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="best.pt or last.pt from finetune_clip_flickr30k.py.")
    parser.add_argument("--config", type=Path,
                        help="Training config.json; default: beside the checkpoint.")
    parser.add_argument("--split", choices=("test", "validation", "both"), default="test",
                        help="Evaluation split. Default: %(default)s.")
    parser.add_argument("--hf-cache-dir", type=Path,
                        help="Hugging Face cache; default: training setting or library cache.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                        help="Inference autocast precision. Default: %(default)s.")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Image/text encoding batch size. Default: %(default)s.")
    parser.add_argument("--score-batch-size", type=int, default=128,
                        help="Retrieval score query block size. Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s.")
    parser.add_argument("--allow-truncated-images", action="store_true",
                        help="Attempt to recover incomplete image bytes.")
    parser.add_argument("--skip-bad-images", action="store_true",
                        help="Skip unreadable images; reported retrieval coverage then changes.")
    parser.add_argument("--output-dir", type=Path,
                        help="Empty output directory; default: unique directory under my_code/test/output.")
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    args.config = (args.config or args.checkpoint.parent / "config.json").expanduser().resolve()
    if not args.config.is_file():
        parser.error("Training config does not exist: {}".format(args.config))
    saved_config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.hf_cache_dir is None and saved_config.get("hf_cache_dir"):
        args.hf_cache_dir = Path(saved_config["hf_cache_dir"])
    if args.hf_cache_dir:
        args.hf_cache_dir = args.hf_cache_dir.expanduser().resolve()
    if args.eval_batch_size <= 0 or args.score_batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch sizes must be positive; num-workers must be nonnegative")
    if args.output_dir is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = Path(__file__).resolve().parent / "output" / (
            "clip_flickr30k_{}_{}_{}".format(args.checkpoint.parent.name, args.checkpoint.stem, suffix))
    args.output_dir = args.output_dir.expanduser().resolve()
    return args, saved_config


def main(argv=None):
    args, saved_config = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from clip.clip import _transform
    from my_code.train import finetune_clip_flickr30k as flickr

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty --output-dir: {}".format(args.output_dir))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model", "identity"}.issubset(payload):
        raise ValueError("Expected a checkpoint saved by finetune_clip_flickr30k.py")
    identity = payload["identity"]
    if not {"dataset", "dataset_revision", "split_sha256", "model"}.issubset(identity):
        raise ValueError("Checkpoint has no Flickr30K dataset identity")
    split_names = ("validation", "test") if args.split == "both" else (args.split,)
    hf_splits = flickr.load_hf_splits(identity["dataset"], identity["dataset_revision"],
                                      split_names, args.hf_cache_dir)
    by_split, captions = flickr.read_flickr30k(hf_splits, identity["dataset"])
    digests = flickr.split_digests(by_split, captions)
    for split in split_names:
        if digests[split] != identity["split_sha256"].get(split):
            raise ValueError("{} split changed since training; refusing to score mismatched data".format(split))
    device = flickr.choose_device(args.device)
    precision = flickr.choose_precision(args.precision, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    state_dict = payload["model"]
    checkpoint_info = {"epoch": int(payload.get("epoch", -1)) + 1,
                       "global_step": int(payload.get("global_step", 0)),
                       "best_score": payload.get("best_score"), "identity": identity}
    del payload
    model = flickr.build_model(dict(state_dict)).float()
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.requires_grad_(False)
    model.eval().to(device)
    preprocess = _transform(model.visual.input_resolution)
    caption_tokens = flickr.tokenize_captions(captions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update({"training_config": saved_config, "checkpoint_info": checkpoint_info,
                   "checkpoint_sha256": flickr.sha256_file(args.checkpoint),
                   "resolved_device": str(device), "resolved_precision": precision,
                   "split_sizes": {name: len(by_split[name]) for name in split_names}})
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Checkpoint: {}; epoch: {}; step: {}; device: {}".format(
        args.checkpoint, checkpoint_info["epoch"], checkpoint_info["global_step"], device), flush=True)
    results = {}
    SummaryWriter = flickr.get_summary_writer()
    with SummaryWriter(log_dir=str(args.output_dir / "tensorboard")) as writer:
        writer.add_text("evaluation/config", json.dumps(config, ensure_ascii=False, indent=2),
                        checkpoint_info["global_step"])
        for split in split_names:
            metrics = flickr.evaluate(
                model, by_split[split], caption_tokens, preprocess, args, device, precision,
                split, args.output_dir / "bad_images.jsonl", set(), hf_splits[split])
            flickr.append_log(args.output_dir / "metrics.jsonl",
                                 {"epoch": checkpoint_info["epoch"],
                                  "step": checkpoint_info["global_step"], **metrics}, writer)
            results.update(metrics)
            (args.output_dir / "metrics.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("Test outputs: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
