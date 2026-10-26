"""Fine-tune CLIP on Flickr30K with selectively contrastive hard triplets.

This is a cross-modal adaptation of Xuan et al.'s SCT objective: each image
anchors its paired caption and the hardest different-caption text in the batch;
the text-to-image direction is symmetric. The matched "hn" objective uses the
same hardest negatives with ordinary NCA triplet loss.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from my_code.train import finetune_clip_flickr30k as base


def directional_triplet_loss(positive, hardest_negative, objective, sct_lambda):
    """Apply paper Eq. (13) to the valid triplets in one retrieval direction."""
    valid = torch.isfinite(hardest_negative)
    valid_count = int(valid.sum().item())
    if not valid_count:
        zero = positive.sum() * 0
        return zero, 0, 0, 0.0, 0.0
    positive = positive[valid]
    negative = hardest_negative[valid]
    hard = negative > positive
    nca = F.softplus(negative - positive)
    if objective == "sct":
        terms = torch.where(hard, sct_lambda * negative, nca)
    elif objective == "hn":
        terms = nca
    else:
        raise ValueError("Unknown objective: {}".format(objective))
    return (terms.mean(), int(hard.sum().item()), valid_count,
            positive.detach().sum().item(), negative.detach().sum().item())


def hard_triplet_loss(similarities, caption_indices, objective="sct", sct_lambda=1.0):
    """Mine hardest valid negatives and average image/text anchor losses.

    Equal caption strings are excluded from the negative set, as in the
    original Flickr30K training script's multi-positive CLIP loss. Similarities
    are unit-normalized cosine values, without CLIP's learned logit scale.
    """
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("Expected a square image-to-text similarity matrix")
    if caption_indices.ndim != 1 or caption_indices.numel() != similarities.shape[0]:
        raise ValueError("Caption index count must match the batch")
    same_caption = caption_indices[:, None].eq(caption_indices[None, :])
    negatives = similarities.masked_fill(same_caption, -torch.inf)
    positive = similarities.diagonal()
    image_negative = negatives.max(dim=1).values
    text_negative = negatives.max(dim=0).values
    image_loss, image_hard, image_valid, image_pos_sum, image_neg_sum = directional_triplet_loss(
        positive, image_negative, objective, sct_lambda)
    text_loss, text_hard, text_valid, text_pos_sum, text_neg_sum = directional_triplet_loss(
        positive, text_negative, objective, sct_lambda)
    stats = {
        "hard_i2t": image_hard, "hard_t2i": text_hard,
        "valid_i2t": image_valid, "valid_t2i": text_valid,
        "positive_sum": image_pos_sum + text_pos_sum,
        "negative_sum": image_neg_sum + text_neg_sum,
    }
    return (image_loss + text_loss) / 2, image_loss, text_loss, stats


def train_epoch(model, dataset, args, device, precision, optimizer, scaler, epoch, global_step,
                total_steps, log_path, writer, bad_image_path, seen_bad_images):
    model.train()
    dataset.epoch = epoch
    drop_singleton = len(dataset) % args.batch_size == 1
    loader = base.make_loader(dataset, args.batch_size, args.num_workers, True, args.seed + epoch,
                              drop_last=drop_singleton, collate_fn=base.collate_image_text_batch)
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    hard_i2t = hard_t2i = valid_i2t = valid_t2i = 0
    positive_sum = negative_sum = 0.0
    skipped_bad = 0
    dropped_singleton = int(drop_singleton)
    dropped_no_negative = 0
    with tqdm(loader, desc="Train {} epoch {}/{}".format(args.objective, epoch + 1, args.epochs),
              unit="batch") as progress:
        for batch, errors in progress:
            skipped_bad += len(errors)
            base.record_bad_images(bad_image_path, errors, "train", epoch + 1, seen_bad_images)
            if batch is None:
                continue
            images, tokens, caption_indices, _ = batch
            if images.shape[0] < 2:
                dropped_singleton += images.shape[0]
                continue
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            caption_indices = caption_indices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with base.autocast_context(device, precision):
                image_embeddings = model.encode_image(images)
                text_embeddings = model.encode_text(tokens)
            image_embeddings = F.normalize(image_embeddings.float(), dim=-1)
            text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
            similarities = image_embeddings @ text_embeddings.T
            loss, image_loss, text_loss, stats = hard_triplet_loss(
                similarities, caption_indices, args.objective, args.sct_lambda)
            if not stats["valid_i2t"] and not stats["valid_t2i"]:
                dropped_no_negative += images.shape[0]
                continue
            lr = base.set_learning_rate(optimizer, args.lr, global_step, total_steps, args.warmup_steps)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            global_step += 1
            batch_size = images.shape[0]
            totals += torch.tensor([loss.item(), image_loss.item(), text_loss.item()]) * batch_size
            count += batch_size
            hard_i2t += stats["hard_i2t"]
            hard_t2i += stats["hard_t2i"]
            valid_i2t += stats["valid_i2t"]
            valid_t2i += stats["valid_t2i"]
            positive_sum += stats["positive_sum"]
            negative_sum += stats["negative_sum"]
            progress.set_postfix(loss="{:.4f}".format(loss.item()), lr="{:.2g}".format(lr),
                                 hard="{:.1%}".format((stats["hard_i2t"] + stats["hard_t2i"]) /
                                                    (stats["valid_i2t"] + stats["valid_t2i"])))
            if global_step % args.log_every == 0:
                base.append_log(log_path, {
                    "epoch": epoch + 1, "step": global_step,
                    "train_step/loss_total": loss.item(),
                    "train_step/loss_i2t": image_loss.item(),
                    "train_step/loss_t2i": text_loss.item(),
                    "train_step/hard_fraction_i2t": stats["hard_i2t"] / stats["valid_i2t"],
                    "train_step/hard_fraction_t2i": stats["hard_t2i"] / stats["valid_t2i"],
                    "train_step/lr": lr,
                }, writer)
    if not count:
        raise RuntimeError("No training batches contain valid image-text triplets")
    return global_step, {
        "train_epoch/loss_total": (totals[0] / count).item(),
        "train_epoch/loss_i2t": (totals[1] / count).item(),
        "train_epoch/loss_t2i": (totals[2] / count).item(),
        "train_epoch/hard_fraction_i2t": hard_i2t / valid_i2t,
        "train_epoch/hard_fraction_t2i": hard_t2i / valid_t2i,
        "train_epoch/positive_cosine": positive_sum / (valid_i2t + valid_t2i),
        "train_epoch/hard_negative_cosine": negative_sum / (valid_i2t + valid_t2i),
        "train_epoch/usable_images": count,
        "train_epoch/skipped_bad_images": skipped_bad,
        "train_epoch/dropped_singleton_images": dropped_singleton,
        "train_epoch/dropped_no_negative_images": dropped_no_negative,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=base.DATASET_ID,
                        help="Hugging Face Flickr30K dataset. Default: %(default)s.")
    parser.add_argument("--dataset-revision", default=base.DATASET_REVISION,
                        help="Pinned Hugging Face revision. Default: %(default)s.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Optional Hugging Face datasets cache.")
    parser.add_argument("--output-dir", type=Path,
                        help="Empty run directory; default: a named directory under my_code/train/output.")
    parser.add_argument("--model", default="ViT-B/32",
                        help="CLIP model name or pretrained state-dict path. Default: %(default)s.")
    parser.add_argument("--download-root", type=Path, help="Optional pretrained weight cache.")
    parser.add_argument("--resume", type=Path, help="Resume from last.pt in its run directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                        help="CUDA autocast precision; CPU uses fp32. Default: %(default)s.")
    parser.add_argument("--objective", choices=("sct", "hn"), default="sct",
                        help="SCT or matched hard-negative NCA control. Default: %(default)s.")
    parser.add_argument("--sct-lambda", type=float, default=1.0,
                        help="Hard-triplet negative penalty in SCT; paper used 1 or 0.1 across datasets. "
                             "Default: %(default)s.")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs. Default: %(default)s.")
    parser.add_argument("--batch-size", type=int, default=256, help="Contrastive batch size. Default: %(default)s.")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Validation encoding batch size. Default: %(default)s.")
    parser.add_argument("--score-batch-size", type=int, default=128,
                        help="Retrieval score query block size. Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. Default: %(default)s.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak AdamW learning rate. Default: %(default)s.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW decay. Default: %(default)s.")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Linear warmup optimizer steps. Default: %(default)s.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm; 0 disables. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s.")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Optimizer steps between logs. Default: %(default)s.")
    parser.add_argument("--allow-truncated-images", action="store_true",
                        help="Attempt partial decoding of truncated images.")
    parser.add_argument("--skip-bad-images", action="store_true",
                        help="Skip unreadable images and record them; changes evaluation coverage.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "score_batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if (args.batch_size < 2 or args.num_workers < 0 or args.warmup_steps < 0 or
            args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0 or args.sct_lambda <= 0):
        parser.error("Batch size must be at least 2; workers/warmup/decay/clip nonnegative; "
                     "lr and sct-lambda positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.hf_cache_dir = args.hf_cache_dir.expanduser().resolve() if args.hf_cache_dir else None
    args.resume = args.resume.expanduser().resolve() if args.resume else None
    if args.output_dir is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
        args.output_dir = (args.resume.parent if args.resume else
                           base.OUTPUT_ROOT / ("clip_flickr30k_{}_{}".format(args.objective, slug)))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume and args.output_dir != args.resume.parent:
        raise ValueError("--output-dir must match the --resume checkpoint directory")
    args.download_root = args.download_root.expanduser().resolve() if args.download_root else None
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(args.resume))
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty --output-dir or specify --resume: {}".format(args.output_dir))
    hf_splits = base.load_hf_splits(args.dataset_id, args.dataset_revision,
                                   ("train", "validation", "test"), args.hf_cache_dir)
    by_split, captions = base.read_flickr30k(hf_splits, args.dataset_id)
    print("Flickr30K images: {}; unique captions: {}".format(
        {name: len(samples) for name, samples in by_split.items()}, len(captions)), flush=True)
    device = base.choose_device(args.device)
    precision = base.choose_precision(args.precision, device)
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print("CUDA {}: {:.2f} / {:.2f} GiB free before model load".format(
            device, free_bytes / 1024 ** 3, total_bytes / 1024 ** 3), flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, preprocess = base.load_pretrained(args.model, device, args.download_root)
    model.float()
    model.logit_scale.requires_grad_(False)  # Eq. (13) uses raw cosine, not CLIP logits.
    caption_tokens = base.tokenize_captions(captions)
    identity = {
        "dataset": args.dataset_id, "dataset_revision": args.dataset_revision,
        "split_sha256": base.split_digests(by_split, captions), "model": args.model,
        "objective": args.objective, "sct_lambda": args.sct_lambda if args.objective == "sct" else None,
        "mining": "paired positive and hardest different-caption in-batch negative, both directions",
        "similarity": "unit-normalized cosine without learned logit_scale",
    }
    train_dataset = base.ImageTextDataset(
        by_split["train"], caption_tokens, base.train_transform(model.visual.input_resolution),
        args.seed, args.allow_truncated_images, args.skip_bad_images, hf_splits["train"])
    if len(train_dataset) < 2:
        raise ValueError("Triplet training needs at least two images")
    optimizer = base.make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    SummaryWriter = base.get_summary_writer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume and not config_path.is_file():
        raise FileNotFoundError("Resume requires config.json beside the checkpoint: {}".format(config_path))
    if not args.resume:
        config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        config.update(identity)
        config.update({
            "paper_objective": "Xuan et al. 2020 selectively contrastive triplet, Eq. 13",
            "cross_modal_adaptation": True,
            "logit_scale_policy": "frozen; SCT uses raw normalized cosine",
            "validation_loss": "standard CLIP multi-positive InfoNCE, shared comparison metric",
            "resolved_device": str(device), "resolved_precision": precision,
            "split_sizes": {name: len(samples) for name, samples in by_split.items()},
            "caption_sampling": "one of each image's original captions, rotating by epoch",
            "selection_metric": "validation/mean_recall",
            "caption_tokenization": "CLIP 77 tokens, truncate=True",
        })
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = args.output_dir / "metrics.jsonl"
    bad_image_path = args.output_dir / "bad_images.jsonl"
    seen_bad_images = set()
    if bad_image_path.is_file():
        with bad_image_path.open("r", encoding="utf-8") as handle:
            seen_bad_images = {json.loads(line)["image_id"] for line in handle if line.strip()}
    start_epoch, global_step, best_score = 0, 0, -math.inf
    if args.resume:
        payload = base.load_checkpoint(args.resume, model, optimizer, scaler, identity)
        start_epoch, global_step, best_score = payload["epoch"] + 1, payload["global_step"], payload["best_score"]
        print("Resumed epoch {} at step {}".format(start_epoch, global_step), flush=True)
    steps_per_epoch = (len(train_dataset) // args.batch_size if len(train_dataset) % args.batch_size == 1
                       else math.ceil(len(train_dataset) / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    with SummaryWriter(log_dir=str(args.output_dir / "tensorboard"),
                       purge_step=global_step + 1 if args.resume else None) as writer:
        for epoch in range(start_epoch, args.epochs):
            global_step, train_metrics = train_epoch(
                model, train_dataset, args, device, precision, optimizer, scaler, epoch,
                global_step, total_steps, log_path, writer, bad_image_path, seen_bad_images)
            print("Epoch {}/{}: training finished; validating {} images".format(
                epoch + 1, args.epochs, len(by_split["validation"])), flush=True)
            phase_start = time.perf_counter()
            val_metrics = base.evaluate(
                model, by_split["validation"], caption_tokens, preprocess, args, device, precision,
                "validation", bad_image_path, seen_bad_images, hf_splits["validation"])
            print("Epoch {}/{}: validation finished in {:.1f}s; writing metrics".format(
                epoch + 1, args.epochs, time.perf_counter() - phase_start), flush=True)
            metrics = {"epoch": epoch + 1, "step": global_step, **train_metrics, **val_metrics}
            base.append_log(log_path, metrics, writer)
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            score = val_metrics["validation/mean_recall"]
            is_best = score > best_score
            if is_best:
                best_score = score
            print("Epoch {}/{}: saving last.pt{}".format(
                epoch + 1, args.epochs, " and best.pt" if is_best else ""), flush=True)
            phase_start = time.perf_counter()
            base.save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler,
                                 epoch, global_step, best_score, identity)
            if is_best:
                base.promote_checkpoint(args.output_dir / "last.pt", args.output_dir / "best.pt")
            print("Epoch {}/{}: checkpoint finished in {:.1f}s".format(
                epoch + 1, args.epochs, time.perf_counter() - phase_start), flush=True)


if __name__ == "__main__":
    main()
