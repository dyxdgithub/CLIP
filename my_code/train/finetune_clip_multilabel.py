"""Full-parameter CLIP fine-tuning for 285-class image multi-label prediction.

The split CSV lists observed positive labels. By default, unlisted classes
are assumed negative (a weak-label proxy). Supply human-verified Open Images
labels to mask unobserved classes instead.
"""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from finetune_clip_retrieval import (
    DATA_DIR, OUTPUT_DIR, Sample, append_log, assign_image_paths,
    autocast_context, choose_device, choose_precision, load_checkpoint,
    load_pretrained, make_loader, make_optimizer, save_checkpoint,
    set_learning_rate, sha256_file, train_transform,
)
import clip


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DATA_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv",
                        help="CSV with ImageID, RelativePath, Split, LabelName, DisplayName. Default: my_code/data split CSV.")
    parser.add_argument("--verified-labels-csv", type=Path,
                        help="Optional Open Images image-level labels CSV with ImageID, Source, LabelName, Confidence. "
                             "Only human-verified 0/1 labels are used; unknown classes are masked.")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Server image root. Relative layout expects root/RelativePath; flat expects root/ImageID.jpg.")
    parser.add_argument("--image-layout", choices=("relative", "flat"), default="relative",
                        help="How to find images below --image-root. Default: %(default)s.")
    parser.add_argument("--skip-file-check", action="store_true",
                        help="Skip startup existence checks; missing files will fail during loading.")
    parser.add_argument("--model", default="ViT-B/32",
                        help="Name from clip.available_models() or a pretrained CLIP state-dict path. Default: %(default)s.")
    parser.add_argument("--download-root", type=Path, help="Pretrained CLIP weight cache.")
    parser.add_argument("--output-dir", type=Path, help="Logs/checkpoints; default: my_code/train/output/clip_multilabel_<model>.")
    parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint from this script.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate validation and test using --resume; do not train.")
    parser.add_argument("--test-after-train", action="store_true",
                        help="Explicitly evaluate the test split with the best checkpoint after training.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                        help="CUDA autocast precision; auto chooses bf16 if supported, else fp16. CPU uses fp32.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of full training epochs. Default: %(default)s.")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per optimizer step. Default: %(default)s.")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Validation encoding batch size. Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=4, help="Image loading workers. Default: %(default)s.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak AdamW learning rate. Default: %(default)s.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay. Default: %(default)s.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup optimizer steps. Default: %(default)s.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm; 0 disables. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s.")
    parser.add_argument("--log-every", type=int, default=100, help="Optimizer steps between JSONL log records. Default: %(default)s.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("Workers/warmup/weight decay/gradient clipping must be nonnegative, and lr must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    return args


def read_labels(split_csv):
    records = {}
    positives = {}
    class_names = {}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "RelativePath", "Split", "LabelName", "DisplayName"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("{} needs columns {}".format(split_csv, sorted(required)))
        for row in reader:
            image_id = row["ImageID"].strip()
            relative = row["RelativePath"].strip()
            split = row["Split"].strip().lower()
            label = row["LabelName"].strip()
            display = row["DisplayName"].strip()
            if not image_id or not relative or not label or not display or split not in ("train", "validation", "test"):
                raise ValueError("Invalid split CSV row for ImageID {!r}".format(image_id))
            old_display = class_names.setdefault(label, display)
            if old_display != display:
                raise ValueError("Label {} has multiple display names".format(label))
            sample = records.get(image_id)
            if sample is None:
                records[image_id] = Sample(image_id, split, (relative,))
                positives[image_id] = set()
            elif sample.split != split:
                raise ValueError("ImageID {} appears in multiple splits".format(image_id))
            elif relative not in sample.relative_paths:
                sample.relative_paths += (relative,)
            positives[image_id].add(label)
    if not records:
        raise ValueError("Split CSV has no images")
    labels = sorted(class_names)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = {
        image_id: set(label_to_index[label] for label in image_labels)
        for image_id, image_labels in positives.items()
    }
    by_split = {"train": [], "validation": [], "test": []}
    for sample in records.values():
        by_split[sample.split].append(sample)
    if any(not samples for samples in by_split.values()):
        raise ValueError("Train, validation, and test splits must all contain images")
    return by_split, targets, labels, [class_names[label] for label in labels], label_to_index


def read_verified_labels(path, targets, label_to_index):
    negatives = {image_id: set() for image_id in targets}
    verified_values = {}
    matched = 0
    corrected_positives = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "Source", "LabelName", "Confidence"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("{} needs columns {}".format(path, sorted(required)))
        for row in reader:
            image_id = row["ImageID"].strip()
            label_index = label_to_index.get(row["LabelName"].strip())
            if image_id not in targets or label_index is None:
                continue
            if "verification" not in row["Source"].lower():
                continue
            try:
                confidence = float(row["Confidence"])
            except ValueError:
                continue
            if confidence not in (0.0, 1.0):
                continue
            key = (image_id, label_index)
            if key in verified_values and verified_values[key] != confidence:
                raise ValueError("Conflicting human verifications for {} / {}".format(image_id, row["LabelName"]))
            verified_values[key] = confidence
            if confidence == 1.0:
                targets[image_id].add(label_index)
                matched += 1
            else:
                if label_index in targets[image_id]:
                    targets[image_id].remove(label_index)
                    corrected_positives += 1
                negatives[image_id].add(label_index)
                matched += 1
    if not matched:
        raise ValueError("No matching human-verified 0/1 labels were found")
    if corrected_positives:
        print("Verified negatives override {} positive CSV labels".format(corrected_positives), flush=True)
    return negatives


class ImageLabelDataset(Dataset):
    def __init__(self, samples, targets, negatives, class_count, transform):
        self.samples = samples
        self.targets = targets
        self.negatives = negatives
        self.class_count = class_count
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        target = torch.zeros(self.class_count, dtype=torch.float32)
        positive = list(self.targets[sample.image_id])
        target[positive] = 1
        if self.negatives is None:
            mask = torch.ones(self.class_count, dtype=torch.bool)
        else:
            mask = torch.zeros(self.class_count, dtype=torch.bool)
            mask[positive] = True
            mask[list(self.negatives[sample.image_id])] = True
        return pixels, target, mask


def class_logits(model, images, class_tokens):
    image_features = F.normalize(model.encode_image(images).float(), dim=-1)
    text_features = F.normalize(model.encode_text(class_tokens).float(), dim=-1)
    return model.logit_scale.exp().float() * image_features @ text_features.T


def masked_bce(logits, targets, mask):
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    positive = mask & targets.bool()
    negative = mask & ~targets.bool()
    positive_loss = (losses * positive).sum() / positive.sum().clamp_min(1)
    negative_loss = (losses * negative).sum() / negative.sum().clamp_min(1)
    groups = int(positive.any()) + int(negative.any())
    return (positive_loss + negative_loss) / max(1, groups), positive_loss, negative_loss


def average_precision(scores, truth):
    positive_count = int(truth.sum().item())
    if positive_count == 0 or positive_count == truth.numel():
        return None
    ordered = truth[scores.argsort(descending=True)]
    ranks = torch.arange(1, len(ordered) + 1, dtype=torch.float32)
    precision = ordered.cumsum(0) / ranks
    return float((precision * ordered).sum() / positive_count)


def mean_average_precision(scores, targets, masks):
    class_ap = []
    for column in range(scores.shape[1]):
        known = masks[:, column]
        ap = average_precision(scores[known, column], targets[known, column])
        if ap is not None:
            class_ap.append(ap)
    micro_ap = average_precision(scores[masks], targets[masks])
    return {
        "mAP_macro": sum(class_ap) / len(class_ap) if class_ap else None,
        "mAP_micro": micro_ap,
        "evaluated_classes": len(class_ap),
        "total_classes": scores.shape[1],
    }


def train_epoch(model, dataset, class_tokens, optimizer, scaler, args,
                device, precision, epoch, global_step, total_steps, log_path):
    model.train()
    loader = make_loader(dataset, args.batch_size, args.num_workers, True, args.seed + epoch)
    loss_sums = torch.zeros(3, dtype=torch.float64)
    image_count = 0
    with tqdm(loader, desc="Train epoch {}/{}".format(epoch + 1, args.epochs), unit="batch") as progress:
        for images, targets, mask in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lr = set_learning_rate(optimizer, args.lr, global_step, total_steps, args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = class_logits(model, images, class_tokens)
                loss, positive_loss, negative_loss = masked_bce(logits, targets, mask)
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
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100))
            global_step += 1
            batch_size = images.shape[0]
            loss_sums += torch.tensor([loss.item(), positive_loss.item(), negative_loss.item()]) * batch_size
            image_count += batch_size
            progress.set_postfix(loss="{:.4f}".format(loss.item()), lr="{:.2g}".format(lr))
            if global_step % args.log_every == 0:
                append_log(log_path, {
                    "epoch": epoch + 1, "step": global_step,
                    "train_step/loss_total": loss.item(),
                    "train_step/loss_positive": positive_loss.item(),
                    "train_step/loss_negative": negative_loss.item(),
                    "train_step/lr": lr,
                })
    return global_step, {
        "train_epoch/loss_total": (loss_sums[0] / image_count).item(),
        "train_epoch/loss_positive": (loss_sums[1] / image_count).item(),
        "train_epoch/loss_negative": (loss_sums[2] / image_count).item(),
    }


@torch.no_grad()
def evaluate(model, dataset, class_tokens, args, device, precision, split):
    model.eval()
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    with autocast_context(device, precision):
        text_features = F.normalize(model.encode_text(class_tokens).float(), dim=-1)
    scores, targets_all, masks_all = [], [], []
    loss_sum, image_count = 0.0, 0
    with tqdm(loader, desc="Evaluate {}".format(split), unit="batch") as progress:
        for images, targets, mask in progress:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, precision):
                image_features = F.normalize(model.encode_image(images).float(), dim=-1)
                logits = model.logit_scale.exp().float() * image_features @ text_features.T
            loss, _, _ = masked_bce(logits, targets.to(device), mask.to(device))
            loss_sum += loss.item() * images.shape[0]
            image_count += images.shape[0]
            scores.append(logits.float().cpu())
            targets_all.append(targets)
            masks_all.append(mask)
    result = mean_average_precision(torch.cat(scores), torch.cat(targets_all), torch.cat(masks_all))
    result = {split + "/" + key: value for key, value in result.items()}
    result[split + "/loss_total"] = loss_sum / image_count
    return result


def main(argv=None):
    args = parse_args(argv)
    args.split_csv = args.split_csv.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve()
    args.verified_labels_csv = args.verified_labels_csv.expanduser().resolve() if args.verified_labels_csv else None
    args.resume = args.resume.expanduser().resolve() if args.resume else None
    if args.output_dir is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
        args.output_dir = args.resume.parent if args.resume else OUTPUT_DIR / ("clip_multilabel_" + slug)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.download_root:
        args.download_root = args.download_root.expanduser().resolve()
    if not args.image_root.is_dir():
        raise NotADirectoryError("Image root does not exist: {}".format(args.image_root))
    by_split, targets, labels, class_names, label_to_index = read_labels(args.split_csv)
    negatives = read_verified_labels(args.verified_labels_csv, targets, label_to_index) if args.verified_labels_csv else None
    assign_image_paths(by_split, args.image_root, args.image_layout, not args.skip_file_check)
    policy = "verified_negative_mask" if negatives is not None else "unlisted_assumed_negative"
    print("Images: {}; classes: {}; label policy: {}".format(
        {split: len(samples) for split, samples in by_split.items()}, len(labels), policy), flush=True)
    if negatives is None:
        print("WARNING: Unlisted classes are assumed negative; mAP is a weak-label proxy.", flush=True)
    device = choose_device(args.device)
    precision = choose_precision(args.precision, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, preprocess = load_pretrained(args.model, device, args.download_root)
    class_tokens = clip.tokenize(["a photo of a {}.".format(name) for name in class_names], truncate=True).to(device)
    train_data = ImageLabelDataset(by_split["train"], targets, negatives, len(labels),
                                   train_transform(model.visual.input_resolution))
    val_data = ImageLabelDataset(by_split["validation"], targets, negatives, len(labels), preprocess)
    test_data = ImageLabelDataset(by_split["test"], targets, negatives, len(labels), preprocess)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    identity = {
        "task": "multilabel",
        "model": args.model,
        "split_csv_sha256": sha256_file(args.split_csv),
        "verified_labels_csv_sha256": sha256_file(args.verified_labels_csv) if args.verified_labels_csv else None,
        "label_policy": policy,
    }
    if not args.resume and any((args.output_dir / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl")):
        raise FileExistsError("Output already contains a run; choose a new --output-dir or use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(identity)
        config.update({
            "resolved_device": str(device), "resolved_precision": precision,
            "split_sizes": {key: len(value) for key, value in by_split.items()},
            "class_names": dict(zip(labels, class_names)),
            "class_prompt": "a photo of a {DisplayName}.",
        })
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path = args.output_dir / "metrics.jsonl"
    start_epoch, global_step, best_score = 0, 0, -math.inf
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scaler, identity)
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        best_score = payload["best_score"]
        print("Resumed epoch {} at step {}".format(start_epoch, global_step), flush=True)
    if args.eval_only:
        for split, dataset in (("validation", val_data), ("test", test_data)):
            print(json.dumps(evaluate(model, dataset, class_tokens, args, device, precision, split), indent=2), flush=True)
        return
    total_steps = args.epochs * math.ceil(len(train_data) / args.batch_size)
    for epoch in range(start_epoch, args.epochs):
        global_step, train_metrics = train_epoch(
            model, train_data, class_tokens, optimizer, scaler, args,
            device, precision, epoch, global_step, total_steps, log_path)
        val_metrics = evaluate(model, val_data, class_tokens, args, device, precision, "validation")
        metrics = {"epoch": epoch + 1, "step": global_step, **train_metrics, **val_metrics}
        append_log(log_path, metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        score = val_metrics["validation/mAP_macro"]
        if score is None:
            raise ValueError("No validation class has both known positive and negative examples")
        if score > best_score:
            best_score = score
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler,
                            epoch, global_step, best_score, identity)
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler,
                        epoch, global_step, best_score, identity)
    best_path = args.output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint found; increase --epochs or check --resume")
    if args.test_after_train:
        load_checkpoint(best_path, model, optimizer, scaler, identity)
        test_metrics = evaluate(model, test_data, class_tokens, args, device, precision, "test")
        append_log(log_path, {"epoch": "best", **test_metrics})
        print(json.dumps(test_metrics, indent=2), flush=True)
    print("Outputs: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
