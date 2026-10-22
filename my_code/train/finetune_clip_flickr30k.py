"""Fully fine-tune CLIP for Flickr30K image-text retrieval.

Downloads the parquet Flickr30K dataset from Hugging Face into its normal cache.
The dataset provides train/validation/test splits and five original captions per
image. Validation selects best.pt; test is never used to select hyperparameters.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import clip
import torch
import torch.nn.functional as F
from clip.model import build_model
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from tqdm import tqdm


OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DATASET_ID = "AnyModal/flickr30k"
DATASET_REVISION = "4e67dafc9f603fe99c42776c9c3e199d993c7872"


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RECALL_K = (1, 5, 10)
_WARNED_TRUNCATED_PATHS = set()


@dataclass
class Sample:
    image_id: str
    split: str
    relative_paths: tuple
    caption_indices: tuple = ()
    image_path: Path = None


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_image_path(root, relative):
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Unsafe RelativePath: {!r}".format(relative))
    return root.joinpath(*path.parts)


def tokenize_captions(captions):
    pieces = []
    with tqdm(total=len(captions), desc="Tokenizing captions", unit="caption") as progress:
        for start in range(0, len(captions), 512):
            batch = captions[start:start + 512]
            pieces.append(clip.tokenize(batch, truncate=True))
            progress.update(len(batch))
    return torch.cat(pieces, dim=0)


def load_rgb_image(path, allow_truncated=False, label=None):
    label = label if label is not None else path
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as exc:
        if not allow_truncated or "truncated" not in str(exc).lower():
            raise OSError("Cannot decode image {}: {}".format(label, exc)) from exc
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if hasattr(path, "seek"):
            path.seek(0)
        with Image.open(path) as image:
            pixels = image.convert("RGB")
    except OSError as exc:
        raise OSError("Cannot decode truncated image {}: {}".format(label, exc)) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    if label not in _WARNED_TRUNCATED_PATHS:
        warnings.warn("Recovered truncated image; pixels may be incomplete: {}".format(label), RuntimeWarning)
        _WARNED_TRUNCATED_PATHS.add(label)
    return pixels


class ImageTextDataset(Dataset):
    def __init__(self, samples, caption_tokens, transform, seed, allow_truncated=False,
                 skip_bad_images=False, image_source=None):
        self.samples = samples
        self.caption_tokens = caption_tokens
        self.transform = transform
        self.seed = seed
        self.allow_truncated = allow_truncated
        self.skip_bad_images = skip_bad_images
        self.image_source = image_source
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        caption_indices = sample.caption_indices
        caption_index = caption_indices[(self.seed + self.epoch + index) % len(caption_indices)]
        try:
            if self.image_source is None:
                source = sample.image_path
            else:
                encoded = self.image_source[index]["image"]
                if isinstance(encoded, Image.Image):
                    source = encoded
                elif isinstance(encoded, dict):
                    if encoded.get("bytes") is not None:
                        source = io.BytesIO(encoded["bytes"])
                    elif encoded.get("path"):
                        source = encoded["path"]
                    else:
                        raise ValueError("Hugging Face image has no bytes or path for {}".format(sample.image_id))
                else:
                    raise ValueError("Unsupported Hugging Face image value for {}".format(sample.image_id))
            image = source.convert("RGB") if isinstance(source, Image.Image) else load_rgb_image(
                source, self.allow_truncated, sample.image_path)
            pixels = self.transform(image)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            if not self.skip_bad_images:
                raise
            return None, {
                "index": index, "image_id": sample.image_id,
                "path": str(sample.image_path), "reason": str(exc),
            }
        return pixels, self.caption_tokens[caption_index], caption_index, index


def collate_image_text_batch(items):
    valid = [item for item in items if item[0] is not None]
    errors = [item[1] for item in items if item[0] is None]
    return (default_collate(valid) if valid else None), errors


def train_transform(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_pretrained(name, device, download_root):
    if Path(name).is_file():
        try:
            state_dict = torch.load(name, map_location="cpu", weights_only=True)
        except (RuntimeError, ValueError):
            state_dict = None  # A JIT archive is handled by clip.load below.
        if isinstance(state_dict, dict):
            model = build_model(state_dict).float()
            size = model.visual.input_resolution
            preprocess = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])
            return model.to(device), preprocess
    model, preprocess = clip.load(
        name, device="cpu", jit=False,
        download_root=str(download_root) if download_root else None)
    return model.to(device), preprocess


def make_loader(dataset, batch_size, workers, shuffle, seed, drop_last=False, collate_fn=None):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator, drop_last=drop_last,
                      collate_fn=collate_fn)


def choose_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was selected but is unavailable (PyTorch CUDA build: {}, "
                "CUDA_VISIBLE_DEVICES: {!r}). Check torch.cuda.is_available() "
                "and the visible GPU IDs in this environment.".format(
                    torch.version.cuda, os.environ.get("CUDA_VISIBLE_DEVICES")))
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("--device supports only CPU or CUDA")
    return device


def choose_precision(name, device):
    if device.type == "cpu":
        if name in ("fp16", "bf16"):
            raise ValueError("CPU training uses fp32; choose --precision fp32 or auto")
        return "fp32"
    if name == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return name


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def multi_positive_loss(logits, caption_indices):
    positives = caption_indices[:, None].eq(caption_indices[None, :])
    image_logits = logits.float()
    text_logits = image_logits.t()
    negative_infinity = torch.finfo(image_logits.dtype).min
    image_loss = torch.logsumexp(image_logits, dim=1) - torch.logsumexp(
        image_logits.masked_fill(~positives, negative_infinity), dim=1)
    text_loss = torch.logsumexp(text_logits, dim=1) - torch.logsumexp(
        text_logits.masked_fill(~positives.t(), negative_infinity), dim=1)
    return (image_loss.mean() + text_loss.mean()) / 2, image_loss.mean(), text_loss.mean()


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "positional_embedding" in name or "token_embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def set_learning_rate(optimizer, base_lr, step, total_steps, warmup_steps):
    if step < warmup_steps:
        factor = (step + 1) / max(1, warmup_steps)
    else:
        fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1 + math.cos(math.pi * min(1.0, fraction)))
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def get_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required for training logs. Install it with "
            "python -m pip install -r my_code/train/requirements.txt"
        ) from exc
    return SummaryWriter


def append_log(path, data, writer=None):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    if writer is not None:
        step = data["step"]
        for name, value in data.items():
            if "/" in name and isinstance(value, (int, float)) and math.isfinite(value):
                writer.add_scalar(name, value, step)
        writer.flush()


def record_bad_images(path, errors, split, epoch, seen):
    new_errors = [error for error in errors if error["image_id"] not in seen]
    if not new_errors:
        return
    with path.open("a", encoding="utf-8") as handle:
        for error in new_errors:
            seen.add(error["image_id"])
            record = {"split": split, "epoch": epoch, **error}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Skipping unreadable image {}: {}".format(error["path"], error["reason"]),
                  file=sys.stderr, flush=True)


def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, best_score, identity):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "identity": identity,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, scaler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("identity") != identity:
        raise ValueError("Checkpoint model or Flickr30K dataset identity differs from this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    return payload


def train_epoch(model, dataset, args, device, precision, optimizer, scaler, epoch, global_step, total_steps,
                log_path, writer, bad_image_path, seen_bad_images):
    model.train()
    dataset.epoch = epoch
    drop_singleton = len(dataset) % args.batch_size == 1
    loader = make_loader(dataset, args.batch_size, args.num_workers, True, args.seed + epoch,
                         drop_last=drop_singleton, collate_fn=collate_image_text_batch)
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    skipped_bad = 0
    dropped_singleton = int(drop_singleton)
    with tqdm(loader, desc="Train epoch {}/{}".format(epoch + 1, args.epochs), unit="batch") as progress:
        for batch, errors in progress:
            skipped_bad += len(errors)
            record_bad_images(bad_image_path, errors, "train", epoch + 1, seen_bad_images)
            if batch is None:
                continue
            images, tokens, caption_indices, _ = batch
            if images.shape[0] < 2:
                dropped_singleton += images.shape[0]
                continue
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            caption_indices = caption_indices.to(device, non_blocking=True)
            lr = set_learning_rate(optimizer, args.lr, global_step, total_steps, args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits, _ = model(images, tokens)
                loss, image_loss, text_loss = multi_positive_loss(logits, caption_indices)
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
            totals += torch.tensor([loss.item(), image_loss.item(), text_loss.item()]) * batch_size
            count += batch_size
            progress.set_postfix(loss="{:.4f}".format(loss.item()), lr="{:.2g}".format(lr))
            if global_step % args.log_every == 0:
                append_log(log_path, {
                    "epoch": epoch + 1, "step": global_step, "train_step/loss_total": loss.item(),
                    "train_step/loss_i2t": image_loss.item(), "train_step/loss_t2i": text_loss.item(),
                    "train_step/lr": lr,
                }, writer)
    if not count:
        raise RuntimeError("No training batches contain at least two readable images")
    return global_step, {
        "train_epoch/loss_total": (totals[0] / count).item(),
        "train_epoch/loss_i2t": (totals[1] / count).item(),
        "train_epoch/loss_t2i": (totals[2] / count).item(),
        "train_epoch/usable_images": count,
        "train_epoch/skipped_bad_images": skipped_bad,
        "train_epoch/dropped_singleton_images": dropped_singleton,
    }


def recall_at_k(query_features, candidate_features, positives, device, block_size):
    candidate_features = candidate_features.to(device)
    hits = {k: 0 for k in RECALL_K}
    top_count = min(max(RECALL_K), candidate_features.shape[0])
    for start in tqdm(range(0, len(query_features), block_size), desc="Retrieval scores", unit="block", leave=False):
        queries = query_features[start:start + block_size].to(device)
        indices = (queries @ candidate_features.T).topk(top_count, dim=1).indices.cpu().tolist()
        for row, ranked in enumerate(indices):
            correct = positives[start + row]
            for k in RECALL_K:
                if any(index in correct for index in ranked[:k]):
                    hits[k] += 1
    return {"R@{}".format(k): hits[k] / len(query_features) for k in RECALL_K}


@torch.no_grad()
def evaluate(model, samples, caption_tokens, preprocess, args, device, precision, split,
             bad_image_path, seen_bad_images, image_source=None):
    model.eval()
    dataset = ImageTextDataset(samples, caption_tokens, preprocess, args.seed,
                               args.allow_truncated_images, args.skip_bad_images, image_source)
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed,
                         collate_fn=collate_image_text_batch)
    image_features = []
    valid_samples = []
    loss_total = 0.0
    image_count = 0
    skipped_bad = 0
    with tqdm(loader, desc="Encode {} images".format(split), unit="batch") as progress:
        for batch, errors in progress:
            skipped_bad += len(errors)
            record_bad_images(bad_image_path, errors, split, "eval", seen_bad_images)
            if batch is None:
                continue
            images, tokens, caption_indices, sample_indices = batch
            valid_samples.extend(samples[index] for index in sample_indices.tolist())
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            caption_indices = caption_indices.to(device, non_blocking=True)
            with autocast_context(device, precision):
                image_embedding = F.normalize(model.encode_image(images).float(), dim=-1)
                text_embedding = F.normalize(model.encode_text(tokens).float(), dim=-1)
                logits = model.logit_scale.exp().float() * image_embedding @ text_embedding.T
                loss, _, _ = multi_positive_loss(logits, caption_indices)
            image_features.append(image_embedding.cpu())
            loss_total += loss.item() * images.shape[0]
            image_count += images.shape[0]
    if not image_count:
        raise RuntimeError("No readable images in {} split".format(split))
    image_features = torch.cat(image_features, dim=0)

    text_indices = sorted({index for sample in valid_samples for index in sample.caption_indices})
    text_position = {index: position for position, index in enumerate(text_indices)}
    text_features = []
    for start in tqdm(range(0, len(text_indices), args.eval_batch_size),
                      desc="Encode {} captions".format(split), unit="batch"):
        indices = text_indices[start:start + args.eval_batch_size]
        tokens = caption_tokens[indices].to(device, non_blocking=True)
        with autocast_context(device, precision):
            text_features.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    text_features = torch.cat(text_features, dim=0)

    image_positives = [set(text_position[index] for index in sample.caption_indices) for sample in valid_samples]
    text_positives = [set() for _ in text_indices]
    for image_position, indices in enumerate(image_positives):
        for text_position_index in indices:
            text_positives[text_position_index].add(image_position)
    i2t = recall_at_k(image_features, text_features, image_positives, device, args.score_batch_size)
    t2i = recall_at_k(text_features, image_features, text_positives, device, args.score_batch_size)
    metrics = {
        split + "/loss_total": loss_total / image_count,
        split + "/usable_images": image_count,
        split + "/skipped_bad_images": skipped_bad,
    }
    metrics.update({split + "/i2t_" + key: value for key, value in i2t.items()})
    metrics.update({split + "/t2i_" + key: value for key, value in t2i.items()})
    metrics[split + "/mean_recall"] = sum(i2t.values()) / 6 + sum(t2i.values()) / 6
    return metrics


def load_hf_splits(dataset_id, revision, split_names, cache_dir=None):
    try:
        from datasets import Image, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets: python -m pip install -r my_code/train/requirements.txt") from exc
    result = {}
    for name in split_names:
        print("Loading Hugging Face {} {} split ({})".format(dataset_id, name, revision), flush=True)
        dataset = load_dataset(dataset_id, revision=revision, split=name,
                               cache_dir=str(cache_dir) if cache_dir else None)
        if "image" not in dataset.column_names:
            raise ValueError("Hugging Face dataset split {} lacks image column".format(name))
        result[name] = dataset.cast_column("image", Image(decode=False))
    return result


def read_flickr30k(datasets_by_split, dataset_id, required_splits=None):
    """Read original captions while retaining Hugging Face rows for image loading."""
    by_split = {"train": [], "validation": [], "test": []}
    captions, caption_to_index = [], {}
    seen_filenames = set()
    for split, dataset in datasets_by_split.items():
        if split not in by_split:
            raise ValueError("Unknown dataset split: {}".format(split))
        for row_number, record in enumerate(dataset.remove_columns("image"), 1):
            filename = record.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError("{} row {} lacks filename".format(split, row_number))
            filename = filename.strip()
            relative_image_path(Path("."), filename)
            if filename in seen_filenames:
                raise ValueError("Duplicate Flickr30K filename across splits: {}".format(filename))
            seen_filenames.add(filename)
            row_split = {"val": "validation"}.get(record.get("split"), record.get("split"))
            if row_split != split:
                raise ValueError("{} has row split {!r} in {}".format(filename, record.get("split"), split))
            originals = record.get("original_alt_text")
            if not isinstance(originals, (list, tuple)) or not originals:
                raise ValueError("No original captions for {}".format(filename))
            indices = []
            for raw in originals:
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("Empty original caption for {}".format(filename))
                caption = raw.strip()
                index = caption_to_index.get(caption)
                if index is None:
                    index = len(captions)
                    caption_to_index[caption] = index
                    captions.append(caption)
                if index not in indices:
                    indices.append(index)
            sample = Sample(filename, split, (filename,), tuple(indices))
            sample.image_path = "hf://datasets/{}/{}".format(dataset_id, filename)
            by_split[split].append(sample)
    required_splits = required_splits or tuple(datasets_by_split)
    if any(not by_split[name] for name in required_splits):
        raise ValueError("Requested Flickr30K splits must be nonempty")
    return by_split, captions


def split_digests(by_split, captions):
    result = {}
    for split, samples in by_split.items():
        if not samples:
            continue
        digest = hashlib.sha256()
        for sample in samples:
            item = [sample.image_id, [captions[index] for index in sample.caption_indices]]
            digest.update(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        result[split] = digest.hexdigest()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID,
                        help="Hugging Face parquet dataset with original_alt_text. Default: %(default)s.")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION,
                        help="Pinned Hugging Face dataset revision. Default: %(default)s.")
    parser.add_argument("--hf-cache-dir", type=Path,
                        help="Optional Hugging Face datasets cache; default: library cache.")
    parser.add_argument("--output-dir", type=Path,
                        help="Run directory; default: my_code/train/output/clip_flickr30k_<model>.")
    parser.add_argument("--model", default="ViT-B/32",
                        help="CLIP model name or pretrained state-dict path; default: %(default)s.")
    parser.add_argument("--download-root", type=Path, help="Optional CLIP pretrained weight cache.")
    parser.add_argument("--resume", type=Path, help="Resume from last.pt; uses its run directory by default.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N; default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                        help="CUDA autocast precision; CPU uses fp32. Default: %(default)s.")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs, including resumed epochs. Default: %(default)s.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Contrastive batch size, without gradient accumulation. Default: %(default)s.")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Validation encoding batch size. Default: %(default)s.")
    parser.add_argument("--score-batch-size", type=int, default=128,
                        help="Retrieval similarity query block size. Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. Default: %(default)s.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak AdamW learning rate. Default: %(default)s.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay. Default: %(default)s.")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Linear warmup optimizer steps. Default: %(default)s.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm; 0 disables. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s.")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Optimizer steps between training logs. Default: %(default)s.")
    parser.add_argument("--allow-truncated-images", action="store_true",
                        help="Try partial Pillow decoding for truncated images.")
    parser.add_argument("--skip-bad-images", action="store_true",
                        help="Skip unreadable images and record them; changes benchmark coverage.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "score_batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.batch_size < 2 or args.num_workers < 0 or args.warmup_steps < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("Batch size must be at least 2; workers/warmup/decay/clip nonnegative; lr positive")
    return args


def main(argv=None):
    import torch

    args = parse_args(argv)
    args.hf_cache_dir = args.hf_cache_dir.expanduser().resolve() if args.hf_cache_dir else None
    args.resume = args.resume.expanduser().resolve() if args.resume else None
    if args.output_dir is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
        args.output_dir = args.resume.parent if args.resume else OUTPUT_ROOT / ("clip_flickr30k_" + slug)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume and args.output_dir != args.resume.parent:
        raise ValueError("--output-dir must match the --resume checkpoint directory")
    args.download_root = args.download_root.expanduser().resolve() if args.download_root else None
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(args.resume))
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty --output-dir or specify --resume: {}".format(args.output_dir))
    hf_splits = load_hf_splits(args.dataset_id, args.dataset_revision,
                              ("train", "validation", "test"), args.hf_cache_dir)
    by_split, captions = read_flickr30k(hf_splits, args.dataset_id)
    print("Flickr30K images: {}; unique captions: {}".format(
        {name: len(samples) for name, samples in by_split.items()}, len(captions)), flush=True)
    device = choose_device(args.device)
    precision = choose_precision(args.precision, device)
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print("CUDA {}: {:.2f} / {:.2f} GiB free before model load".format(
            device, free_bytes / 1024 ** 3, total_bytes / 1024 ** 3), flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, preprocess = load_pretrained(args.model, device, args.download_root)
    model.float()  # AdamW keeps full-precision trainable weights; autocast controls activations.
    caption_tokens = tokenize_captions(captions)
    identity = {"dataset": args.dataset_id, "dataset_revision": args.dataset_revision,
                "split_sha256": split_digests(by_split, captions), "model": args.model}
    train_dataset = ImageTextDataset(
        by_split["train"], caption_tokens, train_transform(model.visual.input_resolution),
        args.seed, args.allow_truncated_images, args.skip_bad_images, hf_splits["train"])
    if len(train_dataset) < 2:
        raise ValueError("Contrastive training needs at least two training images")
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    SummaryWriter = get_summary_writer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume and not config_path.is_file():
        raise FileNotFoundError("Resume requires config.json beside the checkpoint: {}".format(config_path))
    if not args.resume:
        config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        config.update(identity)
        config.update({"resolved_device": str(device), "resolved_precision": precision,
                       "split_sizes": {name: len(samples) for name, samples in by_split.items()},
                       "caption_sampling": "one of each image's original captions, rotating by epoch",
                       "selection_metric": "validation/mean_recall",
                       "caption_tokenization": "CLIP 77 tokens, truncate=True"})
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = args.output_dir / "metrics.jsonl"
    bad_image_path = args.output_dir / "bad_images.jsonl"
    seen_bad_images = set()
    if bad_image_path.is_file():
        with bad_image_path.open("r", encoding="utf-8") as handle:
            seen_bad_images = {json.loads(line)["image_id"] for line in handle if line.strip()}
    start_epoch, global_step, best_score = 0, 0, -math.inf
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scaler, identity)
        start_epoch, global_step, best_score = payload["epoch"] + 1, payload["global_step"], payload["best_score"]
        print("Resumed epoch {} at step {}".format(start_epoch, global_step), flush=True)
    steps_per_epoch = len(train_dataset) // args.batch_size if len(train_dataset) % args.batch_size == 1 else math.ceil(len(train_dataset) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    with SummaryWriter(log_dir=str(args.output_dir / "tensorboard"),
                       purge_step=global_step + 1 if args.resume else None) as writer:
        for epoch in range(start_epoch, args.epochs):
            global_step, train_metrics = train_epoch(
                model, train_dataset, args, device, precision, optimizer, scaler, epoch,
                global_step, total_steps, log_path, writer, bad_image_path, seen_bad_images)
            val_metrics = evaluate(
                model, by_split["validation"], caption_tokens, preprocess, args, device, precision,
                "validation", bad_image_path, seen_bad_images, hf_splits["validation"])
            metrics = {"epoch": epoch + 1, "step": global_step, **train_metrics, **val_metrics}
            append_log(log_path, metrics, writer)
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            score = val_metrics["validation/mean_recall"]
            if score > best_score:
                best_score = score
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler,
                                          epoch, global_step, best_score, identity)
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler,
                                      epoch, global_step, best_score, identity)
    print("Outputs: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
