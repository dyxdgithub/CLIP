"""Full-parameter CLIP fine-tuning for image-caption retrieval.

Run from the repository root. The split CSV supplies ImageID, RelativePath,
and Split; the caption CSV supplies ImageID and caption. Each image is loaded
once per epoch. Repeated caption text is treated as multiple positives.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "clip"
package_spec = importlib.util.spec_from_file_location(
    "clip", PACKAGE_DIR / "__init__.py", submodule_search_locations=[str(PACKAGE_DIR)])
clip = importlib.util.module_from_spec(package_spec)
sys.modules["clip"] = clip
package_spec.loader.exec_module(clip)

import torch
import torch.nn.functional as F
from clip.model import build_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
RECALL_K = (1, 5, 10)


@dataclass
class Sample:
    image_id: str
    split: str
    relative_paths: tuple
    caption_indices: tuple = ()
    image_path: Path = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DATA_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv",
                        help="CSV with ImageID, RelativePath, Split. Default: my_code/data split CSV.")
    parser.add_argument("--caption-csv", type=Path, default=DATA_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv",
                        help="CSV with ImageID and caption. Default: my_code/data caption CSV.")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Server image root. Relative layout expects root/RelativePath; flat expects root/ImageID.jpg.")
    parser.add_argument("--image-layout", choices=("relative", "flat"), default="relative",
                        help="How to find images below --image-root. Default: %(default)s.")
    parser.add_argument("--skip-file-check", action="store_true",
                        help="Skip startup existence checks; missing files will fail during loading.")
    parser.add_argument("--model", default="ViT-B/32",
                        help="Name from clip.available_models() or a pretrained CLIP state-dict path. Default: %(default)s.")
    parser.add_argument("--download-root", type=Path,
                        help="Pretrained CLIP weight cache; default: clip.load's standard cache.")
    parser.add_argument("--output-dir", type=Path,
                        help="Logs/checkpoints directory; default: my_code/train/output/clip_retrieval_<model>.")
    parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint from this script.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate validation and test using --resume; do not train.")
    parser.add_argument("--test-after-train", action="store_true",
                        help="Explicitly evaluate the test split with the best checkpoint after training.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N. Default: %(default)s.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                        help="CUDA autocast precision; auto chooses bf16 if supported, else fp16. CPU uses fp32.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of full training epochs. Default: %(default)s.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Image-caption pairs per contrastive batch; gradient accumulation is not used. Default: %(default)s.")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Validation encoding batch size. Default: %(default)s.")
    parser.add_argument("--score-batch-size", type=int, default=512,
                        help="Queries per retrieval similarity block. Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=4, help="Image loading workers. Default: %(default)s.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak AdamW learning rate. Default: %(default)s.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay. Default: %(default)s.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup optimizer steps. Default: %(default)s.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm; 0 disables. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s.")
    parser.add_argument("--log-every", type=int, default=100, help="Training optimizer steps between JSONL log records. Default: %(default)s.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "score_batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0 or args.warmup_steps < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("Workers/warmup/weight decay/gradient clipping must be nonnegative, and lr must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_data(split_csv, caption_csv):
    records = {}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ImageID", "RelativePath", "Split"}.issubset(reader.fieldnames or ()):
            raise ValueError("{} needs ImageID, RelativePath, Split columns".format(split_csv))
        for row in reader:
            image_id = row["ImageID"].strip()
            split = row["Split"].strip().lower()
            relative = row["RelativePath"].strip()
            if not image_id or not relative or split not in ("train", "validation", "test"):
                raise ValueError("Invalid split CSV row for ImageID {!r}".format(image_id))
            sample = records.get(image_id)
            if sample is None:
                records[image_id] = Sample(image_id, split, (relative,))
            else:
                if sample.split != split:
                    raise ValueError("ImageID {} appears in multiple splits".format(image_id))
                if relative not in sample.relative_paths:
                    sample.relative_paths += (relative,)
    if not records:
        raise ValueError("Split CSV has no images")

    caption_to_index = {}
    captions = []
    image_captions = {image_id: set() for image_id in records}
    with caption_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ImageID", "caption"}.issubset(reader.fieldnames or ()):
            raise ValueError("{} needs ImageID and caption columns".format(caption_csv))
        for row in reader:
            image_id = row["ImageID"].strip()
            if image_id not in records:
                continue
            caption = row["caption"].strip()
            if not caption:
                continue
            index = caption_to_index.get(caption)
            if index is None:
                index = len(captions)
                caption_to_index[caption] = index
                captions.append(caption)
            image_captions[image_id].add(index)
    missing = [image_id for image_id, indices in image_captions.items() if not indices]
    if missing:
        raise ValueError("{} images have no caption; first IDs: {}".format(len(missing), missing[:5]))
    by_split = {"train": [], "validation": [], "test": []}
    for image_id, sample in records.items():
        sample.caption_indices = tuple(sorted(image_captions[image_id]))
        by_split[sample.split].append(sample)
    if any(not samples for samples in by_split.values()):
        raise ValueError("Train, validation, and test splits must all contain images")
    return by_split, captions


def relative_image_path(root, relative):
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Unsafe RelativePath: {!r}".format(relative))
    return root.joinpath(*path.parts)


def assign_image_paths(by_split, root, layout, check_files):
    missing = []
    for samples in by_split.values():
        for sample in samples:
            if layout == "relative":
                candidates = [relative_image_path(root, relative) for relative in sample.relative_paths]
            else:
                candidates = [root / (sample.image_id + suffix) for suffix in IMAGE_SUFFIXES]
            if check_files:
                sample.image_path = next((path for path in candidates if path.is_file()), None)
                if sample.image_path is None:
                    missing.append((sample.image_id, str(candidates[0])))
            else:
                sample.image_path = candidates[0]
    if missing:
        raise FileNotFoundError(
            "{} images are missing under {}; first ID/path pairs: {}".format(len(missing), root, missing[:5])
        )


def tokenize_captions(captions):
    pieces = []
    with tqdm(total=len(captions), desc="Tokenizing captions", unit="caption") as progress:
        for start in range(0, len(captions), 512):
            batch = captions[start:start + 512]
            pieces.append(clip.tokenize(batch, truncate=True))
            progress.update(len(batch))
    return torch.cat(pieces, dim=0)


class ImageTextDataset(Dataset):
    def __init__(self, samples, caption_tokens, transform, seed):
        self.samples = samples
        self.caption_tokens = caption_tokens
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        caption_indices = sample.caption_indices
        caption_index = caption_indices[(self.seed + self.epoch + index) % len(caption_indices)]
        with Image.open(sample.image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, self.caption_tokens[caption_index], caption_index


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


def make_loader(dataset, batch_size, workers, shuffle, seed, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator, drop_last=drop_last)


def choose_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is unavailable")
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
        raise ValueError("Checkpoint model or CSV hashes differ from this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    return payload


def train_epoch(model, dataset, args, device, precision, optimizer, scaler, epoch, global_step, total_steps, log_path, writer):
    model.train()
    dataset.epoch = epoch
    drop_singleton = len(dataset) % args.batch_size == 1
    loader = make_loader(dataset, args.batch_size, args.num_workers, True, args.seed + epoch,
                         drop_last=drop_singleton)
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    with tqdm(loader, desc="Train epoch {}/{}".format(epoch + 1, args.epochs), unit="batch") as progress:
        for images, tokens, caption_indices in progress:
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
    return global_step, {
        "train_epoch/loss_total": (totals[0] / count).item(),
        "train_epoch/loss_i2t": (totals[1] / count).item(),
        "train_epoch/loss_t2i": (totals[2] / count).item(),
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
def evaluate(model, samples, caption_tokens, preprocess, args, device, precision, split):
    model.eval()
    dataset = ImageTextDataset(samples, caption_tokens, preprocess, args.seed)
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    image_features = []
    loss_total = 0.0
    image_count = 0
    with tqdm(loader, desc="Encode {} images".format(split), unit="batch") as progress:
        for images, tokens, caption_indices in progress:
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
    image_features = torch.cat(image_features, dim=0)

    text_indices = sorted({index for sample in samples for index in sample.caption_indices})
    text_position = {index: position for position, index in enumerate(text_indices)}
    text_features = []
    for start in tqdm(range(0, len(text_indices), args.eval_batch_size),
                      desc="Encode {} captions".format(split), unit="batch"):
        indices = text_indices[start:start + args.eval_batch_size]
        tokens = caption_tokens[indices].to(device, non_blocking=True)
        with autocast_context(device, precision):
            text_features.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    text_features = torch.cat(text_features, dim=0)

    image_positives = [set(text_position[index] for index in sample.caption_indices) for sample in samples]
    text_positives = [set() for _ in text_indices]
    for image_position, indices in enumerate(image_positives):
        for text_position_index in indices:
            text_positives[text_position_index].add(image_position)
    i2t = recall_at_k(image_features, text_features, image_positives, device, args.score_batch_size)
    t2i = recall_at_k(text_features, image_features, text_positives, device, args.score_batch_size)
    metrics = {split + "/loss_total": loss_total / image_count}
    metrics.update({split + "/i2t_" + key: value for key, value in i2t.items()})
    metrics.update({split + "/t2i_" + key: value for key, value in t2i.items()})
    metrics[split + "/mean_recall"] = sum(i2t.values()) / 6 + sum(t2i.values()) / 6
    return metrics


def main(argv=None):
    args = parse_args(argv)
    SummaryWriter = get_summary_writer()
    args.split_csv = args.split_csv.expanduser().resolve()
    args.caption_csv = args.caption_csv.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve()
    args.resume = args.resume.expanduser().resolve() if args.resume else None
    if args.output_dir is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
        args.output_dir = args.resume.parent if args.resume else OUTPUT_DIR / ("clip_retrieval_" + slug)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.download_root:
        args.download_root = args.download_root.expanduser().resolve()
    if not args.image_root.is_dir():
        raise NotADirectoryError("Image root does not exist: {}".format(args.image_root))
    by_split, captions = read_data(args.split_csv, args.caption_csv)
    assign_image_paths(by_split, args.image_root, args.image_layout, not args.skip_file_check)
    print("Images: {}; unique captions: {}".format(
        {split: len(samples) for split, samples in by_split.items()}, len(captions)), flush=True)
    device = choose_device(args.device)
    precision = choose_precision(args.precision, device)
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print("CUDA {}: {:.2f} GiB free / {:.2f} GiB total before model load".format(
            device, free_bytes / 1024 ** 3, total_bytes / 1024 ** 3), flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, preprocess = load_pretrained(args.model, device, args.download_root)
    caption_tokens = tokenize_captions(captions)
    identity = {
        "model": args.model,
        "split_csv_sha256": sha256_file(args.split_csv),
        "caption_csv_sha256": sha256_file(args.caption_csv),
    }
    train_dataset = ImageTextDataset(by_split["train"], caption_tokens,
                                     train_transform(model.visual.input_resolution), args.seed)
    if len(train_dataset) < 2:
        raise ValueError("Contrastive training needs at least two training images")
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    if not args.resume and any((args.output_dir / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl", "tensorboard")):
        raise FileExistsError("Output already contains a run; choose a new --output-dir or use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if not args.resume:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(identity)
        config["resolved_device"] = str(device)
        config["resolved_precision"] = precision
        config["split_sizes"] = {key: len(value) for key, value in by_split.items()}
        config["caption_tokenization"] = "CLIP 77 tokens, truncate=True"
        config["caption_sampling"] = "one per ImageID per epoch"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path = args.output_dir / "metrics.jsonl"
    start_epoch, global_step, best_score = 0, 0, -math.inf
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scaler, identity)
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        best_score = payload["best_score"]
        print("Resumed epoch {} at step {}".format(start_epoch, global_step), flush=True)
    tensorboard_dir = args.output_dir / ("tensorboard_eval" if args.eval_only else "tensorboard")
    with SummaryWriter(log_dir=str(tensorboard_dir),
                       purge_step=global_step + 1 if args.resume and not args.eval_only else None) as writer:
        if args.eval_only:
            for split in ("validation", "test"):
                metrics = evaluate(model, by_split[split], caption_tokens, preprocess, args, device, precision, split)
                append_log(log_path, {"epoch": "eval_only", "step": global_step, **metrics}, writer)
                print(json.dumps(metrics, indent=2), flush=True)
            return

        steps_per_epoch = (len(train_dataset) // args.batch_size if len(train_dataset) % args.batch_size == 1
                           else math.ceil(len(train_dataset) / args.batch_size))
        total_steps = args.epochs * steps_per_epoch
        for epoch in range(start_epoch, args.epochs):
            global_step, train_metrics = train_epoch(
                model, train_dataset, args, device, precision, optimizer, scaler,
                epoch, global_step, total_steps, log_path, writer)
            val_metrics = evaluate(model, by_split["validation"], caption_tokens, preprocess,
                                   args, device, precision, "validation")
            metrics = {"epoch": epoch + 1, "step": global_step, **train_metrics, **val_metrics}
            append_log(log_path, metrics, writer)
            print(json.dumps(metrics, indent=2), flush=True)
            score = val_metrics["validation/mean_recall"]
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
            test_metrics = evaluate(model, by_split["test"], caption_tokens, preprocess,
                                    args, device, precision, "test")
            append_log(log_path, {"epoch": "best", "step": global_step, **test_metrics}, writer)
            print(json.dumps(test_metrics, indent=2), flush=True)
    print("Outputs: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
