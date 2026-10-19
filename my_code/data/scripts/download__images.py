"""Download Open Images in order: FiftyOne, TFDS, then ImageID URLs.

By default, the three OpenImages captioned manifests are processed separately.
Hierarchy layout places each row in its parent/leaf folder. Flat layout saves
one image per ImageID in a split folder. A SQLite checkpoint records progress;
each final CSV identifies the source or failure. Run again to retry incomplete
rows.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
OPENIMAGES_DIR = META_DIR / "OpenImages"
SPLIT_FOLDERS = {"train": "train", "test": "test", "validation": "val"}
OPENIMAGES_INPUTS = {
    split: OPENIMAGES_DIR / "Image IDs_with_captions_and_labels_captioned_only_{}.csv".format(split)
    for split in ("train", "test", "validation")
}
OPENIMAGES_STATUS = OPENIMAGES_DIR / "captioned_only_download_status.csv"
DEFAULT_S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
STATUS_FIELDS = ("DownloadStatus", "DownloadSource", "DownloadedImagePath", "DownloadError")
DONE = {"existing", "fiftyone", "tfds", "s3", "original_url", "thumbnail_url"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("hierarchy", "flat"), default="flat", help="flat downloads OpenImages by ImageID; hierarchy requires parent/leaf labels. Default: %(default)s")
    parser.add_argument("--input", type=Path, help="Single original manifest CSV. If omitted, flat processes the three OpenImages CSVs; required for hierarchy.")
    parser.add_argument("--class-tree", type=Path, help="Class-tree JSON; required for hierarchy and ignored for flat layout.")
    parser.add_argument("--output-dir", type=Path, help="Image directory. Flat mode uses OpenImages/train, test, and val for the built-in manifests; a custom flat --input requires this option. Hierarchy defaults to OpenImages/hierarchy.")
    parser.add_argument("--status-output", type=Path, help="Status CSV path. Built-in flat manifests add a split suffix; hierarchy defaults to OpenImages/hierarchy_download_status.csv.")
    parser.add_argument("--checkpoint-db", type=Path, help="SQLite checkpoint. Flat three-file mode adds a split suffix; default: hidden database beside each status CSV.")
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), help="FiftyOne, TFDS, and S3 split. auto reads --subset-column and is required for flat three-file mode; default: train for hierarchy, auto for flat.")
    parser.add_argument("--subset-column", default="Subset", help="Split column when --split auto is used; accepts train, validation, test. Default: %(default)s")
    parser.add_argument("--zoo-dataset", default="open-images-v7", help="FiftyOne Zoo dataset. Default: %(default)s")
    parser.add_argument("--fiftyone-dataset-name", default="tinyclip_n2_fallback", help="Prefix for per-batch FiftyOne dataset names. Default: %(default)s")
    parser.add_argument("--dataset-dir", type=Path, help="FiftyOne cache directory. Default: FiftyOne's configured directory.")
    parser.add_argument("--fiftyone-batch-size", type=int, default=500, help="Unique IDs requested per FiftyOne call. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=16, help="Existing-image checks, FiftyOne, and URL download workers. Default: %(default)s")
    parser.add_argument("--tfds-name", default="open_images_v4/200k", help="TFDS builder/configuration. Default: %(default)s")
    parser.add_argument("--tfds-data-dir", type=Path, help="TFDS prepared-data directory. Default: OpenImages/tfds_open_images_v4.")
    parser.add_argument("--download-tfds-data", action="store_true", help="Allow TFDS to prepare/download missing data; this may require substantial disk space. Default: false.")
    parser.add_argument("--tfds-image-id-key", default="auto", help="TFDS ID feature; auto tries image_id, image/filename, image_filename, id. Default: %(default)s")
    parser.add_argument("--source-url-template", action="append", help="ImageID URL template; repeatable, with {split} and {image_id}. Default: Open Images S3 URL.")
    parser.add_argument("--original-url-column", default="OriginalURL", help="Original image URL column. Default: %(default)s")
    parser.add_argument("--thumbnail-url-column", default="Thumbnail300KURL", help="Thumbnail URL column. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Image ID column. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName", help="Leaf label column. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Parent label column. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-tree label key. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Class-tree display-name key. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Class-tree children key. Default: %(default)s")
    parser.add_argument("--url-timeout", type=float, default=30, help="Seconds per URL request. Default: %(default)s")
    parser.add_argument("--url-retries", type=int, default=3, help="Attempts per URL. Default: %(default)s")
    parser.add_argument("--url-backoff", type=float, default=1, help="Base exponential retry delay in seconds. Default: %(default)s")
    parser.add_argument("--max-image-ids", type=int, help="Process only the first N pending unique IDs per manifest; remaining IDs stay pending. Default: all.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum tqdm refresh interval in seconds. Default: %(default)s")
    args = parser.parse_args(argv)
    if args.layout == "hierarchy" and (args.input is None or args.class_tree is None):
        parser.error("--input and --class-tree are required for hierarchy layout")
    input_split = next(
        (split for split, path in OPENIMAGES_INPUTS.items()
         if args.layout == "flat" and args.input is not None and args.input.expanduser().resolve() == path.resolve()),
        None,
    )
    if args.layout == "flat":
        if args.output_dir is None:
            if args.input is not None and input_split is None:
                parser.error("--output-dir is required with a custom flat --input")
            args.output_dir = OPENIMAGES_DIR / SPLIT_FOLDERS[input_split] if input_split else OPENIMAGES_DIR
        if args.status_output is None:
            args.status_output = (OPENIMAGES_STATUS.with_name("{}_{}{}".format(OPENIMAGES_STATUS.stem, input_split, OPENIMAGES_STATUS.suffix))
                                  if input_split else OPENIMAGES_STATUS)
    else:
        if args.output_dir is None:
            args.output_dir = OPENIMAGES_DIR / "hierarchy"
        if args.status_output is None:
            args.status_output = OPENIMAGES_DIR / "hierarchy_download_status.csv"
    if args.tfds_data_dir is None:
        args.tfds_data_dir = OPENIMAGES_DIR / "tfds_open_images_v4"
    if args.split is None:
        args.split = "auto" if args.layout == "flat" else "train"
    if args.layout == "flat" and args.input is None and args.split != "auto":
        parser.error("--split must be auto when processing all three OpenImages manifests")
    for name in ("fiftyone_batch_size", "workers", "url_retries"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.max_image_ids is not None and args.max_image_ids <= 0:
        parser.error("--max-image-ids must be positive")
    if args.url_timeout <= 0 or args.progress_refresh_seconds <= 0 or args.url_backoff < 0:
        parser.error("Timeout/refresh must be positive and backoff nonnegative")
    args.source_url_template = args.source_url_template or [DEFAULT_S3]
    for template in args.source_url_template:
        try:
            template.format(split="train", image_id="example")
        except (KeyError, ValueError) as error:
            parser.error("Invalid --source-url-template {!r}: {}".format(template, error))
    return args


def load_leaf_paths(path, args):
    with path.open("r", encoding="utf-8-sig") as handle:
        root = json.load(handle)
    paths = {}

    def visit(node, parent=None):
        label = node[args.label_key]
        name = "".join("_" if char in '<>:"/\\|?*' else char for char in str(node.get(args.text_key) or label)).strip().rstrip(".") or "unnamed"
        children = node.get(args.children_key, [])
        if children:
            for child in children:
                visit(child, (label, name))
        elif parent is not None:
            paths.setdefault((parent[0], label), (parent[1], name))

    for node in root[args.children_key]:
        visit(node)
    return paths


def image_suffix(path):
    with Image.open(path) as image:
        image.load()
        suffix = SUFFIXES.get(image.format)
    if suffix is None:
        raise ValueError("Unsupported image format in {}".format(path))
    return suffix


def existing_image(folder, image_id):
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = folder / (image_id + suffix)
        if candidate.is_file():
            try:
                image_suffix(candidate)
                return candidate
            except (OSError, ValueError):
                pass
    return None


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".inprogress")
    try:
        shutil.copyfile(source, temporary)
        image_suffix(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS rows (number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent TEXT NOT NULL, leaf TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, path TEXT NOT NULL, error TEXT NOT NULL)")
        self.db.commit()

    def seed(self, manifest_rows, refresh_seconds):
        prior_count = self.db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        if prior_count:
            if prior_count != len(manifest_rows):
                raise ValueError("Checkpoint does not match the input manifest; use a different --checkpoint-db/--status-output")
            saved_rows = self.db.execute("SELECT number, image_id, parent, leaf FROM rows ORDER BY number")
            with tqdm(total=prior_count, desc="Matching checkpoint", unit="row", mininterval=refresh_seconds) as progress:
                for saved, current in zip(saved_rows, manifest_rows):
                    if tuple(saved) != current:
                        raise ValueError("Checkpoint does not match the input manifest; use a different --checkpoint-db/--status-output")
                    progress.update(1)
        if not prior_count:
            with tqdm(total=len(manifest_rows), desc="Seeding checkpoint", unit="row", mininterval=refresh_seconds) as progress:
                for offset in range(0, len(manifest_rows), 5000):
                    batch = manifest_rows[offset:offset + 5000]
                    self.db.executemany("INSERT INTO rows VALUES (?, ?, ?, ?, 'pending', '', '', '')", batch)
                    progress.update(len(batch))
        # Failed rows are retried; check_existing_images validates completed paths once.
        self.db.execute("UPDATE rows SET status = 'pending', source = '', path = '', error = '' WHERE status = 'failed'")
        self.db.commit()

    def pending_rows(self, image_id):
        return self.db.execute("SELECT number, parent, leaf FROM rows WHERE image_id = ? AND status = 'pending'", (image_id,)).fetchall()

    def record(self, number, status, source="", path="", error=""):
        self.db.execute("UPDATE rows SET status = ?, source = ?, path = ?, error = ? WHERE number = ?", (status, source, path, error[:4000], number))
        self.db.commit()

    def add_error(self, image_id, message):
        self.db.execute("UPDATE rows SET error = substr(CASE WHEN error = '' THEN ? ELSE error || ' | ' || ? END, 1, 4000) WHERE image_id = ? AND status = 'pending'", (message, message, image_id))
        self.db.commit()

    def finish_failed(self, image_id):
        self.db.execute("UPDATE rows SET status = 'failed', error = CASE WHEN error = '' THEN 'No source returned a valid image' ELSE error END WHERE image_id = ? AND status = 'pending'", (image_id,))
        self.db.commit()

    def summary(self):
        return dict(self.db.execute("SELECT status, COUNT(*) FROM rows GROUP BY status"))


def read_manifest(path, args):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {args.image_id_column}
        if args.layout == "hierarchy":
            required.update((args.parent_label_column, args.label_column))
        if args.split == "auto":
            required.add(args.subset_column)
        if not required.issubset(fields):
            raise ValueError("Missing input columns: {}".format(", ".join(sorted(required - set(fields)))))
        if set(STATUS_FIELDS).intersection(fields):
            raise ValueError("Input already contains final status columns")
        rows, urls, splits = [], {}, {}
        with tqdm(desc="Reading manifest (total unknown)", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for number, row in enumerate(reader, start=2):
                image_id = (row[args.image_id_column] or "").strip()
                parent = (row.get(args.parent_label_column) or "").strip() if args.layout == "hierarchy" else ""
                leaf = (row.get(args.label_column) or "").strip() if args.layout == "hierarchy" else ""
                split = (row[args.subset_column] or "").strip().lower() if args.split == "auto" else args.split
                if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id) or (args.layout == "hierarchy" and (not parent or not leaf)):
                    raise ValueError("Invalid ImageID or labels at CSV row {}".format(number))
                if split not in ("train", "validation", "test"):
                    raise ValueError("Invalid split {!r} at CSV row {}".format(split, number))
                if image_id in splits and splits[image_id] != split:
                    raise ValueError("ImageID {} occurs in multiple splits".format(image_id))
                splits[image_id] = split
                rows.append((number, image_id, parent, leaf))
                choices = urls.setdefault(image_id, [])
                for kind, field in (("original_url", args.original_url_column), ("thumbnail_url", args.thumbnail_url_column)):
                    url = (row.get(field) or "").strip()
                    if url and url not in [item[1] for item in choices]:
                        choices.append((kind, url))
                progress.update(1)
    return fields, rows, urls, splits


def write_status(input_path, output_path, fields, store, refresh_seconds):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".inprogress")
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fields + list(STATUS_FIELDS))
        writer.writeheader()
        statuses = iter(store.db.execute("SELECT number, status, source, path, error FROM rows ORDER BY number"))
        total = store.db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        with tqdm(total=total, desc="Writing status CSV", unit="row", mininterval=refresh_seconds) as progress:
            for number, row in enumerate(reader, start=2):
                saved_number, status, saved_source, path, error = next(statuses)
                if number != saved_number:
                    raise RuntimeError("Checkpoint row order differs from manifest")
                row.update(zip(STATUS_FIELDS, (status, saved_source, path, error)))
                writer.writerow(row)
                progress.update(1)
    temporary.replace(output_path)


def destination_folder(paths, output_dir, parent, leaf):
    return output_dir if paths is None else output_dir.joinpath(*paths[(parent, leaf)])


def check_existing_images(store, manifest_rows, paths, output_dir, workers, refresh_seconds):
    """Check checkpoint paths and indexed output folders with bounded image decoders."""
    suffixes = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    folder_names = {}
    updates = []

    def names_in(folder):
        if folder not in folder_names:
            try:
                with os.scandir(folder) as entries:
                    folder_names[folder] = {
                        entry.name.lower() for entry in entries
                        if entry.name.lower().endswith(suffixes)
                    }
            except (FileNotFoundError, NotADirectoryError):
                folder_names[folder] = set()
        return folder_names[folder]

    def flush_updates():
        if updates:
            store.db.executemany(
                "UPDATE rows SET status = ?, source = ?, path = ?, error = ? WHERE number = ?",
                updates,
            )
            store.db.commit()
            updates.clear()

    def check_paths(candidates):
        for candidate in candidates:
            try:
                image_suffix(candidate)
                return candidate
            except (OSError, ValueError):
                pass
        return None

    with tqdm(total=len(manifest_rows), desc="Checking existing images", unit="row", mininterval=refresh_seconds) as progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            cursor = store.db.execute("SELECT number, image_id, parent, leaf, status, path FROM rows ORDER BY number")

            def finish_one():
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    number, status, saved_path = futures.pop(future)
                    found = future.result()
                    if found is None:
                        if status in DONE:
                            updates.append(("pending", "", "", "", number))
                    elif status not in DONE or str(found) != saved_path:
                        updates.append(("existing", str(found), str(found), "", number))
                    progress.update(1)
                if len(updates) >= 5000:
                    flush_updates()

            while True:
                batch = cursor.fetchmany(5000)
                if not batch:
                    break
                for number, image_id, parent, leaf, status, saved_path in batch:
                    if paths is not None and (parent, leaf) not in paths:
                        updates.append(("failed", "", "", "Parent/leaf label pair absent from class tree", number))
                        progress.update(1)
                        continue
                    folder = destination_folder(paths, output_dir, parent, leaf)
                    candidates = []
                    if status in DONE and saved_path and Path(saved_path).parent == folder:
                        candidates.append(Path(saved_path))
                    names = names_in(folder)
                    for suffix in suffixes:
                        candidate = folder / (image_id + suffix)
                        if candidate.name.lower() in names and candidate not in candidates:
                            candidates.append(candidate)
                    if candidates:
                        futures[pool.submit(check_paths, candidates)] = (number, status, saved_path)
                        if len(futures) >= workers * 4:
                            finish_one()
                    else:
                        if status in DONE:
                            updates.append(("pending", "", "", "", number))
                        progress.update(1)
                if len(updates) >= 5000:
                    flush_updates()
            while futures:
                finish_one()
            flush_updates()


def apply_source(image_id, source_path, source_kind, source_label, store, paths, output_dir):
    try:
        suffix = image_suffix(source_path)
    except (OSError, ValueError) as error:
        store.add_error(image_id, "{}: {}".format(source_kind, error))
        return
    for number, parent, leaf in store.pending_rows(image_id):
        folder = destination_folder(paths, output_dir, parent, leaf)
        existing = existing_image(folder, image_id)
        if existing is not None:
            store.record(number, "existing", str(existing), str(existing))
            continue
        destination = folder / (image_id + suffix)
        try:
            atomic_copy(source_path, destination)
            store.record(number, source_kind, source_label, str(destination))
        except (OSError, ValueError) as error:
            store.add_error(image_id, "{} copy: {}".format(source_kind, error))


def fiftyone_stage(ids, store, paths, output_dir, args):
    if not ids:
        return
    try:
        import fiftyone.zoo as foz
    except Exception as error:
        print("FiftyOne unavailable; continuing with TFDS: {}".format(error), flush=True)
        for image_id in ids:
            store.add_error(image_id, "FiftyOne unavailable: {}".format(error))
        return
    with tqdm(total=len(ids), desc="FiftyOne", unit="ID", mininterval=args.progress_refresh_seconds) as progress:
        for offset in range(0, len(ids), args.fiftyone_batch_size):
            batch = [image_id for image_id in ids[offset:offset + args.fiftyone_batch_size] if store.pending_rows(image_id)]
            if not batch:
                progress.update(min(args.fiftyone_batch_size, len(ids) - offset))
                continue
            batch_hash = hashlib.sha1((args.split + "\n" + "\n".join(batch)).encode("utf-8")).hexdigest()[:12]
            kwargs = dict(split=args.split, label_types=[], image_ids=batch, num_workers=args.workers,
                          dataset_name="{}_{}".format(args.fiftyone_dataset_name, batch_hash),
                          drop_existing_dataset=True)
            if args.dataset_dir is not None:
                kwargs["dataset_dir"] = str(args.dataset_dir)
            try:
                dataset = foz.load_zoo_dataset(args.zoo_dataset, **kwargs)
                found = set()
                for sample in dataset:
                    filepath = Path(sample.filepath)
                    image_id = str(sample.get_field("open_images_id") or filepath.stem).strip() if "open_images_id" in sample.field_names else filepath.stem
                    if image_id in batch and filepath.is_file():
                        found.add(image_id)
                        apply_source(image_id, filepath, "fiftyone", str(filepath), store, paths, output_dir)
                for image_id in set(batch) - found:
                    store.add_error(image_id, "FiftyOne did not return this ImageID")
            except Exception as error:
                for image_id in batch:
                    store.add_error(image_id, "FiftyOne: {}".format(error))
            progress.update(min(args.fiftyone_batch_size, len(ids) - offset))


def tfds_id(example, key):
    if key == "auto":
        key = next((item for item in ("image_id", "image/filename", "image_filename", "id") if item in example), None)
    if key is None or key not in example:
        raise KeyError("TFDS ImageID feature unavailable; set --tfds-image-id-key")
    value = example[key]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return Path(str(value)).stem


def tfds_stage(ids, store, paths, output_dir, args):
    remaining = {image_id for image_id in ids if store.pending_rows(image_id)}
    if not remaining:
        return
    try:
        import tensorflow_datasets as tfds
        dataset, info = tfds.load(args.tfds_name, split=args.split, data_dir=str(args.tfds_data_dir), download=args.download_tfds_data, try_gcs=False, shuffle_files=False, with_info=True)
        total = int(info.splits[args.split].num_examples)
        with tqdm(total=total, desc="Scanning TFDS", unit="example", mininterval=args.progress_refresh_seconds) as progress:
            for example in tfds.as_numpy(dataset):
                image_id = tfds_id(example, args.tfds_image_id_key)
                if image_id in remaining:
                    temporary = output_dir / ".fallback_cache" / (image_id + ".tfds.jpg")
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        Image.fromarray(example["image"]).convert("RGB").save(temporary, format="JPEG", quality=95)
                        apply_source(image_id, temporary, "tfds", args.tfds_name + ":" + image_id, store, paths, output_dir)
                    except (OSError, ValueError, KeyError) as error:
                        store.add_error(image_id, "TFDS image: {}".format(error))
                    finally:
                        temporary.unlink(missing_ok=True)
                    remaining.remove(image_id)
                progress.update(1)
                if not remaining:
                    break
        for image_id in remaining:
            store.add_error(image_id, "ImageID absent from TFDS split")
    except Exception as error:
        print("TFDS unavailable or scan failed; continuing with ImageID URLs: {}".format(error), flush=True)
        for image_id in remaining:
            store.add_error(image_id, "TFDS unavailable or scan failed: {}".format(error))


def url_candidates(image_id, metadata_urls, args):
    quoted_id = urllib.parse.quote(image_id, safe="")
    quoted_split = urllib.parse.quote(args.split, safe="")
    candidates = [("s3", template.format(split=quoted_split, image_id=quoted_id)) for template in args.source_url_template]
    candidates.extend(metadata_urls)
    seen = set()
    for kind, url in candidates:
        if url not in seen:
            seen.add(url)
            yield kind, url


def fetch_url(image_id, metadata_urls, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / (image_id + ".download")
    errors = []
    for kind, url in url_candidates(image_id, metadata_urls, args):
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            errors.append("{}: unsupported URL scheme".format(kind))
            continue
        for attempt in range(args.url_retries):
            temporary.unlink(missing_ok=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "TinyCLIP-OpenImages-downloader/1.0"})
                with urllib.request.urlopen(request, timeout=args.url_timeout) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
                image_suffix(temporary)
                return image_id, temporary, kind, url, ""
            except (OSError, ValueError, urllib.error.URLError) as error:
                errors.append("{} attempt {}/{}: {}".format(kind, attempt + 1, args.url_retries, error))
                temporary.unlink(missing_ok=True)
                if isinstance(error, urllib.error.HTTPError) and error.code in (403, 404):
                    break
                if attempt + 1 < args.url_retries:
                    time.sleep(args.url_backoff * 2 ** attempt)
    return image_id, None, "", "", " | ".join(errors) or "No usable URL"


def url_stage(ids, metadata_urls, store, paths, output_dir, args):
    remaining = [image_id for image_id in ids if store.pending_rows(image_id)]
    if not remaining:
        return
    cache_dir = output_dir / ".fallback_cache"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(remaining), desc="ImageID URLs", unit="ID", mininterval=args.progress_refresh_seconds) as progress:
        iterator = iter(remaining)
        futures = {}

        def submit_next():
            image_id = next(iterator, None)
            if image_id is not None:
                future = pool.submit(fetch_url, image_id, metadata_urls.get(image_id, []), cache_dir, args)
                futures[future] = image_id

        for _ in range(min(len(remaining), args.workers * 2)):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                image_id = futures.pop(future)
                temporary = None
                try:
                    _, temporary, kind, url, error = future.result()
                    if temporary is None:
                        store.add_error(image_id, "URL: " + error)
                    else:
                        apply_source(image_id, temporary, kind, url, store, paths, output_dir)
                except Exception as error:
                    store.add_error(image_id, "URL worker: {}".format(error))
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    store.finish_failed(image_id)
                    progress.update(1)
                    submit_next()


def process_manifest(args, expected_split=None):
    input_path = args.input.expanduser().resolve()
    tree_path = args.class_tree.expanduser().resolve() if args.layout == "hierarchy" else None
    output_dir = args.output_dir.expanduser().resolve()
    status_path = args.status_output.expanduser().resolve()
    checkpoint = args.checkpoint_db.expanduser().resolve() if args.checkpoint_db else status_path.with_name("." + status_path.stem + "_checkpoint.sqlite3")
    if len({input_path, status_path, checkpoint}) != 3:
        raise ValueError("--input, --status-output, and --checkpoint-db must be different files")
    args.tfds_data_dir = args.tfds_data_dir.expanduser().resolve()
    if args.dataset_dir is not None:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
    fields, manifest_rows, metadata_urls, splits = read_manifest(input_path, args)
    if expected_split is not None and any(split != expected_split for split in splits.values()):
        raise ValueError("{} contains an ImageID outside the {} split".format(input_path, expected_split))
    paths = load_leaf_paths(tree_path, args) if tree_path is not None else None
    store = DownloadStore(checkpoint)
    try:
        store.seed(manifest_rows, args.progress_refresh_seconds)
        check_existing_images(store, manifest_rows, paths, output_dir, args.workers, args.progress_refresh_seconds)
        pending = list(dict.fromkeys(
            image_id for (image_id,) in store.db.execute(
                "SELECT image_id FROM rows WHERE status = 'pending' ORDER BY number"
            )
        ))
        if args.max_image_ids is not None:
            pending = pending[:args.max_image_ids]
        print("Processing {} pending unique ImageIDs ({} manifest rows)".format(len(pending), len(manifest_rows)), flush=True)
        grouped = {}
        for image_id in pending:
            grouped.setdefault(splits[image_id], []).append(image_id)
        for split, selected_ids in grouped.items():
            args.split = split
            print("Split {}: {} pending ImageIDs".format(split, len(selected_ids)), flush=True)
            fiftyone_stage(selected_ids, store, paths, output_dir, args)
            tfds_stage(selected_ids, store, paths, output_dir, args)
            url_stage(selected_ids, metadata_urls, store, paths, output_dir, args)
        write_status(input_path, status_path, fields, store, args.progress_refresh_seconds)
        summary = store.summary()
        print("Status counts: {}".format(summary), flush=True)
        print("Status CSV: {}".format(status_path), flush=True)
        print("Checkpoint: {}".format(checkpoint), flush=True)
        return 1 if summary.get("failed", 0) or (summary.get("pending", 0) and args.max_image_ids is None) else 0
    finally:
        store.db.close()


def split_path(path, split):
    return path.with_name("{}_{}{}".format(path.stem, split, path.suffix))


def main(argv=None):
    args = parse_args(argv)
    if args.input is not None:
        return process_manifest(args)
    result = 0
    for split, input_path in OPENIMAGES_INPUTS.items():
        task = argparse.Namespace(**vars(args))
        task.input = input_path
        task.output_dir = args.output_dir / SPLIT_FOLDERS[split]
        task.status_output = split_path(args.status_output, split)
        if args.checkpoint_db is not None:
            task.checkpoint_db = split_path(args.checkpoint_db, split)
        print("Processing OpenImages {}: {}".format(split, input_path), flush=True)
        result = max(result, process_manifest(task, expected_split=split))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
