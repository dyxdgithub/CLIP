"""Download the three captioned Open Images CSV subsets.

Uses the CVDF Open Images S3 mirror first, then each row's OriginalURL and
Thumbnail300KURL. Downloads are concurrent, verified, and resumable.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path

from PIL import Image
from tqdm import tqdm


OPENIMAGES_DIR = Path(__file__).resolve().parents[1] / "OpenImages"
CSV_NAME = "Image IDs_with_captions_and_labels_captioned_only_{}.csv"
FOLDERS = {"train": "train", "test": "test", "validation": "val"}
S3_URL = "https://open-images-dataset.s3.amazonaws.com/{}/{}.jpg"
SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FIELDS = ("ImageID", "Subset", "OriginalURL", "Thumbnail300KURL")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=OPENIMAGES_DIR,
                        help="Directory with the three captioned_only CSVs. Default: my_code/data/OpenImages.")
    parser.add_argument("--output-dir", type=Path, default=OPENIMAGES_DIR,
                        help="Output root containing train/, test/ and val/. Default: my_code/data/OpenImages.")
    parser.add_argument("--state-dir", type=Path,
                        help="Directory for the SQLite checkpoint and reports. Defaults to --output-dir; "
                             "use a fast local disk when images are on a slow external drive.")
    parser.add_argument("--split", choices=("all", *FOLDERS), default="all",
                        help="Process all three CSVs or one official Open Images split. Default: %(default)s.")
    parser.add_argument("--workers", type=int, default=64,
                        help="Maximum concurrent image downloads. Default: %(default)s.")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Seconds per network connection/read. Default: %(default)s.")
    parser.add_argument("--retries", type=int, default=3,
                        help="Maximum attempts for each source URL per run. Default: %(default)s.")
    parser.add_argument("--backoff", type=float, default=1.0,
                        help="Base exponential retry delay in seconds. Default: %(default)s.")
    parser.add_argument("--max-image-ids", type=int,
                        help="Consider only the first N unique IDs per CSV, useful for a small trial.")
    parser.add_argument("--check-existing", action="store_true",
                        help="Recheck the presence and saved size of every completed image. "
                             "Normal resumes trust completed SQLite entries.")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Decode every existing image again, even when its saved file size matches; "
                             "also implies --check-existing.")
    parser.add_argument("--write-status-csv", action="store_true",
                        help="Export the full per-image status CSV after each split. "
                             "Omit for faster resumes; download_summary.json is always updated.")
    parser.add_argument("--retry-permanent", action="store_true",
                        help="Retry source URLs previously returning HTTP 404 or 410.")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 1 or args.timeout <= 0 or args.backoff < 0:
        parser.error("workers/retries/timeout must be positive; backoff must be nonnegative")
    if args.max_image_ids is not None and args.max_image_ids < 1:
        parser.error("--max-image-ids must be positive")
    args.csv_dir = args.csv_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.state_dir = (args.state_dir.expanduser().resolve() if args.state_dir
                      else args.output_dir)
    return args


def open_database(state_dir, output_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / ".openimages_v7_download.sqlite3"
    old_path = output_dir / db_path.name
    if state_dir != output_dir and not db_path.exists() and old_path.is_file():
        temporary = db_path.with_name(db_path.name + ".migrating")
        temporary.unlink(missing_ok=True)
        print("Copying existing download checkpoint to {} (do not run another downloader simultaneously)".format(
            db_path), flush=True)
        try:
            with closing(sqlite3.connect(old_path, timeout=30)) as old_db:
                with closing(sqlite3.connect(temporary)) as new_db:
                    with tqdm(desc="Migrating checkpoint", unit="page") as progress:
                        def update_progress(_status, remaining, total):
                            progress.total = total
                            progress.update(total - remaining - progress.n)

                        old_db.backup(new_db, pages=1024, progress=update_progress)
            os.replace(temporary, db_path)
        finally:
            temporary.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "split TEXT NOT NULL, image_id TEXT NOT NULL, original_url TEXT NOT NULL, "
        "thumbnail_url TEXT NOT NULL, original_md5 TEXT NOT NULL, "
        "manifest_token TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
        "source TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '', "
        "error TEXT NOT NULL DEFAULT '', file_size INTEGER NOT NULL DEFAULT 0, "
        "permanent_urls TEXT NOT NULL DEFAULT '[]', "
        "PRIMARY KEY (split, image_id))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS manifests ("
        "split TEXT PRIMARY KEY, signature TEXT NOT NULL, path TEXT NOT NULL)"
    )
    db.commit()
    return db


def import_manifest(db, split, path):
    if not path.is_file():
        raise FileNotFoundError("Missing {} CSV: {}".format(split, path))
    stat = path.stat()
    signature = hashlib.sha256(
        "{}:{}:{}".format(path.resolve(), stat.st_size, stat.st_mtime_ns).encode()
    ).hexdigest()
    previous = db.execute(
        "SELECT signature FROM manifests WHERE split = ?", (split,)
    ).fetchone()
    if previous == (signature,):
        return db.execute(
            "SELECT COUNT(*) FROM images WHERE split = ?", (split,)
        ).fetchone()[0]

    batch = []
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle, tqdm(
        desc="Reading " + split + " CSV", unit="row"
    ) as progress:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("{} needs columns {}".format(path, ", ".join(FIELDS)))
        for row_number, row in enumerate(reader, start=2):
            image_id = (row["ImageID"] or "").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{16}", image_id):
                raise ValueError("Invalid ImageID at {} row {}: {!r}".format(path, row_number, image_id))
            if (row["Subset"] or "").strip().lower() != split:
                raise ValueError("Wrong Subset at {} row {}".format(path, row_number))
            batch.append((
                split, image_id, (row["OriginalURL"] or "").strip(),
                (row["Thumbnail300KURL"] or "").strip(),
                (row.get("OriginalMD5") or "").strip(), signature,
            ))
            rows += 1
            if len(batch) >= 5000:
                upsert_batch(db, batch)
                progress.update(len(batch))
                batch.clear()
    if batch:
        upsert_batch(db, batch)
        progress.update(len(batch))
    db.execute("DELETE FROM images WHERE split = ? AND manifest_token != ?", (split, signature))
    db.execute(
        "INSERT INTO manifests (split, signature, path) VALUES (?, ?, ?) "
        "ON CONFLICT(split) DO UPDATE SET signature=excluded.signature, path=excluded.path",
        (split, signature, str(path)),
    )
    db.commit()
    unique = db.execute("SELECT COUNT(*) FROM images WHERE split = ?", (split,)).fetchone()[0]
    print("{}: {} CSV rows, {} unique ImageIDs".format(split, rows, unique), flush=True)
    return unique


def upsert_batch(db, batch):
    db.executemany(
        "INSERT INTO images (split, image_id, original_url, thumbnail_url, original_md5, manifest_token) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(split, image_id) DO UPDATE SET "
        "original_url=CASE WHEN images.manifest_token != excluded.manifest_token OR excluded.original_url != '' "
        "THEN excluded.original_url ELSE images.original_url END, "
        "thumbnail_url=CASE WHEN images.manifest_token != excluded.manifest_token OR excluded.thumbnail_url != '' "
        "THEN excluded.thumbnail_url ELSE images.thumbnail_url END, "
        "original_md5=CASE WHEN images.manifest_token != excluded.manifest_token OR excluded.original_md5 != '' "
        "THEN excluded.original_md5 ELSE images.original_md5 END, "
        "manifest_token=excluded.manifest_token",
        batch,
    )
    db.commit()


def valid_image(path):
    with Image.open(path) as image:
        image.load()
        if image.width < 1 or image.height < 1:
            raise ValueError("Image has zero width or height")
        return image.format


def existing_image(folder, image_id, saved_path, saved_size, status, verify):
    candidates = []
    if saved_path:
        saved = folder.parent / saved_path
        if saved.parent == folder:
            candidates.append(saved)
    candidates.extend(folder / (image_id + suffix) for suffix in SUFFIXES)
    for path in dict.fromkeys(candidates):
        try:
            size = path.stat().st_size
            if size == 0:
                continue
            if verify or status != "ok" or path != (folder.parent / saved_path) or size != saved_size:
                valid_image(path)
            return path, size
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    return None, 0


def download_url(url, temporary, expected_md5, args):
    for attempt in range(args.retries):
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "OpenImages-V7-CSV-downloader/1.0"})
            digest = hashlib.md5() if expected_md5 else None
            with urllib.request.urlopen(request, timeout=args.timeout) as response, temporary.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
            if digest is not None and digest.digest() != base64.b64decode(expected_md5, validate=True):
                raise ValueError("OriginalMD5 mismatch")
            image_format = valid_image(temporary)
            return image_format, "", False
        except urllib.error.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            if exc.code in (404, 410):
                return None, "HTTP {}".format(exc.code), True
            message = "HTTP {}".format(exc.code)
            if exc.code in (400, 401, 403):
                return None, message, False
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            message = "{}: {}".format(type(exc).__name__, exc)
        if attempt + 1 < args.retries:
            time.sleep(args.backoff * (2 ** attempt) + random.uniform(0, args.backoff))
    return None, message[:300], False


def download_one(row, root, args):
    split, image_id, original_url, thumbnail_url, original_md5, status, saved_path, saved_size, blocked_json = row
    folder = root / FOLDERS[split]
    try:
        folder.mkdir(parents=True, exist_ok=True)
        existing, size = existing_image(folder, image_id, saved_path, saved_size, status, args.verify_existing)
        if existing is not None:
            source = "existing" if status != "ok" else None
            return image_id, "ok", source, str(Path(FOLDERS[split]) / existing.name), "", size, blocked_json

        destination = folder / (image_id + ".jpg")
        temporary = folder / (image_id + ".jpg.part")
        converted = folder / (image_id + ".jpg.converted.part")
        blocked = set(json.loads(blocked_json)) if not args.retry_permanent else set()
        candidates = [
            ("cvdf_s3", S3_URL.format(split, image_id), ""),
            ("original_url", original_url, original_md5),
            ("thumbnail_url", thumbnail_url, ""),
        ]
        errors = []
        tried_urls = set()
        for source, url, md5 in candidates:
            if not url or url in tried_urls:
                continue
            tried_urls.add(url)
            if url in blocked:
                errors.append("{}: previous HTTP 404/410".format(source))
                continue
            image_format, error, permanent = download_url(url, temporary, md5, args)
            if permanent:
                blocked.add(url)
            if image_format is None:
                errors.append("{}: {}".format(source, error))
                continue
            try:
                if image_format == "JPEG":
                    os.replace(temporary, destination)
                else:
                    with Image.open(temporary) as image:
                        image.convert("RGB").save(converted, format="JPEG", quality=95)
                    valid_image(converted)
                    os.replace(converted, destination)
                return image_id, "ok", source, str(Path(FOLDERS[split]) / destination.name), "", destination.stat().st_size, json.dumps(sorted(blocked))
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                errors.append("{} conversion: {}".format(source, exc))
            finally:
                temporary.unlink(missing_ok=True)
                converted.unlink(missing_ok=True)
        return image_id, "failed", "", "", " | ".join(errors)[:2000], 0, json.dumps(sorted(blocked))
    except Exception as exc:
        return image_id, "failed", "", "", "{}: {}".format(type(exc).__name__, exc)[:2000], 0, blocked_json


def process_split(db, split, root, args, count):
    scope_count = min(count, args.max_image_ids) if args.max_image_ids else count
    upper_id = None
    if scope_count < count:
        upper_id = db.execute(
            "SELECT image_id FROM images WHERE split=? ORDER BY image_id LIMIT 1 OFFSET ?",
            (split, scope_count - 1),
        ).fetchone()[0]
    check_all = args.check_existing or args.verify_existing
    conditions = "split=? AND image_id>?"
    if upper_id is not None:
        conditions += " AND image_id<=?"
    if not check_all:
        conditions += " AND status!='ok'"
    query_args = (split, "") + ((upper_id,) if upper_id is not None else ())
    total = scope_count if check_all else db.execute(
        "SELECT COUNT(*) FROM images WHERE " + conditions, query_args,
    ).fetchone()[0]
    print("{}: {} completed entries skipped; {} images to {}".format(
        split, scope_count - total, total,
        "recheck" if check_all else "download or recover"), flush=True)
    if total == 0:
        return
    submitted = 0
    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor, tqdm(
        total=total, desc="Downloading " + split, unit="image"
    ) as progress:
        futures = {}
        pending_updates = []
        last_commit = time.monotonic()

        def flush_updates():
            nonlocal last_commit
            if not pending_updates:
                return
            db.executemany(
                "UPDATE images SET status=?, source=COALESCE(?, source), path=?, error=?, "
                "file_size=?, permanent_urls=? WHERE split=? AND image_id=?",
                pending_updates,
            )
            db.commit()
            pending_updates.clear()
            last_commit = time.monotonic()

        def complete_one():
            nonlocal completed, failures
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                image_id = futures.pop(future)
                result = future.result()
                if result[0] != image_id:
                    raise RuntimeError("Download worker returned a different ImageID")
                _, status, source, path, error, size, blocked = result
                pending_updates.append((status, source, path, error, size, blocked, split, image_id))
                completed += 1
                failures += status == "failed"
                progress.update(1)
            if len(pending_updates) >= 128 or time.monotonic() - last_commit >= 5:
                flush_updates()

        last_id = ""
        while submitted < total:
            page_args = (split, last_id) + ((upper_id,) if upper_id is not None else ())
            rows = db.execute(
                "SELECT split, image_id, original_url, thumbnail_url, original_md5, "
                "status, path, file_size, permanent_urls FROM images WHERE " + conditions +
                " ORDER BY image_id LIMIT 2048", page_args,
            ).fetchall()
            if not rows:
                break
            last_id = rows[-1][1]
            for row in rows:
                future = executor.submit(download_one, row, root, args)
                futures[future] = row[1]
                submitted += 1
                if len(futures) >= args.workers * 4:
                    complete_one()
        while futures:
            complete_one()
        flush_updates()
    print("{}: checked {}, failed {}".format(split, completed, failures), flush=True)


def write_report(db, split, root, write_status_csv):
    status_path = root / ("captioned_only_download_status_{}.csv".format(split))
    if write_status_csv:
        temporary = status_path.with_name(status_path.name + ".part")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("ImageID", "Subset", "DownloadStatus", "DownloadSource", "DownloadedImagePath", "DownloadError"))
            for image_id, status, source, path, error in db.execute(
                "SELECT image_id, status, source, path, error FROM images WHERE split=? ORDER BY image_id",
                (split,),
            ):
                writer.writerow((image_id, split, status, source, path, error))
        os.replace(temporary, status_path)
    summary = dict(db.execute(
        "SELECT status, COUNT(*) FROM images WHERE split=? GROUP BY status", (split,)
    ))
    report_note = str(status_path) if write_status_csv else "per-image CSV skipped (--write-status-csv to export)"
    print("{}: {} | {}".format(split, summary, report_note), flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    splits = tuple(FOLDERS) if args.split == "all" else (args.split,)
    db = open_database(args.state_dir, args.output_dir)
    try:
        result = 0
        summaries = {}
        for split in splits:
            path = args.csv_dir / CSV_NAME.format(split)
            count = import_manifest(db, split, path)
            process_split(db, split, args.output_dir, args, count)
            summary = write_report(db, split, args.state_dir, args.write_status_csv)
            summaries[split] = summary
            summary_path = args.state_dir / "download_summary.json"
            temporary = summary_path.with_name(summary_path.name + ".part")
            temporary.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            os.replace(temporary, summary_path)
            if summary.get("failed", 0) or (summary.get("pending", 0) and args.max_image_ids is None):
                result = 1
        return result
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
