import argparse
import html
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET = "wikimedia/wit_base"
IMAGE_FIELD_HINTS = ("image", "photo", "picture", "thumbnail")
URL_FIELD_HINTS = ("url", "src")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a Hugging Face dataset without downloading the whole dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name or local path.")
    parser.add_argument("--config", default=None, help="Optional dataset config name.")
    parser.add_argument("--split", default="train", help="Split to preview, for example train/validation/test.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to preview.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional Hugging Face endpoint, for example c.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7897.",
    )
    parser.add_argument("--retries", type=int, default=5, help="Retry count for unstable remote reads.")
    parser.add_argument("--retry-wait", type=float, default=5, help="Seconds to wait between retries.")
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional path for an HTML preview file with tabular sample content.",
    )
    parser.add_argument(
        "--max-value-chars",
        type=int,
        default=300,
        help="Maximum characters printed for long text values.",
    )
    return parser.parse_args()


def truncate_text(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def summarize_value(value: Any, max_chars: int) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return truncate_text(value, max_chars)
    if isinstance(value, bytes):
        return f"<bytes: {len(value)} bytes>"
    if isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, dict):
        keys = ", ".join(map(str, list(value.keys())[:8]))
        suffix = "" if len(value) <= 8 else ", ..."
        return f"<dict: {keys}{suffix}>"
    if isinstance(value, (list, tuple)):
        preview = ", ".join(summarize_value(item, 60) for item in value[:3])
        suffix = "" if len(value) <= 3 else ", ..."
        return f"[{preview}{suffix}]"

    size = getattr(value, "size", None)
    mode = getattr(value, "mode", None)
    if size is not None and mode is not None:
        return f"<image: mode={mode}, size={size}>"

    return truncate_text(repr(value), max_chars)


def load_preview_dataset(args: argparse.Namespace):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    from datasets import load_dataset

    kwargs = {
        "path": args.dataset,
        "name": args.config,
        "split": args.split,
        "streaming": args.streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    return load_dataset(**{key: value for key, value in kwargs.items() if value is not None})


def iter_preview_samples(dataset: Any, num_samples: int) -> list[dict[str, Any]]:
    return list(islice(iter(dataset), num_samples))


def load_samples_with_retries(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    last_error = None
    attempts = max(args.retries, 1)
    for attempt in range(1, attempts + 1):
        try:
            dataset = load_preview_dataset(args)
            samples = iter_preview_samples(dataset, args.num_samples)
            return dataset, samples
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break

            print(
                f"Preview attempt {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {args.retry_wait:g}s...",
                file=sys.stderr,
            )
            time.sleep(args.retry_wait)

    raise last_error


def print_dataset_summary(dataset: Any, samples: list[dict[str, Any]]) -> None:
    print(f"Dataset type: {type(dataset).__name__}")

    features = getattr(dataset, "features", None)
    if features:
        print("Features:")
        for name, feature in features.items():
            print(f"  - {name}: {feature}")
    elif samples:
        print("Columns:")
        for name in samples[0].keys():
            print(f"  - {name}")
    else:
        print("No samples found in this split.")


def print_samples(samples: list[dict[str, Any]], max_value_chars: int) -> None:
    for index, sample in enumerate(samples):
        print(f"\nSample #{index}")
        for key, value in sample.items():
            print(f"  {key}: {summarize_value(value, max_value_chars)}")


def looks_like_image_url(field_name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False

    lower_name = field_name.lower()
    lower_value = value.lower()
    has_image_hint = any(hint in lower_name for hint in IMAGE_FIELD_HINTS)
    has_url_hint = any(hint in lower_name for hint in URL_FIELD_HINTS)
    has_image_ext = lower_value.split("?", 1)[0].endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))
    return value.startswith(("http://", "https://")) and (has_image_hint or has_image_ext or has_url_hint)


def sample_to_html_rows(sample: dict[str, Any], max_value_chars: int) -> str:
    rows = []
    for key, value in sample.items():
        escaped_key = html.escape(str(key))
        if looks_like_image_url(key, value):
            escaped_url = html.escape(value, quote=True)
            body = f'<a href="{escaped_url}">{escaped_url}</a><br><img src="{escaped_url}" loading="lazy">'
        else:
            body = html.escape(summarize_value(value, max_value_chars))
        rows.append(f"<tr><th>{escaped_key}</th><td>{body}</td></tr>")
    return "\n".join(rows)


def write_html_preview(path: Path, samples: Iterable[dict[str, Any]], max_value_chars: int) -> None:
    cards = []
    for index, sample in enumerate(samples):
        cards.append(
            f"""
            <section class="sample">
              <h2>Sample #{index}</h2>
              <table>{sample_to_html_rows(sample, max_value_chars)}</table>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Preview</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f6f8fb;
    }}
    h1 {{ margin-bottom: 16px; }}
    .sample {{
      margin-bottom: 18px;
      padding: 16px;
      background: white;
      border: 1px solid #d8dee9;
      border-radius: 8px;
    }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ padding: 10px; border-top: 1px solid #edf1f7; vertical-align: top; word-break: break-word; }}
    th {{ width: 220px; text-align: left; color: #52606d; }}
    img {{ max-width: 260px; max-height: 180px; margin-top: 8px; object-fit: contain; }}
  </style>
</head>
<body>
  <h1>Dataset Preview</h1>
  {''.join(cards)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    print(f"\nHTML preview written to: {path.resolve()}")


def main() -> None:
    args = parse_args()
    try:
        dataset, samples = load_samples_with_retries(args)
        from datasets import DatasetDict, IterableDatasetDict

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            available_splits = ", ".join(dataset.keys())
            raise ValueError(
                f"load_dataset returned multiple splits ({available_splits}). "
                f"Please choose one with --split."
            )

        print_dataset_summary(dataset, samples)
        print_samples(samples, args.max_value_chars)

        if args.html is not None:
            write_html_preview(args.html, samples, args.max_value_chars)
    except Exception as exc:
        print(f"Failed to preview dataset: {exc}", file=sys.stderr)
        print(
            "Tip: keep --streaming enabled for large remote datasets, "
            "try --hf-endpoint https://hf-mirror.com, "
            "or retry when Hugging Face/network access is stable.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
