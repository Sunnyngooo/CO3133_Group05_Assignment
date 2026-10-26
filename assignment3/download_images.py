"""Download N24News images with bounded retries, image validation and resume."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import faulthandler
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

from PIL import Image, ImageOps
import requests

from data_utils import ROOT, canonical_url, load_records, select_records, selection_digest, write_json

MAX_BYTES = 30 * 1024 * 1024
_thread_state = threading.local()


def session():
    if not hasattr(_thread_state, "session"):
        client = requests.Session()
        client.headers.update({"User-Agent": "CO3133-Group05-N24News-Research/1.0"})
        client.max_redirects = 3
        _thread_state.session = client
    return _thread_state.session


def inspect_image(data):
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        if min(image.size) < 32:
            raise ValueError("Image dimensions below 32 pixels")
        signature = f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()
        return {
            "width": image.width, "height": image.height,
            "pixel_sha256": hashlib.sha256(signature).hexdigest(),
            "file_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        }


def download_one(record, image_dir, timeout, retries, image_source="metadata"):
    image_id = record["image_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
        raise ValueError("Unsafe image ID")
    path = image_dir / (image_id + ".jpg")
    result = {"image_id": image_id, "section": record["section"], "path": str(path.relative_to(ROOT)), "image_source": image_source}
    if path.exists():
        try:
            return {**result, "status": "cached", **inspect_image(path.read_bytes()), "attempts": 0, "error": ""}
        except (OSError, ValueError, Image.DecompressionBombError):
            # Preserve an invalid existing file until a valid replacement is available.
            pass
    url = canonical_url(record["image_url"]) if image_source == "original" else record["image_url"]
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname != "static01.nyt.com":
        return {**result, "status": "failed", "attempts": 0, "error": "Unexpected image source"}
    error = ""
    for attempt in range(1, retries + 2):
        try:
            with session().get(url, timeout=(10, timeout), stream=True) as response:
                response.raise_for_status()
                if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > MAX_BYTES:
                    raise ValueError("Image exceeds size limit")
                chunks = []
                size = 0
                transfer_started = time.monotonic()
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Image exceeds size limit")
                    if time.monotonic() - transfer_started > 45:
                        raise TimeoutError("Image transfer exceeded 45 seconds")
                    chunks.append(chunk)
                data = b"".join(chunks)
            if len(data) > MAX_BYTES:
                raise ValueError("Image exceeds size limit")
            info = inspect_image(data)
            temporary = path.with_suffix(".jpg.part")
            temporary.write_bytes(data)
            temporary.replace(path)
            return {**result, "status": "downloaded", **info, "attempts": attempt, "error": ""}
        except (requests.RequestException, TimeoutError, OSError, ValueError, Image.DecompressionBombError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, requests.HTTPError) and exc.response.status_code in (400, 401, 403, 404, 410):
                break
            if attempt <= retries:
                delay = min(2 ** (attempt - 1), 4)
                if isinstance(exc, requests.HTTPError) and exc.response.status_code == 429:
                    try:
                        delay = min(max(float(exc.response.headers.get("Retry-After", 10)), 10), 60)
                    except ValueError:
                        delay = 10
                time.sleep(delay)
    return {**result, "status": "failed", "attempts": attempt, "error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/assignment3/raw/nytimes_dataset.json")
    parser.add_argument("--per-class", type=int, default=0, help="0 selects all metadata groups (default); positive values cap each class")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--run-name", default="full_seed44_original")
    parser.add_argument("--image-source", choices=("metadata", "original"), default="original",
                        help="original removes query-only rendering parameters from the same public image asset URL")
    parser.add_argument("--diagnostic-timeout", type=int, default=0)
    args = parser.parse_args()
    if args.diagnostic_timeout:
        faulthandler.dump_traceback_later(args.diagnostic_timeout, repeat=True)
    if args.per_class < 0 or args.workers not in range(1, 33) or args.retries < 0 or args.timeout <= 0:
        parser.error("Invalid per-class/workers/retries/timeout setting")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run-name must contain only letters, numbers, underscores or hyphens")
    selected = select_records(load_records(args.metadata), args.per_class, args.seed)
    folder = "images_original" if args.image_source == "original" else "images"
    image_dir = ROOT / "data/assignment3/raw" / folder
    image_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "data/assignment3/processed" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "selection.json", selected)
    log_path = run_dir / "downloads.jsonl"
    started = time.monotonic()
    results = []
    print(f"Selected {len(selected)} records / {len(set(r['section'] for r in selected))} classes", flush=True)
    with log_path.open("w", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, image_dir, args.timeout, args.retries, args.image_source) for row in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if len(results) % 100 == 0 or len(results) == len(selected):
                counts = Counter(row["status"] for row in results)
                elapsed = time.monotonic() - started
                print(f"{len(results)}/{len(selected)} {dict(counts)} elapsed={elapsed:.1f}s", flush=True)
    results.sort(key=lambda row: row["image_id"])
    fields = ["image_id", "section", "path", "image_source", "status", "width", "height", "bytes", "pixel_sha256", "file_sha256", "attempts", "error"]
    with (run_dir / "image_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    valid = [row for row in results if row["status"] != "failed"]
    summary = {
        "run_name": args.run_name, "selection_sha256": selection_digest(selected),
        "seed": args.seed, "per_class_cap": args.per_class,
        "selected": len(selected), "class_counts": dict(sorted(Counter(row["section"] for row in selected).items())),
        "valid_images": len(valid), "failed_images": len(results) - len(valid),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "total_valid_file_bytes": sum(row["bytes"] for row in valid),
        "workers": args.workers, "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_source": args.image_source,
        "scope": "image download/verification only; semantic pair quality not verified",
    }
    write_json(run_dir / "download_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
