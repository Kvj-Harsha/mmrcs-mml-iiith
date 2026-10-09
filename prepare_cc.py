"""Download Conceptual Captions (CC3M): validation images + captions, and train captions (text only).

    python prepare_cc.py                  # first run: ~30-40 min (image download), re-runs skip finished work
    python prepare_cc.py --retry-failed   # also re-try links that failed last time

Conceptual Captions (Sharma et al., ACL 2018) has 3,318,333 train and 15,840 validation pairs (the ~12.5K test split
is hidden). It distributes image URLs, not images, and many links have died since 2018, so we download the
validation images ourselves and record what failed. Train captions are downloaded as text only (for vocabulary).

Result:
    data/cc/val/images/            downloaded validation images (original bytes), cc_val_<row>.<ext>
    data/cc/val/captions.json      one entry per downloaded image: {image, row, url, caption}
    data/cc/val/all_captions.json  all 15,840 validation rows (including dead links), for text analysis
    data/cc/val/download_log.csv   status of every row: ok / http_error / connection_error / timeout / not_image /
                                   corrupt / placeholder / junk
    data/cc/train_captions.txt     all 3,318,333 train captions, one per line (no images)
    data/downloads/cc/             the original parquet files (Hugging Face mirror of the official TSVs)

Links keep dying, so the number of downloaded images can differ slightly between runs and teammates.
Needs Python 3.9+, pandas, pyarrow, Pillow.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from prepare_data import download

HF = ("https://huggingface.co/datasets/google-research-datasets/conceptual_captions/resolve/"
      "refs%2Fconvert%2Fparquet/unlabeled")
VAL_PARQUET = f"{HF}/validation/0000.parquet"
TRAIN_PARQUETS = [f"{HF}/train/0000.parquet", f"{HF}/train/0001.parquet"]
N_VAL, N_TRAIN = 15_840, 3_318_333

ROOT = Path(__file__).resolve().parent / "data"
RAW = ROOT / "downloads" / "cc"
OUT = ROOT / "cc"
VAL = OUT / "val"
IMG = VAL / "images"

THREADS, TIMEOUT, RETRIES, MAX_BYTES = 32, 10, 1, 20_000_000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif", "MPO": "jpg"}
PLACEHOLDER_MIN_URLS = 3
# Junk images that still decode: blank (one colour), extreme aspect ratio (the CC paper only kept aspect <= 2, so > 3
# means the server now returns something else, e.g. a rendered error message) or tiny.
JUNK_MAX_ASPECT, JUNK_MIN_SIDE = 3, 100
LOG_FIELDS = ["row", "url", "domain", "status", "http_code", "error", "bytes", "width", "height", "mode", "format",
              "sha1", "image"]


def inspect(data: bytes) -> dict:
    """Decode image bytes; raise if they are not a supported, intact image."""
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in EXT:
            raise ValueError(f"unsupported format {im.format}")
        im.load()
        return {"width": im.width, "height": im.height, "mode": im.mode, "format": im.format}


def fetch(row: int, url: str) -> dict:
    """Download one validation image; returns its log entry."""
    rec = {"row": row, "url": url, "domain": urllib.parse.urlparse(url).netloc.lower(), "status": "", "http_code": "",
           "error": "", "bytes": "", "width": "", "height": "", "mode": "", "format": "", "sha1": "", "image": ""}
    existing = list(IMG.glob(f"cc_val_{row:05d}.*"))
    if existing:  # resume: already downloaded and fully validated in an earlier run, so only read the header
        data = existing[0].read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            meta = {"width": im.width, "height": im.height, "mode": im.mode, "format": im.format}
        rec.update(status="ok", http_code=200, bytes=len(data), sha1=hashlib.sha1(data).hexdigest(),
                   image=existing[0].name, **meta)
        return rec
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                rec["http_code"] = resp.status
                ctype = resp.headers.get("Content-Type", "")
                data = resp.read(MAX_BYTES)
            try:
                meta = inspect(data)
            except Exception as e:
                rec.update(status="not_image" if "html" in ctype or "text" in ctype else "corrupt", error=str(e)[:120])
                return rec
            name = f"cc_val_{row:05d}.{EXT[meta['format']]}"
            (IMG / name).write_bytes(data)
            rec.update(status="ok", bytes=len(data), sha1=hashlib.sha1(data).hexdigest(), image=name, error="", **meta)
            return rec
        except urllib.error.HTTPError as e:
            rec.update(status="http_error", http_code=e.code, error=str(e)[:120])
            return rec  # a real HTTP answer: retrying will not help
        except Exception as e:  # timeouts, DNS failures, refused connections, SSL errors
            rec.update(status="timeout" if "timed out" in str(e).lower() else "connection_error",
                       error=f"{type(e).__name__}: {str(e)[:100]}")
    return rec


def download_images(val: pd.DataFrame, retry_failed: bool) -> list[dict]:
    """Fetch every validation image (in parallel), reusing the previous log for failed rows unless asked to retry."""
    IMG.mkdir(parents=True, exist_ok=True)
    log_path = VAL / "download_log.csv"
    previous = {}
    if log_path.exists() and not retry_failed:
        with open(log_path, encoding="utf-8", newline="") as f:
            previous = {int(r["row"]): r for r in csv.DictReader(f) if r["status"] not in ("ok", "placeholder")}
    todo = [(r, u) for r, u in zip(val.row, val.image_url) if r not in previous]
    print(f"[images] {len(todo):,} to fetch, {len(previous):,} known failures reused (use --retry-failed to re-try)")
    records = list(previous.values())
    with ThreadPoolExecutor(THREADS) as ex:
        futures = [ex.submit(fetch, r, u) for r, u in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 500 == 0 or i == len(futures):
                ok = sum(r["status"] == "ok" for r in records)
                print(f"\r[images] {i:,}/{len(futures):,} done, {ok:,} ok so far", end="", flush=True)
    print()
    return sorted(records, key=lambda r: int(r["row"]))


def junk_reason(path: Path, width: int, height: int) -> str:
    """Why a decodable image is still unusable, or '' if it is fine."""
    if min(width, height) < JUNK_MIN_SIDE:
        return f"tiny image ({width}x{height})"
    if max(width, height) / min(width, height) > JUNK_MAX_ASPECT:
        return f"extreme aspect ratio ({width}x{height})"
    with Image.open(path) as im:
        lo, hi = im.convert("L").getextrema()
    return "blank image (one colour)" if lo == hi else ""


def mark_invalid(records: list[dict]) -> None:
    """Drop images that downloaded fine but are not real photos.

    placeholder: the same file served for >= 3 different URLs (dead hosts' 'image not available' picture).
    junk: blank, tiny or extreme-aspect images (e.g. a rendered server error message).
    """
    by_hash = Counter(r["sha1"] for r in records if r["status"] == "ok")
    for r in records:
        if r["status"] != "ok":
            continue
        reason = (f"same file as {by_hash[r['sha1']]} URLs" if by_hash[r["sha1"]] >= PLACEHOLDER_MIN_URLS else "")
        status = "placeholder" if reason else "junk"
        reason = reason or junk_reason(IMG / r["image"], int(r["width"]), int(r["height"]))
        if reason:
            (IMG / r["image"]).unlink(missing_ok=True)
            r.update(status=status, error=reason, image="")


def write_train_captions() -> int:
    """Stream train captions from parquet to a text file (one per line) without loading 3.3M rows at once."""
    target = OUT / "train_captions.txt"
    if target.exists():
        n = sum(1 for _ in open(target, encoding="utf-8"))
        if n == N_TRAIN:
            print(f"[skip] train_captions.txt already written ({n:,} lines)")
            return n
    n = 0
    with open(target, "w", encoding="utf-8") as f:
        for i in range(len(TRAIN_PARQUETS)):
            for batch in pq.ParquetFile(RAW / f"train_{i:04d}.parquet").iter_batches(batch_size=100_000, columns=["caption"]):
                for cap in batch.column(0).to_pylist():
                    f.write(" ".join(cap.split()) + "\n")
                    n += 1
    print(f"[train] wrote {n:,} train captions")
    return n


def fingerprint(rows) -> str:
    return hashlib.sha256(",".join(map(str, sorted(rows))).encode()).hexdigest()[:12]


def check(val: pd.DataFrame, records: list[dict], n_train: int) -> bool:
    print("\n=== Checks ===")
    caps = json.load(open(VAL / "captions.json", encoding="utf-8"))
    files = {p.name for p in IMG.iterdir()}
    status = Counter(r["status"] for r in records)
    ok = True
    for name, cond in [
        ("15,840 validation rows, all logged once", len(val) == N_VAL and len(records) == N_VAL
         and len({int(r["row"]) for r in records}) == N_VAL),
        ("every image file has exactly one caption entry and vice versa",
         files == {c["image"] for c in caps} and len(caps) == len(files)),
        ("every caption entry points to an 'ok' row", len(caps) == status["ok"]),
        ("no empty captions", all(c["caption"].strip() for c in caps)),
        (f"train captions: {n_train:,} lines", n_train == N_TRAIN),
    ]:
        print(f"{'PASS' if cond else 'FAIL'}  {name}")
        ok &= cond
    print("download status:", {k: f"{v:,} ({v / N_VAL:.1%})" for k, v in status.most_common()})
    print(f"downloaded validation images: {len(caps):,} | fingerprint {fingerprint(c['row'] for c in caps)}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retry-failed", action="store_true", help="re-try links that failed in an earlier run")
    args = parser.parse_args()

    download(VAL_PARQUET, RAW / "validation_0000.parquet")
    for i, url in enumerate(TRAIN_PARQUETS):
        download(url, RAW / f"train_{i:04d}.parquet")
    VAL.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(RAW / "validation_0000.parquet").reset_index().rename(columns={"index": "row"})
    with open(VAL / "all_captions.json", "w", encoding="utf-8") as f:
        json.dump([{"row": int(r), "url": u, "caption": c} for r, u, c in zip(val.row, val.image_url, val.caption)],
                  f, ensure_ascii=False)

    records = download_images(val, args.retry_failed)
    mark_invalid(records)
    with open(VAL / "download_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(records)

    caption_of = dict(zip(val.row, val.caption))
    with open(VAL / "captions.json", "w", encoding="utf-8") as f:
        json.dump([{"image": r["image"], "row": int(r["row"]), "url": r["url"], "caption": caption_of[int(r["row"])]}
                   for r in records if r["status"] == "ok"], f, indent=1, ensure_ascii=False)

    n_train = write_train_captions()
    sys.exit(0 if check(val, records, n_train) else 1)


if __name__ == "__main__":
    main()
