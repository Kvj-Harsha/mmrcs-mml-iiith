"""Download and build the COCO dataset (Karpathy split).

Everyone runs the same command and gets the same images in the same split:

    python prepare_data.py                     # 30k train (default)
    python prepare_data.py --train-size 10k    # small, fast version
    python prepare_data.py --train-size full   # all 113,287 train images (+13 GB download, for the DGX)
    python prepare_data.py --delete-zip        # delete the image zips at the end

Training-set sizes (val and test are always the official Karpathy 5,000 + 5,000):
    10k    10,000 images  random 10k of Karpathy "restval" (seed 42)        ~50/25/25
    30k    30,504 images  all of Karpathy "restval" (in val2014.zip)       ~75/12.5/12.5
    full  113,287 images  restval + Karpathy "train" (needs train2014.zip) ~92/4/4 (paper standard)
The sets are nested: 10k is inside 30k, and 30k is inside full.

Result (one folder per split, everything about an image lives in its split folder):
    data/coco/{train,val,test}/
        images/          the photos
        captions.json    5+ human captions per image
        objects.json     labelled objects per image (category + bounding box)
    data/downloads/      the original downloaded files

Switching size rebuilds data/coco/train/ (extra images are removed, missing ones extracted).

Cross-split duplicates: COCO contains a few photos uploaded twice under different ids, and the Karpathy split puts
some of them in two splits. They are detected (perceptual hash, then confirmed by pixel correlation) and the extra
copy is dropped so no photo appears in two splits. Test is never touched (stays the official 5,000):
    train <-> val/test   -> drop the train copy
    val   <-> test       -> drop the val copy
The dropped images are listed in data/coco/removed_duplicates.json.

Needs Python 3.9+ and Pillow (preinstalled on Colab).
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path

from PIL import Image

KARPATHY_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
IMAGE_URLS = {
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
}
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

TRAIN_SIZES = {"10k": 10_000, "30k": 30_504, "full": 113_287}  # before removing cross-split duplicates
SEED = 42

# Duplicate detection: candidates differ in <= 3 of 64 dHash bits; confirmed if the 64x64 grayscale thumbnails
# correlate > 0.95 and the aspect ratios match. Calibrated on hand-checked pairs: true duplicates 0.97-1.00,
# false hash matches 0.17-0.69.
HASH_BITS = 3
MIN_CORR = 0.95
MAX_ASPECT_DIFF = 0.05
KEEP_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = kept when a photo is in two splits

ROOT = Path(__file__).resolve().parent / "data"
RAW = ROOT / "downloads"
OUT = ROOT / "coco"


def download(url: str, dest: Path) -> None:
    """Download url to dest, skipping if it already exists."""
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done, last_pct = 0, -1
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct != last_pct:
                    print(f"\r  {pct:3d}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
                    last_pct = pct
    print()
    tmp.rename(dest)


def load_karpathy() -> list[dict]:
    """Return the image entries from dataset_coco.json."""
    json_path = RAW / "dataset_coco.json"
    if not json_path.exists():
        with zipfile.ZipFile(RAW / "caption_datasets.zip") as zf:
            with zf.open("dataset_coco.json") as src, open(json_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["images"]


def extract_annotations(folders: set[str]) -> None:
    """Extract the official COCO captions + object-label files to data/downloads/annotations/."""
    out_dir = RAW / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{kind}_{folder}.json" for folder in sorted(folders) for kind in ("captions", "instances")]
    with zipfile.ZipFile(RAW / "annotations_trainval2014.zip") as zf:
        for name in names:
            target = out_dir / name
            if not target.exists():
                with zf.open(f"annotations/{name}") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"[annotations] {', '.join(names)} ready")


def build_splits(images: list[dict], train_size: str) -> dict[str, list[dict]]:
    """Official Karpathy val/test + a nested train set: 10k ⊂ 30k (all restval) ⊂ full (restval + train)."""
    by_split = {"val": [], "test": [], "restval": [], "train": []}
    for img in images:
        by_split[img["split"]].append(img)

    restval = sorted(by_split["restval"], key=lambda x: x["cocoid"])
    if train_size == "10k":
        train = random.Random(SEED).sample(restval, TRAIN_SIZES["10k"])
    elif train_size == "30k":
        train = restval
    else:
        train = restval + by_split["train"]

    splits = {"train": train, "val": by_split["val"], "test": by_split["test"]}
    for name in splits:
        splits[name] = sorted(splits[name], key=lambda x: x["cocoid"])
    return splits


def extract_images(splits: dict[str, list[dict]]) -> None:
    """Copy only the needed images out of the zips into per-split folders; remove images no longer in a split."""
    zips = {}
    try:
        for name, items in splits.items():
            img_dir = OUT / name / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            wanted = {x["filename"] for x in items}
            stale = [p for p in img_dir.iterdir() if p.name not in wanted]
            for p in stale:
                p.unlink()
            if stale:
                print(f"[cleanup] {name}: removed {len(stale)} images not in this split")
            for i, item in enumerate(items, 1):
                folder = item["filepath"]
                if folder not in zips:
                    zips[folder] = zipfile.ZipFile(RAW / f"{folder}.zip")
                member = zips[folder].getinfo(f"{folder}/{item['filename']}")
                target = img_dir / item["filename"]
                if not (target.exists() and target.stat().st_size == member.file_size):
                    with zips[folder].open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                if i % 500 == 0 or i == len(items):
                    print(f"\r[extract] {name}: {i}/{len(items)}", end="", flush=True)
            print()
    finally:
        for zf in zips.values():
            zf.close()


def image_hashes(splits: dict[str, list[dict]]) -> dict[str, int]:
    """64-bit difference hash (dHash) of every image; cached in data/downloads/image_hashes.json."""
    cache_path = RAW / "image_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    hashes, todo = {}, [(name, x["filename"]) for name, items in splits.items() for x in items]
    for i, (name, fname) in enumerate(todo, 1):
        if fname not in cache:
            with Image.open(OUT / name / "images" / fname) as im:
                px = list(im.convert("L").resize((9, 8), Image.BILINEAR).tobytes())
            cache[fname] = sum(1 << b for b, (r, c) in enumerate((r, c) for r in range(8) for c in range(8))
                               if px[r * 9 + c + 1] > px[r * 9 + c])
        hashes[fname] = cache[fname]
        if i % 2000 == 0 or i == len(todo):
            print(f"\r[hash] {i}/{len(todo)}", end="", flush=True)
    print()
    cache_path.write_text(json.dumps(cache))
    return hashes


def thumb_and_aspect(path: Path) -> tuple[list[float], float]:
    with Image.open(path) as im:
        return list(im.convert("L").resize((64, 64), Image.BILINEAR).tobytes()), im.width / im.height


def correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5 if va and vb else 0.0


def remove_cross_split_duplicates(splits: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Find photos that appear in two splits and drop the copy from the lower-priority split."""
    split_of = {x["filename"]: name for name, items in splits.items() for x in items}
    hashes = image_hashes(splits)
    names = list(hashes)
    candidates = set()
    for k in range(4):  # pigeonhole: <= 3 differing bits means at least one 16-bit chunk is identical
        buckets = {}
        for f in names:
            buckets.setdefault((hashes[f] >> (16 * k)) & 0xFFFF, []).append(f)
        for group in buckets.values():
            if 1 < len(group) <= 300:
                candidates.update(tuple(sorted(p)) for p in combinations(group, 2) if split_of[p[0]] != split_of[p[1]])

    removed = []
    for a, b in sorted(candidates):
        if bin(hashes[a] ^ hashes[b]).count("1") > HASH_BITS:
            continue
        (ta, aspect_a), (tb, aspect_b) = thumb_and_aspect(OUT / split_of[a] / "images" / a), \
                                         thumb_and_aspect(OUT / split_of[b] / "images" / b)
        corr = correlation(ta, tb)
        if corr > MIN_CORR and abs(aspect_a - aspect_b) < MAX_ASPECT_DIFF:
            keep, drop = sorted([a, b], key=lambda f: KEEP_PRIORITY[split_of[f]])
            removed.append({"dropped": drop, "dropped_from": split_of[drop], "kept": keep,
                            "kept_in": split_of[keep], "pixel_correlation": round(corr, 4)})

    drop_files = {r["dropped"] for r in removed}
    cleaned = {name: [x for x in items if x["filename"] not in drop_files] for name, items in splits.items()}
    print(f"[duplicates] {len(removed)} photos found in two splits -> dropped: "
          + ", ".join(f"{n} {sum(r['dropped_from'] == n for r in removed)}" for n in ("train", "val", "test")))
    return cleaned, removed


def write_captions(splits: dict[str, list[dict]]) -> None:
    for name, items in splits.items():
        records = [
            {
                "image": x["filename"],
                "cocoid": x["cocoid"],
                "captions": [s["raw"].strip() for s in x["sentences"]],
            }
            for x in items
        ]
        with open(OUT / name / "captions.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=1, ensure_ascii=False)


def write_objects(splits: dict[str, list[dict]], folders: set[str]) -> None:
    """Per split: the official COCO object labels of each image, matched by image id."""
    cats, meta, objs = {}, {}, {}
    for folder in sorted(folders):
        with open(RAW / "annotations" / f"instances_{folder}.json", encoding="utf-8") as f:
            inst = json.load(f)
        cats.update({c["id"]: c for c in inst["categories"]})
        meta.update({im["id"]: im for im in inst["images"]})
        for a in inst["annotations"]:
            c = cats[a["category_id"]]
            objs.setdefault(a["image_id"], []).append({
                "category": c["name"],
                "supercategory": c["supercategory"],
                "bbox": [round(v, 1) for v in a["bbox"]],  # x, y, width, height in pixels
                "area": round(a["area"], 1),
                "iscrowd": a["iscrowd"],
            })
    for name, items in splits.items():
        records = [
            {
                "image": x["filename"],
                "cocoid": x["cocoid"],
                "width": meta[x["cocoid"]]["width"],
                "height": meta[x["cocoid"]]["height"],
                "objects": objs.get(x["cocoid"], []),
            }
            for x in items
        ]
        with open(OUT / name / "objects.json", "w", encoding="utf-8") as f:
            json.dump(records, f)


def fingerprint(items: list[dict]) -> str:
    ids = ",".join(str(x["cocoid"]) for x in items)
    return hashlib.sha256(ids.encode()).hexdigest()[:12]


def check(splits: dict[str, list[dict]], train_size: str, removed: list[dict]) -> bool:
    print(f"\n=== Checks (train size: {train_size}) ===")
    ok = True
    ids = {name: {x["cocoid"] for x in items} for name, items in splits.items()}

    for name, items in splits.items():
        folder = OUT / name
        files = {p.name for p in (folder / "images").iterdir() if p.stat().st_size > 0}
        caps = json.load(open(folder / "captions.json", encoding="utf-8"))
        objs = json.load(open(folder / "objects.json", encoding="utf-8"))
        expected_files = {x["filename"] for x in items}
        same = (
            files == expected_files
            and {c["image"] for c in caps} == expected_files
            and {o["image"] for o in objs} == expected_files
            and all(c["image"] == o["image"] and c["cocoid"] == o["cocoid"] for c, o in zip(caps, objs))
        )
        id_in_name = all(int(c["image"].split("_")[-1].split(".")[0]) == c["cocoid"] for c in caps)
        n_caps = [len(c["captions"]) for c in caps]
        print(
            f"{name:5s}: {len(files):6d} image files | {len(caps):6d} with captions | {len(objs):6d} with object info"
            f" | captions/image {min(n_caps)}-{max(n_caps)} | images=captions=objects: {same}"
            f" | filename id = caption id: {id_in_name} | fingerprint {fingerprint(items)}"
        )
        ok &= same and id_in_name and min(n_caps) >= 5

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = len(ids[a] & ids[b])
        print(f"overlap {a}/{b}: {overlap} image ids")
        ok &= overlap == 0

    dropped = {r["dropped"] for r in removed}
    leftover = sum(x["filename"] in dropped for items in splits.values() for x in items)
    print(f"cross-split duplicate photos: {len(removed)} found, {leftover} remaining")
    ok &= leftover == 0

    base = {"train": TRAIN_SIZES[train_size], "val": 5000, "test": 5000}
    expected = {k: v - sum(r["dropped_from"] == k for r in removed) for k, v in base.items()}
    ok &= all(len(splits[k]) == v for k, v in expected.items()) and expected["test"] == 5000
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-size", choices=sorted(TRAIN_SIZES), default="30k",
                        help="training-set size (default: 30k); val/test are always the official 5k + 5k")
    parser.add_argument("--delete-zip", action="store_true", help="delete the image zips after extraction")
    args = parser.parse_args()

    download(KARPATHY_URL, RAW / "caption_datasets.zip")
    splits = build_splits(load_karpathy(), args.train_size)
    folders = {x["filepath"] for items in splits.values() for x in items}  # val2014, + train2014 for "full"

    for folder in sorted(folders):
        download(IMAGE_URLS[folder], RAW / f"{folder}.zip")
    download(ANNOTATIONS_URL, RAW / "annotations_trainval2014.zip")
    extract_annotations(folders)

    extract_images(splits)
    splits, removed = remove_cross_split_duplicates(splits)
    extract_images(splits)  # removes the dropped copies from the image folders
    (OUT / "removed_duplicates.json").write_text(json.dumps(removed, indent=1))
    write_captions(splits)
    write_objects(splits, folders)
    ok = check(splits, args.train_size, removed)

    if args.delete_zip and ok:
        for folder in sorted(folders):
            (RAW / f"{folder}.zip").unlink()
            print(f"[cleanup] deleted {folder}.zip")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
