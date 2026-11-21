"""Data cleaning for the 10K COCO task.

Two stages, so the cheap rules run over all of COCO and the slow ones only over images that were
actually selected:

    build_candidates()   metadata rules over every train2017/val2017 image (labels, captions, size)
    check_images()       decode check and perceptual hash of chosen images (done by split.py)

Every rule counts what it removed, and the counts end up in results/task10k/cleaning_report.json.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

MIN_SIDE = 128           # pixels; smaller images are dropped
MIN_AREA_FRAC = 0.005    # an object counts toward the image label if it covers this share of the image
MIN_WORDS, MAX_WORDS = 3, 50
N_CAPTIONS = 5
DUP_HAMMING = 4          # dHash distance at or below this means "same picture"

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACE = re.compile(r"\s+")


def clean_caption(text):
    """Normalised caption, or None if it is unusable. Case and punctuation are kept (CLIP is cased
    and punctuation-insensitive); only whitespace, control characters and unicode forms are fixed."""
    text = unicodedata.normalize("NFKC", text)
    text = _SPACE.sub(" ", _CONTROL.sub(" ", text)).strip()
    if not any(ch.isalpha() for ch in text):
        return None
    n_words = len(text.split())
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        return None
    return text


def clean_caption_set(raw, stats=None):
    """First N_CAPTIONS distinct usable captions, or None if the image has fewer."""
    stats = stats if stats is not None else Counter()
    seen, out = set(), []
    for text in raw:
        c = clean_caption(text)
        if c is None:
            stats["caption_unusable"] += 1
            continue
        key = c.lower().rstrip(". ")
        if key in seen:
            stats["caption_duplicate"] += 1
            continue
        seen.add(key)
        out.append(c)
    if len(out) < N_CAPTIONS:
        return None
    return out[:N_CAPTIONS]


def image_labels(annotations, width, height, min_frac=MIN_AREA_FRAC):
    """Sorted category ids with at least one non-crowd instance covering >= min_frac of the image."""
    area = float(width * height)
    return sorted({a["category_id"] for a in annotations
                   if not a.get("iscrowd", 0) and a["area"] / area >= min_frac})


def build_candidates(ann_dir, object_labels_json, stats=None):
    """{image_id: record} for every image that passes the metadata rules.

    A record holds file, dir ('train2017' or 'val2017'), karpathy split, labels, captions, size.
    """
    stats = stats if stats is not None else Counter()
    ann_dir = Path(ann_dir)
    with open(object_labels_json) as f:
        meta = json.load(f)["images"]

    out = {}
    for split in ("train2017", "val2017"):
        with open(ann_dir / f"instances_{split}.json") as f:
            inst = json.load(f)
        with open(ann_dir / f"captions_{split}.json") as f:
            caps = json.load(f)
        by_image, caps_by_image = {}, {}
        for a in inst["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for c in caps["annotations"]:
            caps_by_image.setdefault(c["image_id"], []).append(c["caption"])

        for im in inst["images"]:
            stats[f"{split}_images"] += 1
            i = im["id"]
            m = meta.get(str(i))
            if m is None:
                stats["no_metadata"] += 1
                continue
            if min(im["width"], im["height"]) < MIN_SIDE:
                stats["too_small"] += 1
                continue
            anns = by_image.get(i, [])
            labels = image_labels(anns, im["width"], im["height"])
            if not labels:
                stats["no_label_after_area_rule" if anns else "no_annotation"] += 1
                continue
            captions = clean_caption_set(caps_by_image.get(i, []), stats)
            if captions is None:
                stats["fewer_than_5_captions"] += 1
                continue
            out[i] = {"id": i, "file": m["file"], "dir": m["dir"], "karpathy": m["karpathy"],
                      "labels": labels, "captions": captions, "raw_captions": caps_by_image[i][:7],
                      "width": im["width"], "height": im["height"]}
    stats["candidates"] = len(out)
    return out


def check_image(path):
    """(ok, reason, mode). Full decode, so truncated files are caught."""
    try:
        with Image.open(path) as im:
            mode = im.mode
            im.load()
            if min(im.size) < MIN_SIDE:
                return False, "too_small_on_disk", mode
        return True, "", mode
    except Exception as e:  # corrupt, truncated, unsupported
        return False, f"unreadable:{type(e).__name__}", ""


def dhash_bits(path):
    """64-bit difference hash as a uint64."""
    import imagehash
    with Image.open(path) as im:
        h = imagehash.dhash(im.convert("RGB"), hash_size=8)
    return int(np.packbits(h.hash.flatten()).view(">u8")[0])


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a, b):
    """Hamming distance between uint64 hashes; broadcasts."""
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    flat = np.ascontiguousarray(x).reshape(-1)
    return _POP8[flat.view(np.uint8).reshape(-1, 8)].sum(axis=1).reshape(x.shape)


def near_duplicates(hashes, max_dist=DUP_HAMMING):
    """Index pairs (i < j) whose hashes are within max_dist."""
    h = np.asarray(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h)):
        d = hamming(h[i], h[i + 1:])
        pairs += [(i, i + 1 + int(j)) for j in np.flatnonzero(d <= max_dist)]
    return pairs


def main():
    import argparse
    import random
    import time

    p = argparse.ArgumentParser(description="Build the cleaned candidate table (metadata rules).")
    p.add_argument("--ann-dir", default="data/annotations")
    p.add_argument("--object-labels", default="data/extracted_labels/object_labels.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="data/task10k/candidates.json")
    p.add_argument("--report", default="results/task10k/cleaning_report.json")
    p.add_argument("--check-sample", type=int, default=0,
                   help="also decode-check and hash this many random candidates (gate run)")
    args = p.parse_args()

    t0 = time.time()
    stats = Counter()
    cands = build_candidates(args.ann_dir, args.object_labels, stats)
    print(f"{len(cands)} candidates in {time.time() - t0:.0f}s")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cands, f)
    report = {"metadata_rules": dict(stats)}

    if args.check_sample:
        ids = random.Random(0).sample(sorted(cands), args.check_sample)
        modes, bad, hashes = Counter(), [], {}
        for i in ids:
            c = cands[i]
            path = Path(args.data_root) / c["dir"] / c["file"]
            ok, reason, mode = check_image(path)
            modes[mode] += 1
            if not ok:
                bad.append((i, reason))
                continue
            hashes[i] = dhash_bits(path)
        order = list(hashes)
        dups = [(order[a], order[b]) for a, b in near_duplicates([hashes[i] for i in order])]
        print(f"sample {len(ids)}: modes {dict(modes)}, unreadable/small {len(bad)}, near-duplicate pairs {len(dups)}")
        report["sample_check"] = {"n": len(ids), "modes": dict(modes), "bad": bad, "duplicate_pairs": dups}

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
