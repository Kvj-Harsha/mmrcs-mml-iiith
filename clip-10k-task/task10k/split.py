"""Build the 10K split from the cleaned candidates.

    train 5,000  train2017, class-balanced (quota greedy), not Karpathy val/test
    test  3,000  train2017, stratified, drawn only from Karpathy val/test images (never seen by BLIP)
    val   2,000  official val2017, stratified

then decode-check and perceptual-hash the chosen images and re-select around any bad or duplicated
image until the split is clean.

    python -m task10k.split                       # full 5000 / 3000 / 2000
    python -m task10k.split --n-train 500 --n-test 300 --n-val 200 --out-dir data/task10k_gate
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from selection.metrics import entropy, gini
from task10k import clean

UNSEEN = ("val", "test")  # Karpathy splits BLIP did not train on


def label_matrix(records, cats):
    col = {c: j for j, c in enumerate(cats)}
    X = np.zeros((len(records), len(cats)), dtype=bool)
    for i, r in enumerate(records):
        X[i, [col[c] for c in r["labels"]]] = True
    return X


def stratified_subset(X, n, seed):
    """n rows of X whose label proportions follow the whole table (iterative stratification,
    Sechidis et al. 2011, two folds). Returns sorted row indices."""
    rng = np.random.default_rng(seed)
    N, C = X.shape
    fold_frac = np.array([n / N, 1 - n / N])
    desired = fold_frac[:, None] * X.sum(0)[None, :].astype(float)   # (2, C) label quota per fold
    size = fold_frac * N
    fold = np.full(N, -1)
    remaining = np.ones(N, dtype=bool)
    while remaining.any():
        counts = X[remaining].sum(0)
        live = np.flatnonzero(counts > 0)
        if len(live) == 0:                      # rows with no label: split by size only
            idx = np.flatnonzero(remaining)
            for i in rng.permutation(idx):
                f = int(np.argmax(size + rng.random(2) * 1e-9))
                fold[i], size[f] = f, size[f] - 1
            break
        c = int(rng.choice(live[counts[live] == counts[live].min()]))
        for i in rng.permutation(np.flatnonzero(remaining & X[:, c])):
            best = desired[:, c]
            cand = np.flatnonzero(best == best.max())
            if len(cand) > 1:
                cand = cand[size[cand] == size[cand].max()]
            f = int(rng.choice(cand))
            fold[i] = f
            desired[f] -= X[i]
            size[f] -= 1
            remaining[i] = False
    chosen = np.flatnonzero(fold == 0)
    if len(chosen) > n:
        chosen = rng.choice(chosen, n, replace=False)
    elif len(chosen) < n:
        extra = rng.choice(np.flatnonzero(fold == 1), n - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def quota_balanced(X, budget, seed, target=150, over_penalty=0.5):
    """Greedy class balancing: each step takes the image that helps the most classes still under
    `target` images and adds the fewest to classes already over it. Plain coverage objectives do not
    balance counts (Gini 0.50 vs 0.49 for random on this pool); this reaches about 0.08."""
    rng = np.random.default_rng(seed)
    Xf = X.astype(np.float32)
    n = np.zeros(X.shape[1], dtype=np.float32)
    taken = np.zeros(len(X), dtype=bool)
    out = []
    for _ in range(budget):
        u = np.where(n < target, 1.0 - n / target, -over_penalty).astype(np.float32)
        g = Xf @ u
        g[taken] = -np.inf
        i = int(rng.choice(np.flatnonzero(g >= g.max() - 1e-6)))
        taken[i] = True
        n += Xf[i]
        out.append(i)
    return np.sort(np.asarray(out))


def assert_disjoint_and_valid(splits):
    """splits: {'train': [...], 'test': [...], 'val': [...]} of candidate records."""
    ids = {k: [r["id"] for r in v] for k, v in splits.items()}
    for k, v in ids.items():
        assert len(set(v)) == len(v), f"{k}: repeated image id"
    names = list(ids)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            both = set(ids[names[a]]) & set(ids[names[b]])
            assert not both, f"{names[a]} and {names[b]} share {len(both)} images, e.g. {sorted(both)[:3]}"
    for k in ("train", "test"):
        assert all(r["dir"] == "train2017" for r in splits[k]), f"{k} must come from train2017"
    assert all(r["dir"] == "val2017" for r in splits["val"]), "val must come from val2017"
    assert all(r["karpathy"] in UNSEEN for r in splits["test"]), "test must be Karpathy val/test images"
    for k, v in splits.items():
        assert all(len(r["captions"]) == clean.N_CAPTIONS and r["labels"] for r in v), f"{k}: caption/label rule broken"


def _inspect(task):
    """Worker: decode check and hash of one image. task = (id, path)."""
    i, path = task
    ok, reason, mode = clean.check_image(path)
    return i, ok, reason, mode, (clean.dhash_bits(path) if ok else 0)


def inspect_images(records, data_root, cache, workers=12):
    """Fill `cache` ({id: (ok, reason, mode, hash)}) for records not yet inspected."""
    todo = [(r["id"], str(Path(data_root) / r["dir"] / r["file"])) for r in records if r["id"] not in cache]
    with ProcessPoolExecutor(workers) as ex:
        for i, ok, reason, mode, h in ex.map(_inspect, todo, chunksize=64):
            cache[i] = (ok, reason, mode, h)


def pick(cands, n_train, n_test, n_val, exclude, seed, target):
    """One selection pass. cands: list of records. Returns splits dict of records."""
    pool = [r for r in cands if r["id"] not in exclude]
    test_pool = [r for r in pool if r["dir"] == "train2017" and r["karpathy"] in UNSEEN]
    val_pool = [r for r in pool if r["dir"] == "val2017"]
    cats = sorted({c for r in cands for c in r["labels"]})
    Xt = label_matrix(test_pool, cats)
    test = [test_pool[i] for i in stratified_subset(Xt, n_test, seed)]
    taken = {r["id"] for r in test}
    train_pool = [r for r in pool if r["dir"] == "train2017" and r["karpathy"] not in UNSEEN
                  and r["id"] not in taken]
    Xr = label_matrix(train_pool, cats)
    train = [train_pool[i] for i in quota_balanced(Xr, n_train, seed, target)]
    Xv = label_matrix(val_pool, cats)
    val = [val_pool[i] for i in stratified_subset(Xv, n_val, seed)]
    return {"train": train, "test": test, "val": val}, {"train": train_pool, "test": test_pool, "val": val_pool}


def build(cands, n_train, n_test, n_val, data_root, seed=0, target=150, max_rounds=12, log=print):
    exclude, cache = set(), {}
    removed = Counter()
    for rnd in range(1, max_rounds + 1):
        splits, pools = pick(cands, n_train, n_test, n_val, exclude, seed, target)
        allrec = [r for k in ("train", "test", "val") for r in splits[k]]
        inspect_images(allrec, data_root, cache)
        bad = {r["id"] for r in allrec if not cache[r["id"]][0]}
        order = [r for r in allrec if r["id"] not in bad]       # train first, so test/val lose the duplicate
        pairs = clean.near_duplicates([cache[r["id"]][3] for r in order])
        dup = {order[b]["id"] for _, b in pairs}
        for i in bad:
            removed[cache[i][1].split(":")[0]] += 1
        removed["near_duplicate"] += len(dup)
        log(f"round {rnd}: {len(bad)} unreadable/small, {len(dup)} near-duplicates among {len(allrec)} chosen")
        if not bad and not dup:
            return splits, pools, cache, dict(removed), rnd
        exclude |= bad | dup
    raise RuntimeError("split did not become clean; inspect the excluded images")


def class_table(splits, pools, cats, names):
    rows = []
    X = {k: label_matrix(v, cats) for k, v in splits.items()}
    P = {k: label_matrix(v, cats) for k, v in pools.items()}
    for j, c in enumerate(cats):
        rows.append({"category_id": c, "name": names.get(c, str(c)),
                     **{f"{k}_images": int(X[k][:, j].sum()) for k in splits},
                     **{f"{k}_pool_share": float(P[k][:, j].mean()) for k in splits},
                     **{f"{k}_share": float(X[k][:, j].mean()) for k in splits}})
    return rows


def balance_summary(splits, pools, cats, rng_seed=0):
    out = {}
    for k in splits:
        X = label_matrix(splits[k], cats)
        f = X.sum(0)
        h, hn = entropy(X)
        out[k] = {"images": len(X), "labels_per_image": float(X.sum(1).mean()), "gini": gini(X),
                  "entropy_norm": hn, "min_class_images": int(f.min()), "median_class_images": float(np.median(f)),
                  "max_class_images": int(f.max())}
        Xp = label_matrix(pools[k], cats)
        out[k]["max_share_gap_pp"] = float(100 * np.abs(X.mean(0) - Xp.mean(0)).max())
    rnd = np.random.default_rng(rng_seed)
    Xp = label_matrix(pools["train"], cats)
    pick_rows = rnd.choice(len(Xp), len(splits["train"]), replace=False)
    Xr = Xp[pick_rows]
    out["train_random_reference"] = {"gini": gini(Xr), "entropy_norm": entropy(Xr)[1],
                                     "min_class_images": int(Xr.sum(0).min()),
                                     "max_class_images": int(Xr.sum(0).max())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/task10k/candidates.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--annotations", default="data/annotations/instances_val2017.json")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--target", type=int, default=150, help="images per class the balancer aims for")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="data/task10k")
    p.add_argument("--report", default="results/task10k/split_report.json")
    args = p.parse_args()

    with open(args.candidates) as f:
        cands = list(json.load(f).values())
    cands.sort(key=lambda r: r["id"])
    # a smaller split (gate runs) needs a proportionally smaller per-class target
    target = max(2, round(args.target * args.n_train / 5000))
    splits, pools, cache, removed, rounds = build(cands, args.n_train, args.n_test, args.n_val,
                                                  args.data_root, args.seed, target)
    assert_disjoint_and_valid(splits)

    cats = sorted({c for r in cands for c in r["labels"]})
    with open(args.annotations) as f:
        names = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, recs in splits.items():
        rows = [{"id": r["id"], "path": f"{args.data_root}/{r['dir']}/{r['file']}", "dir": r["dir"],
                 "karpathy": r["karpathy"], "labels": r["labels"], "captions": r["captions"],
                 "image_mode": cache[r["id"]][2]} for r in recs]
        with open(out_dir / f"{k}.json", "w") as f:
            json.dump(rows, f)

    report = {"sizes": {k: len(v) for k, v in splits.items()}, "target_per_class": target, "rounds": rounds,
              "removed_during_selection": removed, "balance": balance_summary(splits, pools, cats),
              "classes": class_table(splits, pools, cats, names),
              "image_modes": dict(Counter(cache[r["id"]][2] for k in splits for r in splits[k]))}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("sizes", "rounds", "removed_during_selection", "image_modes")}))
    for k, v in report["balance"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
