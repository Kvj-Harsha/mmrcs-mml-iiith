"""Extra metrics for the 10K task, all computed from embeddings (no model needed).

Retrieval, binary exact-match relevance (image i owns captions 5i..5i+4):
    image -> text has five correct captions per query, text -> image has one correct image.
    MRR, nDCG@k, mAP, recall of all five captions, R-Precision, Precision@10 (all scaled to 0-100).
Embedding diagnostics: matched vs unmatched cosine, modality gap, alignment, uniformity.
Classification extras: AUROC, label-ranking AP, label recall@k, sample F1, Hamming loss, class buckets.
Protocols: three disjoint 1K galleries of the test split, and the held-out 5K (val + test).

Rank convention is the same as metrics.ranks: rank = number of candidates scoring strictly higher.
"""
import numpy as np
import torch

CAPS = 5
NDCG_KS = (5, 10, 50)
RECALL_ALL_KS = (5, 10, 50)
BUCKETS = ("head", "mid", "tail")


@torch.no_grad()
def positive_ranks(img, txt, caps=CAPS, chunk=512):
    """(N, caps) 0-based rank of each correct caption among all captions, sorted ascending per image."""
    n = img.shape[0]
    assert txt.shape[0] == caps * n
    out = torch.empty(n, caps, dtype=torch.long)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = img[s:e] @ txt.T
        cols = torch.arange(s, e, device=img.device)[:, None] * caps + torch.arange(caps, device=img.device)[None, :]
        pos = sims.gather(1, cols)
        for j in range(caps):
            out[s:e, j] = (sims > pos[:, j:j + 1]).sum(dim=1).cpu()
    return out.sort(dim=1).values.numpy()


def i2t_query_metrics(R):
    """Per-image metrics from the sorted 0-based ranks R (N, caps) of the correct captions."""
    c = R.shape[1]
    j = np.arange(1, c + 1)
    # 1-based positions; the max keeps two tied correct captions from sharing a position
    pos = np.maximum(R + 1, j[None, :]).astype(float)
    out = {"MRR": 100 / pos[:, 0], "mAP": 100 * (j[None, :] / pos).mean(1)}
    for k in NDCG_KS:
        ideal = (1 / np.log2(np.arange(2, min(c, k) + 2))).sum()
        out[f"nDCG@{k}"] = 100 * np.where(pos <= k, 1 / np.log2(pos + 1), 0).sum(1) / ideal
    for k in RECALL_ALL_KS:
        out[f"Recall-all@{k}"] = 100 * (pos <= k).sum(1) / c
    out["R-Precision"] = out["Recall-all@5"] if c == 5 else 100 * (pos <= c).sum(1) / c
    out["Precision@10"] = 100 * (pos <= 10).sum(1) / 10
    for k in (1, 5, 10):
        out[f"R@{k}"] = 100 * (pos[:, 0] <= k).astype(float)
    return out


def t2i_query_metrics(rank):
    """Per-caption metrics from the 0-based rank of its single correct image."""
    pos = rank.astype(float) + 1
    out = {"MRR": 100 / pos, "mAP": 100 / pos}      # one relevant item, so AP is the reciprocal rank
    for k in NDCG_KS:
        out[f"nDCG@{k}"] = 100 * np.where(pos <= k, 1 / np.log2(pos + 1), 0)
    for k in (1, 5, 10):
        out[f"R@{k}"] = 100 * (pos <= k).astype(float)
    return out


def query_metrics(img, txt):
    from task10k.metrics import ranks
    _, rank_t2i, _ = ranks(img, txt)
    return i2t_query_metrics(positive_ranks(img, txt)), t2i_query_metrics(rank_t2i)


def _bootstrap(per_query, n_boot, seed):
    names = list(per_query)
    M = np.stack([per_query[k] for k in names], axis=1)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        boots[b] = M[rng.integers(0, len(M), len(M))].mean(0)
    return {k: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]
            for i, k in enumerate(names)}


def extended_retrieval(img, txt, n_boot=1000, seed=0):
    """{'i2t': {metric: mean}, 't2i': {...}, 'ci95': {'i2t_MRR': [lo, hi], ...}} for one gallery."""
    qi, qt = query_metrics(img, txt)
    out = {"i2t": {k: float(v.mean()) for k, v in qi.items()},
           "t2i": {k: float(v.mean()) for k, v in qt.items()}}
    ci = {}
    for d, q in (("i2t", qi), ("t2i", qt)):
        for k, v in _bootstrap(q, n_boot, seed).items():
            ci[f"{d}_{k}"] = v
    out["ci95"] = ci
    return out


def embedding_diagnostics(img, txt, caps=CAPS, n_unif=3000, seed=0):
    img = img.double().cpu().numpy()
    txt = txt.double().cpu().numpy()
    n, m = len(img), len(txt)
    owner = np.arange(m) // caps
    matched = (img[owner] * txt).sum(1)
    unmatched = (img.sum(0) @ txt.sum(0) - matched.sum()) / (n * m - m)     # exact mean over all other pairs
    rng = np.random.default_rng(seed)

    def uniformity(x):
        x = x[rng.choice(len(x), min(len(x), n_unif), replace=False)]
        d2 = np.clip(2 - 2 * (x @ x.T), 0, None)[np.triu_indices(len(x), 1)]
        return float(np.log(np.exp(-2 * d2).mean()))

    return {"cos_matched": float(matched.mean()), "cos_unmatched": float(unmatched),
            "cos_margin": float(matched.mean() - unmatched),
            "modality_gap": float(np.linalg.norm(img.mean(0) - txt.mean(0))),
            "alignment": float((2 - 2 * matched).mean()),
            "uniformity_image": uniformity(img), "uniformity_text": uniformity(txt)}


def _mean_dict(q):
    return {k: float(v.mean()) for k, v in q.items()}


def compact(img, txt):
    """Headline numbers for one gallery, used by the protocol table."""
    qi, qt = query_metrics(img, txt)
    i, t = _mean_dict(qi), _mean_dict(qt)
    out = {f"i2t_{k}": i[k] for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10", "mAP")}
    out.update({f"t2i_{k}": t[k] for k in ("R@1", "R@5", "R@10", "MRR", "nDCG@10")})
    out["rsum"] = sum(out[f"{d}_R@{k}"] for d in ("i2t", "t2i") for k in (1, 5, 10))
    return out


def _subset(img, txt, idx, caps=CAPS):
    idx = np.asarray(idx)
    cols = (idx[:, None] * caps + np.arange(caps)[None, :]).ravel()
    return img[idx], txt[cols]


def protocols(emb, seed=0):
    """emb: {split: (img, txt)}. Test split as 3 disjoint 1K galleries (mean and std over folds), the
    whole 3K test gallery, and the held-out 5K gallery made of val + test."""
    img, txt = emb["test"]
    n = len(img)
    perm = np.random.default_rng(seed).permutation(n)
    k = n // 3
    folds = [compact(*_subset(img, txt, perm[i * k:(i + 1) * k])) for i in range(3)]
    out = {"test_1k_folds": {m: {"mean": float(np.mean([f[m] for f in folds])),
                                  "std": float(np.std([f[m] for f in folds], ddof=1))} for m in folds[0]},
           "fold_size": k, "test_full": compact(img, txt)}
    if "val" in emb:
        out["heldout_5k"] = compact(torch.cat([emb["val"][0], img]), torch.cat([emb["val"][1], txt]))
    return out


# ---------------------------------------------------------------- classification

def auroc_per_class(Y, S):
    from scipy.stats import rankdata
    out = np.full(S.shape[1], np.nan)
    for c in range(S.shape[1]):
        npos = int(Y[:, c].sum())
        nneg = len(Y) - npos
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(S[:, c])
        out[c] = (r[Y[:, c]].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return out


def label_ranking_ap(Y, S):
    """Mean over images of: for each true class, share of true classes ranked at or above it."""
    rank = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
    big = S.shape[1] + 1
    true_rank = np.sort(np.where(Y, rank, big), axis=1)
    valid = true_rank < big
    j = np.arange(1, S.shape[1] + 1)[None, :]
    per_image = (np.where(valid, j / np.where(valid, true_rank, 1), 0)).sum(1) / np.maximum(Y.sum(1), 1)
    return float(per_image[Y.any(1)].mean())


def class_buckets(counts):
    """head / mid / tail thirds of the classes ordered by how many images contain them."""
    order = np.argsort(-np.asarray(counts), kind="stable")
    out = np.empty(len(counts), dtype=object)
    third = int(np.ceil(len(counts) / 3))
    for r, c in enumerate(order):
        out[c] = BUCKETS[min(r // third, 2)]
    return out.tolist()


def classification_extras(Y, S, thresholds, buckets):
    from task10k.metrics import average_precision
    auc = auroc_per_class(Y, S)
    ap = np.array([average_precision(Y[:, c], S[:, c]) for c in range(S.shape[1])])
    order = np.argsort(-S, axis=1)
    rows = np.arange(len(Y))
    n_true = np.maximum(Y.sum(1), 1)
    out = {"mean_auroc": float(np.nanmean(auc)), "per_class_auroc": [None if np.isnan(a) else float(a) for a in auc],
           "lrap": label_ranking_ap(Y, S), "top1_hit": float(Y[rows, order[:, 0]].mean())}
    for k in (3, 5):
        out[f"label_recall@{k}"] = float((Y[rows[:, None], order[:, :k]].sum(1) / n_true)[Y.any(1)].mean())
    pred = S >= thresholds[None, :]
    tp = (pred & Y).sum(1)
    denom = pred.sum(1) + Y.sum(1)
    out["sample_f1"] = float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0).mean())
    out["hamming_loss"] = float((pred != Y).mean())
    out["micro_precision"] = float((pred & Y).sum() / max(pred.sum(), 1))
    out["micro_recall"] = float((pred & Y).sum() / max(Y.sum(), 1))
    b = np.array(buckets)
    for name in BUCKETS:
        m = b == name
        out[f"mAP_{name}"] = float(np.nanmean(ap[m])) if np.isfinite(ap[m]).any() else None
        out[f"auroc_{name}"] = float(np.nanmean(auc[m])) if np.isfinite(auc[m]).any() else None
    return out


def save_embeddings(path, emb):
    """float16 copies of the L2-normalised embeddings, for later metrics without the model."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{kind}_{k}": v.cpu().numpy().astype(np.float16)
                                 for k, (i, t) in emb.items() for kind, v in (("img", i), ("txt", t))})
