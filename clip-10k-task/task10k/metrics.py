"""Metrics for the 10K task, computed from embeddings so they are testable without a model.

Retrieval (every image has exactly 5 captions):
    R@1/5/10 both directions, median and mean rank, rsum, CLIPScore of the matched pairs,
    95% bootstrap intervals over queries.
Zero-shot multi-label classification of the 80 COCO classes from text prompts:
    per-class average precision, mAP, micro/macro F1 at per-class thresholds chosen on val.
"""
import numpy as np
import torch
import torch.nn.functional as F

KS = (1, 5, 10)
PROMPT = "a photo of a {}"


@torch.no_grad()
def ranks(img, txt, caps_per_image=5, chunk=512):
    """Rank of the correct item for every query, plus cosine of every matched pair.

    img (N, D) and txt (5N, D) are L2-normalised, with the captions of image i at rows 5i..5i+4.
    rank = number of candidates scoring strictly higher than the best correct one (0 = top).
    """
    n_img, n_txt = img.shape[0], txt.shape[0]
    assert n_txt == caps_per_image * n_img, "expected exactly 5 captions per image, in image order"
    txt2img = torch.arange(n_txt, device=img.device) // caps_per_image
    rank_i2t = torch.empty(n_img, dtype=torch.long, device=img.device)
    for s in range(0, n_img, chunk):
        sims = img[s:s + chunk] @ txt.T
        cols = (torch.arange(s, min(s + chunk, n_img), device=img.device)[:, None] * caps_per_image
                + torch.arange(caps_per_image, device=img.device)[None, :])
        best = sims.gather(1, cols).max(dim=1).values
        rank_i2t[s:s + chunk] = (sims > best[:, None]).sum(dim=1)
    gt_cos = torch.empty(n_txt, device=img.device)
    rank_t2i = torch.empty(n_txt, dtype=torch.long, device=img.device)
    for s in range(0, n_txt, chunk):
        sims = txt[s:s + chunk] @ img.T
        # the matched score comes from the same matmul as the competitors, so rounding cannot
        # make an item outrank itself
        gt = sims.gather(1, txt2img[s:s + chunk, None])
        gt_cos[s:s + chunk] = gt[:, 0]
        rank_t2i[s:s + chunk] = (sims > gt).sum(dim=1)
    return rank_i2t.cpu().numpy(), rank_t2i.cpu().numpy(), gt_cos.cpu().numpy()


def _direction(rank):
    out = {f"R@{k}": 100.0 * float((rank < k).mean()) for k in KS}
    out["medr"] = float(np.median(rank) + 1)      # 1-based, as usually reported
    out["meanr"] = float(rank.mean() + 1)
    return out


def retrieval_summary(rank_i2t, rank_t2i, gt_cos, n_boot=1000, seed=0):
    i2t, t2i = _direction(rank_i2t), _direction(rank_t2i)
    point = {"i2t": i2t, "t2i": t2i,
             "rsum": sum(i2t[f"R@{k}"] + t2i[f"R@{k}"] for k in KS),
             "clipscore": float(2.5 * np.clip(gt_cos, 0, None).mean())}
    rng = np.random.default_rng(seed)
    boots = {"rsum": []}
    for d in ("i2t", "t2i"):
        for k in KS:
            boots[f"{d}_R@{k}"] = []
    for _ in range(n_boot):
        a = rank_i2t[rng.integers(0, len(rank_i2t), len(rank_i2t))]
        b = rank_t2i[rng.integers(0, len(rank_t2i), len(rank_t2i))]
        total = 0.0
        for d, r in (("i2t", a), ("t2i", b)):
            for k in KS:
                v = 100.0 * float((r < k).mean())
                boots[f"{d}_R@{k}"].append(v)
                total += v
        boots["rsum"].append(total)
    point["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return point


def class_scores(img, class_emb):
    """(N, C) cosine of every image to every class prompt."""
    return (img @ class_emb.T).float().cpu().numpy()


def average_precision(y_true, score):
    """Area under the precision-recall curve as the mean of precision at each positive."""
    order = np.argsort(-score, kind="stable")
    hit = y_true[order].astype(bool)
    if not hit.any():
        return float("nan")
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(precision[hit].mean())


def best_f1_thresholds(Y, S):
    """Per-class threshold maximising F1 on (Y, S); returns (C,)."""
    th = np.zeros(S.shape[1])
    for c in range(S.shape[1]):
        order = np.argsort(-S[:, c], kind="stable")
        hit = Y[order, c].astype(bool)
        tp = np.cumsum(hit)
        k = np.arange(1, len(hit) + 1)
        f1 = 2 * tp / (k + hit.sum())
        j = int(np.argmax(f1))
        th[c] = S[order[j], c]
    return th


def classification_summary(Y, S, thresholds):
    """Y (N, C) bool ground truth, S (N, C) scores. Thresholded F1 uses the given per-class thresholds."""
    ap = np.array([average_precision(Y[:, c], S[:, c]) for c in range(S.shape[1])])
    pred = S >= thresholds[None, :]
    tp = (pred & Y).sum(0).astype(float)
    fp = (pred & ~Y).sum(0).astype(float)
    fn = (~pred & Y).sum(0).astype(float)
    f1_c = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))
    prior = Y.mean(0)
    return {"mAP": float(np.nanmean(ap)), "mean_class_prior": float(np.nanmean(prior)),
            "f1_micro": micro, "f1_macro": float(f1_c.mean()),
            "per_class_ap": [None if np.isnan(a) else float(a) for a in ap],
            "per_class_f1": f1_c.tolist(), "n_positive": Y.sum(0).astype(int).tolist()}


def label_matrix(items, cats):
    col = {c: j for j, c in enumerate(cats)}
    Y = np.zeros((len(items), len(cats)), dtype=bool)
    for i, it in enumerate(items):
        Y[i, [col[c] for c in it["labels"]]] = True
    return Y


@torch.no_grad()
def embed(model, processor, items, device, batch_size=128, workers=8):
    """L2-normalised image embeddings (N, D) and caption embeddings (5N, D), captions in image order."""
    from evaluation.common import ImageDataset

    def to_pixels(img):
        return processor.image_processor(img, return_tensors="pt")["pixel_values"][0]

    loader = torch.utils.data.DataLoader(ImageDataset(items, to_pixels), batch_size=batch_size,
                                         num_workers=workers, shuffle=False)
    img = torch.cat([F.normalize(model.get_image_features(pixel_values=px.to(device)).float(), dim=-1)
                     for px, _ in loader])
    texts = [c for it in items for c in it["captions"]]
    txt = []
    for s in range(0, len(texts), 512):
        t = processor.tokenizer(texts[s:s + 512], padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(device)
        txt.append(F.normalize(model.get_text_features(**t).float(), dim=-1))
    return img, torch.cat(txt)


@torch.no_grad()
def class_embeddings(model, processor, names, device, template=PROMPT):
    t = processor.tokenizer([template.format(n) for n in names], padding=True, return_tensors="pt").to(device)
    return F.normalize(model.get_text_features(**t).float(), dim=-1)


def evaluate(model, processor, splits, class_names, cats, device, n_boot=1000, return_embeddings=False):
    """splits: {'train': items, 'val': items, 'test': items}. Returns one result dict.

    Classification thresholds are picked on val and applied to every split (val itself is therefore
    slightly optimistic for F1; AP needs no threshold)."""
    model.eval()
    emb = {k: embed(model, processor, v, device) for k, v in splits.items()}
    cls = class_embeddings(model, processor, class_names, device)
    Y = {k: label_matrix(v, cats) for k, v in splits.items()}
    S = {k: class_scores(emb[k][0], cls) for k in splits}
    thresholds = best_f1_thresholds(Y["val"], S["val"])

    out = {"retrieval": {}, "classification": {}, "n_images": {k: len(v) for k, v in splits.items()}}
    for k in splits:
        r = ranks(*emb[k])
        out["retrieval"][k] = retrieval_summary(*r, n_boot=n_boot)
        out["classification"][k] = classification_summary(Y[k], S[k], thresholds)
    all_img = torch.cat([emb[k][0] for k in splits])
    all_txt = torch.cat([emb[k][1] for k in splits])
    out["retrieval"]["all10k"] = retrieval_summary(*ranks(all_img, all_txt), n_boot=n_boot)
    out["classification"]["all10k"] = classification_summary(
        np.concatenate([Y[k] for k in splits]), np.concatenate([S[k] for k in splits]), thresholds)
    out["thresholds"] = thresholds.tolist()
    out.update(_extended(model, emb, Y, S, thresholds, n_boot))
    return (out, emb) if return_embeddings else out


def _extended(model, emb, Y, S, thresholds, n_boot):
    """MRR / nDCG / mAP / recall-all, embedding diagnostics, classification extras, 1K and 5K protocols.
    Added next to the original fields, which are left exactly as they were."""
    from task10k import metrics_ext as X
    all_emb = (torch.cat([emb[k][0] for k in emb]), torch.cat([emb[k][1] for k in emb]))
    views = {**emb, "all10k": all_emb}
    Yall, Sall = np.concatenate([Y[k] for k in emb]), np.concatenate([S[k] for k in emb])
    # Classes are bucketed by natural frequency (val + test, which keep COCO's mix); train is flat by design.
    counts = Y["val"].sum(0) + Y["test"].sum(0) if {"val", "test"} <= set(Y) else Yall.sum(0)
    buckets = X.class_buckets(counts)
    ext = {"retrieval_ext": {k: X.extended_retrieval(*v, n_boot=n_boot) for k, v in views.items()},
           "diagnostics": {k: X.embedding_diagnostics(*v) for k, v in views.items()},
           "classification_ext": {**{k: X.classification_extras(Y[k], S[k], thresholds, buckets) for k in emb},
                                  "all10k": X.classification_extras(Yall, Sall, thresholds, buckets)},
           "class_buckets": buckets,
           "logit_scale": float(model.logit_scale.exp())}
    if "test" in emb:
        ext["protocols"] = X.protocols(emb)
    return ext
