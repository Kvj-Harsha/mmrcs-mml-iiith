"""Tables and figures for the final notebook. Everything is read from the saved result JSONs, so
the notebook shows exactly the numbers the gated runs produced."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/task10k")
SPLITS = ("train", "val", "test", "all10k")


def load(results_dir=R):
    results_dir = Path(results_dir)
    out = {"cleaning": json.load(open(results_dir / "cleaning_report.json")),
           "split": json.load(open(results_dir / "split_report.json")),
           "zero": json.load(open(results_dir / "zeroshot.json")),
           "sweep": [json.load(open(f)) for f in sorted(glob.glob(str(results_dir / "sweep" / "*.json")))],
           "finals": [json.load(open(f)) for f in sorted(glob.glob(str(results_dir / "final" / "s*.json")))]}
    out["sweep"].sort(key=lambda r: r["lr"])
    return out


def flat(ev, split):
    """One split's metrics as {name: value}, everything in percent except ranks and CLIPScore."""
    r, c = ev["retrieval"][split], ev["classification"][split]
    out = {}
    for d, label in (("i2t", "image to text"), ("t2i", "text to image")):
        for k in (1, 5, 10):
            out[f"{label} R@{k}"] = r[d][f"R@{k}"]
    out["rsum"] = r["rsum"]
    for d, label in (("i2t", "image to text"), ("t2i", "text to image")):
        out[f"{label} median rank"] = r[d]["medr"]
        out[f"{label} mean rank"] = r[d]["meanr"]
    out["CLIPScore"] = r["clipscore"]
    out["mAP (80 classes)"] = 100 * c["mAP"]
    out["F1 micro"] = 100 * c["f1_micro"]
    out["F1 macro"] = 100 * c["f1_macro"]
    return out


def ci_of(ev, split, name):
    """95% bootstrap interval of a retrieval metric, or None if it has none."""
    key = {"image to text R@1": "i2t_R@1", "image to text R@5": "i2t_R@5", "image to text R@10": "i2t_R@10",
           "text to image R@1": "t2i_R@1", "text to image R@5": "t2i_R@5", "text to image R@10": "t2i_R@10",
           "rsum": "rsum"}.get(name)
    return ev["retrieval"][split]["ci95"][key] if key else None


def seed_stats(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def metric_table(zero, finals, split):
    """Zero-shot vs fine-tuned (mean +- std over seeds) for one split."""
    z = flat(zero, split)
    ft = [flat(f["eval"], split) for f in finals]
    rows = []
    for name, zv in z.items():
        mean, std = seed_stats([f[name] for f in ft])
        ci = ci_of(zero, split, name)
        rows.append({"metric": name,
                     "zero-shot": f"{zv:.1f}" if "rank" not in name and name != "CLIPScore" else f"{zv:.2f}",
                     "zero-shot 95% CI": f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "",
                     f"fine-tuned (mean ± std, {len(ft)} seeds)":
                         (f"{mean:.1f} ± {std:.1f}" if "rank" not in name and name != "CLIPScore"
                          else f"{mean:.2f} ± {std:.2f}"),
                     "change": f"{mean - zv:+.1f}" if name != "CLIPScore" else f"{mean - zv:+.3f}"})
    return pd.DataFrame(rows).set_index("metric")


def summary_by_split(zero, finals, names=("rsum", "image to text R@1", "text to image R@1", "mAP (80 classes)")):
    """One row per split; columns are (metric, zero-shot | fine-tuned mean +- std)."""
    rows = {}
    for split in SPLITS:
        z = flat(zero, split)
        ft = [flat(f["eval"], split) for f in finals]
        row = {}
        for n in names:
            mean, std = seed_stats([f[n] for f in ft])
            row[(n, "zero-shot")] = f"{z[n]:.1f}"
            row[(n, "fine-tuned")] = f"{mean:.1f} ± {std:.1f}"
        rows[split] = row
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "split"
    return df


def sweep_table(sweep):
    return pd.DataFrame([{"learning rate": f"{s['lr']:g}", "best epoch": s["best_epoch"], "epochs run": s["epochs_run"],
                          "best val rsum": round(s["best_val_rsum"], 1),
                          "final train loss": round(s["curve"][-1]["train_loss"], 3)} for s in sweep]).set_index("learning rate")


def cleaning_table(cleaning):
    m = cleaning["metadata_rules"]
    labels = {"train2017_images": "train2017 images examined", "val2017_images": "val2017 images examined",
              "no_annotation": "removed: no object annotation", "no_label_after_area_rule":
              "removed: only objects under 0.5% of the image", "too_small": "removed: shorter side under 128 px",
              "fewer_than_5_captions": "removed: fewer than 5 usable captions",
              "caption_duplicate": "captions dropped: duplicates within an image",
              "caption_unusable": "captions dropped: <3 or >50 words, no letters",
              "candidates": "images kept as candidates"}
    return pd.DataFrame([{"rule": labels[k], "count": m.get(k, 0)} for k in labels]).set_index("rule")


def balance_table(split_report):
    b = split_report["balance"]
    rows = []
    for k, name in (("train", "train (balanced)"), ("train_random_reference", "random 5K from the same pool"),
                    ("val", "val (stratified)"), ("test", "test (stratified)")):
        v = b[k]
        rows.append({"set": name, "Gini of class counts": round(v["gini"], 3),
                     "normalised entropy": round(v["entropy_norm"], 3),
                     "fewest images in a class": v["min_class_images"],
                     "most images in a class": v["max_class_images"]})
    return pd.DataFrame(rows).set_index("set")


def per_class_table(zero, finals, split="test", min_pos=20):
    names = zero["class_names"]
    z = zero["classification"][split]
    ap_z = np.array([np.nan if a is None else a for a in z["per_class_ap"]])
    ap_f = np.mean([[np.nan if a is None else a for a in f["eval"]["classification"][split]["per_class_ap"]]
                    for f in finals], axis=0)
    df = pd.DataFrame({"class": names, "images in split": z["n_positive"], "AP zero-shot": 100 * ap_z,
                       "AP fine-tuned": 100 * ap_f})
    df["change"] = df["AP fine-tuned"] - df["AP zero-shot"]
    df["reliable"] = df["images in split"] >= min_pos
    return df.set_index("class")


# ---------------------------------------------------------------- extended metrics

EXT_RETRIEVAL_ROWS = (
    ("image to text", "i2t", ("MRR", "mAP", "nDCG@5", "nDCG@10", "nDCG@50", "Recall-all@5", "Recall-all@10",
                              "Recall-all@50", "Precision@10")),
    ("text to image", "t2i", ("MRR", "nDCG@5", "nDCG@10", "nDCG@50")),
)
CLS_EXTRA_ROWS = (("mean_auroc", "mean AUROC"), ("lrap", "label-ranking AP"), ("top1_hit", "top-1 class is correct"),
                  ("label_recall@3", "label recall@3"), ("label_recall@5", "label recall@5"),
                  ("sample_f1", "sample-wise F1"), ("micro_precision", "micro precision"),
                  ("micro_recall", "micro recall"), ("hamming_loss", "Hamming loss"),
                  ("mAP_head", "mAP, head classes"), ("mAP_mid", "mAP, mid classes"), ("mAP_tail", "mAP, tail classes"),
                  ("auroc_head", "AUROC, head classes"), ("auroc_mid", "AUROC, mid classes"),
                  ("auroc_tail", "AUROC, tail classes"))
DIAG_ROWS = (("cos_matched", "cosine, matched pairs"), ("cos_unmatched", "cosine, unmatched pairs"),
             ("cos_margin", "margin (matched - unmatched)"), ("modality_gap", "modality gap"),
             ("alignment", "alignment (lower is tighter)"), ("uniformity_image", "uniformity, images (lower is more spread)"),
             ("uniformity_text", "uniformity, captions"), ("logit_scale", "logit scale (1 / temperature)"))
PROTOCOL_ROWS = (("i2t_R@1", "image to text R@1"), ("t2i_R@1", "text to image R@1"), ("rsum", "rsum"),
                 ("i2t_MRR", "image to text MRR"), ("t2i_MRR", "text to image MRR"),
                 ("i2t_nDCG@10", "image to text nDCG@10"), ("t2i_nDCG@10", "text to image nDCG@10"))
GALLERIES = (("test_1k_folds", "1K gallery (test, mean of 3 folds)"), ("test_full", "3K gallery (full test)"),
             ("heldout_5k", "5K gallery (val + test)"))


def ext_retrieval_table(zero, finals, split="test"):
    """MRR, nDCG@k, mAP, recall of all five captions: zero-shot with a bootstrap interval vs fine-tuned."""
    rows = []
    for label, d, names in EXT_RETRIEVAL_ROWS:
        for n in names:
            z = zero["retrieval_ext"][split][d][n]
            mean, std = seed_stats([f["eval"]["retrieval_ext"][split][d][n] for f in finals])
            lo, hi = zero["retrieval_ext"][split]["ci95"][f"{d}_{n}"]
            rows.append({"metric": f"{label} {n}" + (" (= R-Precision)" if n == "Recall-all@5" else ""),
                         "zero-shot": f"{z:.1f}", "zero-shot 95% CI": f"[{lo:.1f}, {hi:.1f}]",
                         f"fine-tuned (mean ± std, {len(finals)} seeds)": f"{mean:.1f} ± {std:.1f}",
                         "change": f"{mean - z:+.1f}"})
    return pd.DataFrame(rows).set_index("metric")


def ext_summary_by_split(zero, finals, items=(("i2t", "MRR"), ("i2t", "nDCG@10"), ("t2i", "MRR"), ("t2i", "nDCG@10"))):
    rows = {}
    for split in SPLITS:
        row = {}
        for d, n in items:
            mean, std = seed_stats([f["eval"]["retrieval_ext"][split][d][n] for f in finals])
            label = f"{'image to text' if d == 'i2t' else 'text to image'} {n}"
            row[(label, "zero-shot")] = f"{zero['retrieval_ext'][split][d][n]:.1f}"
            row[(label, "fine-tuned")] = f"{mean:.1f} ± {std:.1f}"
        rows[split] = row
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "split"
    return df


def protocol_table(zero, finals):
    """Same headline metrics on a 1K, 3K and 5K gallery. Fine-tuned: mean ± std over seeds."""
    cols = {}
    for key, title in GALLERIES:
        zp = zero["protocols"][key]
        for stage in ("zero-shot", "fine-tuned"):
            col = []
            for m, _ in PROTOCOL_ROWS:
                get = (lambda p: p[m]["mean"]) if key == "test_1k_folds" else (lambda p: p[m])
                if stage == "zero-shot":
                    col.append(f"{get(zp):.1f}")
                else:
                    mean, std = seed_stats([get(f["eval"]["protocols"][key]) for f in finals])
                    col.append(f"{mean:.1f} ± {std:.1f}")
            cols[(title, stage)] = col
    df = pd.DataFrame(cols, index=[label for _, label in PROTOCOL_ROWS])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "metric"
    return df


def _compare_rows(zero_vals, finals_vals, rows, scale, digits):
    out = []
    for key, label in rows:
        z = zero_vals[key]
        ft = [v[key] for v in finals_vals]
        if z is None or any(v is None for v in ft):
            continue
        mean, std = seed_stats(ft)
        out.append({"metric": label, "zero-shot": f"{scale * z:.{digits}f}",
                    f"fine-tuned (mean ± std, {len(ft)} seeds)": f"{scale * mean:.{digits}f} ± {scale * std:.{digits}f}",
                    "change": f"{scale * (mean - z):+.{digits}f}"})
    return pd.DataFrame(out).set_index("metric")


def cls_extras_table(zero, finals, split="test"):
    return _compare_rows(zero["classification_ext"][split], [f["eval"]["classification_ext"][split] for f in finals],
                         CLS_EXTRA_ROWS, 100.0, 1)


def diagnostics_table(zero, finals, split="test"):
    z = dict(zero["diagnostics"][split], logit_scale=zero["logit_scale"])
    ft = [dict(f["eval"]["diagnostics"][split], logit_scale=f["eval"]["logit_scale"]) for f in finals]
    return _compare_rows(z, ft, DIAG_ROWS, 1.0, 3)


def worst_classes_table(zero, finals, split="test", n=10, min_pos=20):
    t = per_class_table(zero, finals, split, min_pos)
    return t[t["reliable"]].sort_values("AP fine-tuned").head(n)[["images in split", "AP zero-shot", "AP fine-tuned", "change"]].round(1)


# ---------------------------------------------------------------- figures

def plot_ndcg(zero, finals, split="test"):
    import matplotlib.pyplot as plt
    ks = (5, 10, 50)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (d, title) in zip(axes, (("i2t", "image to text"), ("t2i", "text to image"))):
        z = [zero["retrieval_ext"][split][d][f"nDCG@{k}"] for k in ks]
        stats = [seed_stats([f["eval"]["retrieval_ext"][split][d][f"nDCG@{k}"] for f in finals]) for k in ks]
        x = np.arange(len(ks))
        ax.bar(x - 0.2, z, 0.4, color="#9aa7b1", label="zero-shot")
        ax.bar(x + 0.2, [m for m, _ in stats], 0.4, yerr=[s for _, s in stats], color="#1f5f8b", label="fine-tuned")
        for xi, (zv, (m, _)) in enumerate(zip(z, stats)):
            ax.text(xi - 0.2, zv + 0.8, f"{zv:.0f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, m + 0.8, f"{m:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"nDCG@{k}" for k in ks])
        ax.set_title(title)
    axes[0].set_ylabel(f"nDCG (%), {split} split")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_class_counts(split_report):
    import matplotlib.pyplot as plt
    rows = sorted(split_report["classes"], key=lambda c: -c["train_images"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for k, color in (("train", "#1f5f8b"), ("test", "#d98a00"), ("val", "#5a9e6f")):
        ax.plot(x, [r[f"{k}_images"] for r in rows], marker="o", ms=3, lw=1.2, color=color, label=k)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("images containing the class (log)")
    ax.set_title("Images per class: train is flat by design, test and val follow COCO's natural long tail")
    ax.legend()
    fig.tight_layout()
    return fig


def contact_sheet(split_dir="data/task10k", per_split=12, seed=0):
    import random
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(3, per_split // 2, figsize=(14, 6.6))
    for r, k in enumerate(("train", "test", "val")):
        rows = random.Random(seed).sample(json.load(open(Path(split_dir) / f"{k}.json")), per_split)
        for c in range(per_split // 2):
            ax = axes[r][c]
            it = rows[c]
            ax.imshow(Image.open(it["path"]).convert("RGB"))
            ax.set_title(it["captions"][0][:40], fontsize=6)
            ax.axis("off")
        axes[r][0].text(-0.05, 0.5, k, transform=axes[r][0].transAxes, rotation=90, ha="right", va="center", fontsize=11)
    fig.suptitle("Random images with their first caption: train (top), test, val (bottom)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_per_class_ap(zero, split="test"):
    import matplotlib.pyplot as plt
    names = zero["class_names"]
    ap = np.array([np.nan if a is None else a for a in zero["classification"][split]["per_class_ap"]]) * 100
    n = np.array(zero["classification"][split]["n_positive"])
    order = np.argsort(ap)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    colors = ["#b3382c" if n[i] < 20 else "#1f5f8b" for i in order]
    ax.bar(range(len(order)), ap[order], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
    ax.axhline(100 * zero["classification"][split]["mAP"], color="#444", ls="--", lw=1)
    ax.set_ylabel("average precision (%)")
    ax.set_title(f"Zero-shot average precision per class, {split} split (red: fewer than 20 images, noisy; dashed: mAP)")
    fig.tight_layout()
    return fig


def plot_sweep(sweep, zero_val_rsum):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for s in sweep:
        ax.plot([c["epoch"] for c in s["curve"]], [c["val_rsum"] for c in s["curve"]], marker="o", label=f"lr {s['lr']:g}")
    ax.axhline(zero_val_rsum, color="#444", ls="--", lw=1, label="zero-shot")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val rsum")
    ax.set_title("Validation rsum by epoch for each learning rate")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_per_class_change(table, top=12):
    import matplotlib.pyplot as plt
    t = table[table["reliable"]].sort_values("change")
    pick = pd.concat([t.head(top // 2), t.tail(top)])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(pick.index, pick["change"], color=["#b3382c" if v < 0 else "#2d7a4f" for v in pick["change"]])
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("change in average precision (points), classes with at least 20 test images")
    ax.set_title("Largest per-class changes after fine-tuning (test)")
    fig.tight_layout()
    return fig
