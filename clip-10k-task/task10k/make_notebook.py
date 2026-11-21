"""Build notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb.

    python -m task10k.make_notebook
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb

The notebook reads the saved results in results/task10k/ (made by the task10k modules). Setting
RERUN = True in its second cell recomputes everything, which takes about 1.5 hours on one GPU.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb")

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# CLIP ViT-B/32 on a cleaned, class-balanced 10K COCO 2017 split: zero-shot vs fine-tuned

**Task.** Build a 10,000-image split of COCO 2017 (5,000 train, 3,000 test, 2,000 validation), clean it, score a
model zero-shot on all of it, fine-tune on the 5,000 train images, and score again.

**Model.** OpenAI CLIP ViT-B/32. BLIP was the first choice but is not COCO-free: its pretraining included the COCO
train images. CLIP was trained on a private web collection (WIT-400M) with no COCO labels or captions. Nobody outside
OpenAI can prove that collection contains no COCO image, so "COCO-free" here means "not trained on COCO data on purpose".

**Metrics.** CLIP has no captioning head, so the metrics are image-text retrieval (recall, rank, CLIPScore) and
zero-shot classification of the 80 COCO classes from text prompts (average precision, F1). Section 5 adds ranking
metrics (MRR, nDCG, mAP), classification extras, embedding diagnostics, and 1K/3K/5K gallery protocols.

**How to read it.** Sections 1 to 4 follow the order of the work. Every table is read from the saved result files in
`results/task10k/`, so the numbers are the ones the checked runs produced.
""")

code("""
import os, sys, subprocess
root = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(root); sys.path.insert(0, root)
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from task10k import report
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
""")

code("""
# Set to True to recompute everything (about 1.5 h on one GPU). False reads the saved results.
RERUN = False

def run(*args):
    print("$", " ".join(args)); subprocess.run(args, check=True)

if RERUN:
    py = sys.executable
    run(py, "-m", "task10k.clean")
    run(py, "-m", "task10k.split")
    run(py, "-m", "task10k.zeroshot")
    for lr in ("3e-7", "1e-6", "3e-6", "1e-5"):
        run(py, "-m", "task10k.finetune", "--lr", lr, "--out", f"results/task10k/sweep/lr{lr}.json")
    for seed in (0, 1, 2):
        run(py, "-m", "task10k.finetune", "--lr", "1e-6", "--seed", str(seed), "--final",
            "--out", f"results/task10k/final/s{seed}.json")

res = report.load()
zero, finals, sweep = res["zero"], res["finals"], res["sweep"]
print(f"{len(finals)} fine-tuning seeds, {len(sweep)} learning rates in the sweep")
""")

md("""
## 1. Data: cleaning and the three splits

**Cleaning runs in two stages.** The cheap metadata rules run over all 123,287 COCO train2017 and val2017 images.
The slow image checks (full decode, near-duplicate hash) run only on the 10,000 images that were chosen, and a bad
image is replaced by the next candidate.

| Rule | What it does |
|---|---|
| Object labels | An object counts toward an image's label only if it covers at least 0.5% of the image and is not a crowd region. Images left with no label are dropped. |
| Captions | Exactly 5 per image. Whitespace, control characters and unicode forms are normalised; captions under 3 or over 50 words, without letters, or duplicated inside an image are dropped. |
| Size | Images with a side under 128 px are dropped. |
| Decode check | Every chosen image is fully decoded; unreadable files are replaced. Grayscale images are kept and converted to RGB at load. |
| Near-duplicates | A perceptual hash (dHash, Hamming distance of at most 4) over the 10K; the later image of a pair is replaced. No picture appears in two splits. |
""")

code("""
display(report.cleaning_table(res["cleaning"]))
sp = res["split"]
print("chosen images:", sp["sizes"], "| near-duplicates replaced while selecting:",
      sp["removed_during_selection"].get("near_duplicate", 0), "| image modes:", sp["image_modes"])
""")

md("""
**The split.**
- **Train (5,000)**: COCO train2017, **class-balanced** by a greedy selector that always takes the image helping the most
  classes still below 150 images, so every class is represented about equally.
- **Test (3,000)**: COCO train2017, disjoint from train, **stratified** (it keeps the natural class mix). It is drawn
  only from images that sit in the Karpathy val/test splits, which BLIP did not train on, so the same split stays clean
  if BLIP is added later.
- **Validation (2,000)**: the official COCO val2017 images, stratified. Used to pick the learning rate and the epoch.
""")

code("""
display(report.balance_table(sp))
report.plot_class_counts(sp); plt.show()
""")

code("""
rare = [(c["name"], c["test_images"], c["val_images"]) for c in sp["classes"] if c["test_images"] < 15 or c["val_images"] < 10]
print("Classes with few test or val images (their per-class numbers are noisy):")
for name, t, v in rare: print(f"  {name:12s} test {t:3d}  val {v:3d}")
report.contact_sheet(); plt.show()
""")

md("""
## 2. Zero-shot results

CLIP scores every image against every caption. Retrieval is measured per split, with the gallery being that split's own
images and captions (5 per image), and for all 10,000 images together. Intervals are 95% bootstrap intervals over
queries. Classification uses the prompt "a photo of a {class}" for each of the 80 classes; the F1 thresholds are chosen
per class on the validation split and applied to every split, so validation F1 is slightly optimistic.
""")

code("""
display(report.metric_table(zero, [{"eval": zero}], "test")[["zero-shot", "zero-shot 95% CI"]].rename_axis("test split, zero-shot"))
""")

code("""
rows = []
for split in report.SPLITS:
    f = report.flat(zero, split)
    rows.append({"split": split, "images": sum(zero["n_images"].values()) if split == "all10k" else zero["n_images"][split],
                 "image to text R@1": f["image to text R@1"], "R@5": f["image to text R@5"],
                 "text to image R@1": f["text to image R@1"], "t2i R@5": f["text to image R@5"],
                 "rsum": f["rsum"], "CLIPScore": round(f["CLIPScore"], 3), "mAP": f["mAP (80 classes)"],
                 "F1 micro": f["F1 micro"], "F1 macro": f["F1 macro"]})
table = pd.DataFrame(rows).set_index("split").round({"image to text R@1": 1, "R@5": 1, "text to image R@1": 1,
                                                       "t2i R@5": 1, "rsum": 1, "CLIPScore": 3, "mAP": 1,
                                                       "F1 micro": 1, "F1 macro": 1})
display(table)
print("Retrieval recall falls as the gallery grows, so compare splits of equal size only. The train split has the "
      "largest single-split gallery (5,000 images), which lowers its recall; whether its balanced content is also "
      "harder is not isolated here.")
""")

code("""
report.plot_per_class_ap(zero, "test"); plt.show()
""")

md("""
## 3. Fine-tuning

Everything in CLIP is trained (both towers and the temperature) on the 5,000 train images, which give 25,000
image-caption pairs. The loss is the symmetric contrastive loss in which all captions of the same image count as
positives. AdamW with weight decay 0.1, cosine schedule with 10% warm-up, batch 64, bf16, random-crop and flip
augmentation, at most 10 epochs, and the epoch with the best **validation** retrieval rsum is kept.
The learning rate was picked on validation from a sweep; the test split was not scored until the rate was fixed.
Three seeds at the chosen rate give the mean and spread.
""")

code("""
display(report.sweep_table(sweep))
report.plot_sweep(sweep, report.flat(zero, "val")["rsum"]); plt.show()
""")

code("""
print("A fourth rate, 3e-7, was started but stopped by the thermal watchdog at epoch 5 (validation rsum 465.9, "
      "still rising slowly, below the 1e-6 run at the same epoch: 471.1). It is not in the table.")
best = max(sweep, key=lambda s: s["best_val_rsum"])
print(f"Chosen learning rate: {best['lr']:g} (best validation rsum {best['best_val_rsum']:.1f}; zero-shot validation rsum "
      f"{report.flat(zero, 'val')['rsum']:.1f}).")
display(pd.DataFrame([{"seed": f["seed"], "best epoch": f["best_epoch"], "epochs run": f["epochs_run"],
                       "val rsum": round(f["best_val_rsum"], 1), "minutes": round(f["train_seconds"] / 60, 1)} for f in finals]).set_index("seed"))
""")

md("""
## 4. Before and after

The headline comparison is on the **test** split, which was used for nothing but this table. The second table repeats the
main metrics for every split. CLIPScore is not a like-for-like comparison here: fine-tuning changes the scale of the
cosine similarities, so a small change in it says little about quality.
""")

code("""
display(report.metric_table(zero, finals, "test"))
""")

code("""
display(report.summary_by_split(zero, finals))
""")

code("""
pc = report.per_class_table(zero, finals, "test")
report.plot_per_class_change(pc); plt.show()
print("Classes whose average precision dropped (at least 20 test images):")
display(pc[(pc["reliable"]) & (pc["change"] < 0)][["images in split", "AP zero-shot", "AP fine-tuned", "change"]].round(1))
""")

code("""
t0, ft = report.flat(zero, "test"), [report.flat(f["eval"], "test") for f in finals]
m = lambda k: report.seed_stats([f[k] for f in ft])
lines = [
  f"**Test rsum** went from {t0['rsum']:.1f} to {m('rsum')[0]:.1f} ± {m('rsum')[1]:.1f} "
  f"(zero-shot 95% interval {report.ci_of(zero, 'test', 'rsum')[0]:.1f} to {report.ci_of(zero, 'test', 'rsum')[1]:.1f}).",
  f"**Image-to-text R@1** {t0['image to text R@1']:.1f} to {m('image to text R@1')[0]:.1f}; "
  f"**text-to-image R@1** {t0['text to image R@1']:.1f} to {m('text to image R@1')[0]:.1f}.",
  f"**Classification mAP** {t0['mAP (80 classes)']:.1f} to {m('mAP (80 classes)')[0]:.1f} ± {m('mAP (80 classes)')[1]:.1f}.",
  *[f"**{lab} (test)** {zero['retrieval_ext']['test'][d][n]:.1f} to "
    f"{report.seed_stats([f['eval']['retrieval_ext']['test'][d][n] for f in finals])[0]:.1f}."
    for lab, d, n in (("Image-to-text MRR", "i2t", "MRR"), ("Text-to-image MRR", "t2i", "MRR"),
                      ("Image-to-text nDCG@10", "i2t", "nDCG@10"), ("Text-to-image nDCG@10", "t2i", "nDCG@10"))],
  f"**Train split rsum** moved from {report.flat(zero,'train')['rsum']:.1f} to "
  f"{report.seed_stats([report.flat(f['eval'],'train')['rsum'] for f in finals])[0]:.1f}; the model has seen these images, "
  f"so it is the in-sample reference, not a result.",
  f"{int((pc['reliable'] & (pc['change'] < 0)).sum())} of {int(pc['reliable'].sum())} classes with at least 20 test images lost average precision.",
]
display(Markdown("### Key results\\n\\n" + "\\n".join("- " + l for l in lines)))
""")

md("""
## 5. More metrics: ranking quality, classification, embeddings, gallery size

Recall@k only asks whether the correct item is in the top k. The metrics below look at the whole ranking. Relevance is
**exact-match**: an image has 5 correct captions, a caption has 1 correct image, and everything else counts as wrong
(a caption of a similar photo is not credited). All ranking numbers are on a 0 to 100 scale.

| Metric | Definition |
|---|---|
| **MRR** | Mean of 1 / (position of the first correct item). 100 means the correct item is always first. |
| **nDCG@k** | Each correct item in the top k earns 1 / log2(position + 1); the sum is divided by the best possible sum. Rewards putting correct items high. |
| **mAP** | Average precision over all correct items (image to text: 5 captions). For text to image there is one correct item, so mAP equals MRR and is shown once. |
| **Recall-all@k** | Share of an image's 5 correct captions found in the top k. Recall@k in section 4 counts only the best one. |
| **R-Precision** | Precision at k = 5, the number of correct captions. With 5 correct captions it equals Recall-all@5. |
| **Precision@10** | Correct captions in the top 10, divided by 10. It cannot exceed 50 with 5 correct captions. |
| **AUROC** | Chance that a class's score is higher for an image containing it than for one without. 50 is chance. |
| **Label-ranking AP** | For each image, how high its true classes sit in the ranking of all 80 classes. |
| **Label recall@k** | Share of an image's true classes among its k highest-scoring classes. |
| **Head / mid / tail** | Classes split into thirds by how many val+test images contain them (the natural COCO frequency; train is flat by design). |
| **Modality gap** | Distance between the centre of the image embeddings and the centre of the caption embeddings. |
| **Alignment / uniformity** | Alignment: how close matched pairs are (lower is closer). Uniformity: how evenly embeddings spread over the sphere (lower is more spread). |
""")

code("""
display(report.ext_retrieval_table(zero, finals, "test").rename_axis("test split"))
""")

code("""
display(report.ext_summary_by_split(zero, finals))
report.plot_ndcg(zero, finals, "test"); plt.show()
""")

md("""
**Gallery size.** Recall falls as the gallery grows, so numbers from a 1,000-image gallery are not comparable with
numbers from a 3,000-image one. The usual COCO protocol uses 1K galleries; the table gives the test split as three
disjoint 1K galleries (mean over the folds), as the full 3K gallery, and as a 5K gallery made of validation plus test
images. The 5K gallery includes validation images, which were used to pick the epoch, so its fine-tuned numbers are
slightly less independent.
""")

code("""
display(report.protocol_table(zero, finals))
""")

md("""
**Classification, beyond mAP.** Same prompts and the same validation-picked thresholds as section 2.
""")

code("""
display(report.cls_extras_table(zero, finals, "test").rename_axis("test split, 80 classes"))
print("Worst classes after fine-tuning (at least 20 test images):")
display(report.worst_classes_table(zero, finals, "test"))
""")

md("""
**What fine-tuning does to the embeddings.** CLIP's image and caption embeddings sit in two separate regions
(the modality gap). The table shows whether training closes it, how much closer matched pairs get, and whether the
temperature moved.
""")

code("""
display(report.diagnostics_table(zero, finals, "test").rename_axis("test split"))
""")

md("""
## 6. Caveats

- **CLIP's training data is private.** The model was not trained on COCO labels or captions, but its web data may
  contain COCO images. The zero-shot numbers could be slightly optimistic for that reason.
- **Small classes are noisy.** Toaster, hair drier, toothbrush and scissors have only a handful of test and validation
  images, so their per-class numbers move a lot. They are flagged in the charts.
- **Balanced train, natural test.** Balancing the train set deliberately departs from COCO's class mix, so the model
  sees rare classes far more often than the test set does. A model fine-tuned on a natural mix could behave differently.
- **Validation is used twice.** It picks the learning rate and the epoch, and the classification thresholds come from it,
  so its numbers are the least independent. Test and train are not used for any choice.
- **Three seeds** give a rough spread, not a tight interval.
- **Exact-match relevance.** The ranking metrics treat any caption or image other than the original as wrong, so they
  understate quality when another caption also describes the picture well. A graded notion of relevance (for example by
  shared object labels) was not used.
- **Not covered:** captioning (CLIP cannot caption) and BLIP. The split was built so BLIP can be scored on it later.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
