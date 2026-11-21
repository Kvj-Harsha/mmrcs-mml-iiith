# Multimodal Media Retrieval and Captioning

IIITH capstone project (CV-06).

![Model](https://img.shields.io/badge/model-CLIP%20ViT--B%2F32-blue)
![Task](https://img.shields.io/badge/task-image%E2%86%94text%20retrieval-informational)
![Test rsum](https://img.shields.io/badge/test%20rsum-397.1%20%E2%86%92%20447.1-brightgreen)
![Data](https://img.shields.io/badge/data-COCO%2010K%20(5K%20%2F%202K%20%2F%203K)-orange)
![Seeds](https://img.shields.io/badge/fine--tuning-3%20seeds-lightgrey)
![Notebooks](https://img.shields.io/badge/notebooks-2-lightgrey)

Given an image, find matching captions. Given text, find matching images. Generate captions for images that have none.

Problem statement: [problem-statement.md](problem-statement.md)

## Repository

| Path | What it is |
|---|---|
| `prepare_data.py` | Downloads COCO and builds the train/val/test split (step 1) |
| `eda/eda.ipynb` | Raw EDA → preprocessing → clean EDA → comparison (step 2) |
| `eda/report.md` | **Full data report**: findings and figures |
| `eda/before_vs_after.md` | **Before vs after preprocessing** comparison |
| `eda/eda_utils.py` | Analysis functions shared by both notebooks (same statistics for COCO and CC) |
| `prepare_cc.py` | Downloads Conceptual Captions: validation images + captions, train captions (text only) |
| `eda/cc_eda.ipynb` | Conceptual Captions: raw EDA → preprocessing → clean EDA → comparison with COCO |
| `eda/cc_report.md` | **Conceptual Captions report**, including COCO vs CC |
| `eda/figures/` | All plots used in the reports (`cc_*` and `coco_vs_cc_*` for Conceptual Captions) |
| `eda/summary_raw.json`, `eda/summary_clean.json`, `eda/comparison.csv` | All COCO numbers, machine-readable |
| `eda/cc_summary_raw.json`, `eda/cc_summary_clean.json`, `eda/cc_comparison.csv`, `eda/coco_vs_cc.csv` | All CC numbers |
| `clip-10k-task/notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb` | **Main notebook** of the 10K CLIP task: cleaned split, zero-shot vs fine-tuned CLIP, every metric ([results](#10k-coco-clip-task-zero-shot-vs-fine-tuned-retrieval)) |
| `clip-10k-task/notebooks/coco10k_clip_live_demo.ipynb` | **Live-demo notebook**: type a caption or upload a photo and compare both models side by side (offline, CPU) |
| `clip-10k-task/task10k/` | Code behind both notebooks: cleaning, split, zero-shot, fine-tuning, metrics, report tables |

## Quick start (for teammates)

Everyone runs the same commands and gets **exactly the same data**. There are two steps:
1. **Download and split:** `prepare_data.py`.
2. **Clean and build the vocabulary:** run the notebook. This also regenerates the EDA.

### 1. Get the code and install the libraries

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install pillow numpy pandas matplotlib seaborn scipy scikit-learn nltk jupyter
```

You need Python 3.9 or newer.

### 2. Pick a training-set size and run both steps

Val and test are always the same official sets. Only the training set changes.

| Size | Command (step 1) | Train / val / test | Disk during setup | Disk after `--delete-zip` | When to use |
|---|---|---|---|---|---|
| **30k (default)** | `python prepare_data.py` | 30,496 / 4,998 / 5,000 | ~14 GB | ~7 GB | normal work, **use this unless told otherwise** |
| 10k | `python prepare_data.py --train-size 10k` | 9,998 / 4,998 / 5,000 | ~11 GB | ~4 GB | quick tests on a weak laptop |
| full | `python prepare_data.py --train-size full` | ~113k / 4,998 / 5,000 | ~40 GB | ~20 GB | final runs on the DGX Spark only |

Then, for any size, run step 2:

```bash
jupyter nbconvert --to notebook --execute --inplace eda/eda.ipynb
```

- **Download:** ~7 GB (+13 GB for `full`). Interrupted? Just run the same command again; finished downloads are skipped.
- **Step 1** takes ~15 min (10k) to ~30 min (30k) after the download on a laptop. It extracts the images, fingerprints every image once to remove cross-split duplicates, and runs the checks. Re-runs are much faster: the fingerprints are cached.
- **Step 2** takes ~8 min (10k) or ~16 min (30k) on a laptop CPU, and about an hour for `full`.
- **Freeing disk:** add `--delete-zip` to step 1 to delete the big zips at the end.
- **Switching size later:** run step 1 with the new size, then step 2 again. Step 1 rebuilds `data/coco/train/`, and step 2 rebuilds the cleaned captions and vocabulary.

### 3. Check you have the right data

- **Step 1** must end with `ALL CHECKS PASSED`, and the fingerprints it prints must match this table:

| Split | Fingerprint |
|---|---|
| train, `30k` (default) | `bb749cb18143` |
| train, `10k` | `d7c01b1138ee` |
| train, `full` | printed by the script (it depends on duplicates found in `train2014`) |
| val | `e7f4d744e6f3` |
| test | `d2e58497f4f6` |

- **Step 2** must finish without errors.
- **Result:** you then have everything in `data/` (see [Data folder](#data-folder-not-pushed-to-github)).

### On Google Colab

```python
!git clone <this-repo-url>
%cd <repo-folder>
!python prepare_data.py --train-size 10k        # or leave out --train-size for 30k
!jupyter nbconvert --to notebook --execute --inplace eda/eda.ipynb
```

Pillow, numpy, pandas and the other libraries are already installed on Colab. Colab deletes files when the session ends, so copy `data/coco/` to Google Drive if you want to keep it.

## What we did

### 1. Data: COCO with the standard Karpathy split

- **COCO** provides the photos plus 5 human-written captions per image, and object labels.
- **The Karpathy split** is the train/val/test division used by almost every captioning and retrieval paper.

| Split | Images | Source |
|---|---:|---|
| train | **30,496** | all Karpathy "restval" images (part of the Karpathy train pool), minus 8 cross-split duplicates |
| val | 4,998 | official Karpathy val, minus 2 photos that are also in test |
| test | 5,000 | official Karpathy test (all, untouched) |

All 40,494 images come from one 6.6 GB download (COCO `val2014`).

**Checks run by `prepare_data.py` every time:**
- correct counts, and zero overlap between splits
- every image file has its captions and object labels (matched by COCO image ID, which is also in the filename)
- the captions match the official COCO file exactly
- **no photo appears in two splits**: COCO contains a few photos uploaded twice under different IDs, and the Karpathy split puts 10 of them in two splits. The script detects them (image hash + pixel check) and drops the extra copy, keeping test untouched. The list is in `data/coco/removed_duplicates.json`.
- the split fingerprints below match for everyone

### Why this split (and not 80/10/10)

- **Val and test are fixed benchmark sets,** not a percentage of our data. Keeping the official Karpathy 5k + 5k means our results can be compared directly with published papers, and 5,000 test images give stable scores.
- **A random 80/10/10 split** of our own would make our results comparable to no paper.
- **What matters is the size of the training set:**

| `--train-size` | Train images | Ratio | Download | Use |
|---|---:|---|---|---|
| `10k` | 9,998 | 50 / 25 / 25 | 6.6 GB | quick experiments and debugging |
| **`30k` (default)** | **30,496** | **≈ 75 / 12.5 / 12.5** | 6.6 GB | **current working set** |
| `full` | 113,287 | ≈ 92 / 4 / 4 (paper standard) | + 13 GB (`train2014.zip`) | final runs on the DGX Spark |

- The sets are **nested** (10k ⊂ 30k ⊂ full), so smaller-set results stay valid.
- Switching sizes rebuilds `data/coco/train/`.
- Moving from 10k to 30k halved the share of val captions containing an unknown word (20% → 10%).

### 2. EDA on the raw data

We analysed the data exactly as downloaded:

- **Integrity:** counts, duplicate captions, captions shared across images.
- **Text, full NLP statistics:**
  - vocabulary size, type–token ratio, root TTR, MTLD
  - **hapax and dis legomena**, the frequency spectrum, **Zipf's law** and **Heaps' law**
  - caption, word and character lengths
  - stopwords, n-grams, PMI collocations
  - part-of-speech mix, lemmas, typos
  - out-of-vocabulary rates, and vocabulary overlap between splits
- **Images (all 40,494):** size, aspect ratio, colour mode, corruption, colour statistics, brightness, contrast, sharpness, and **duplicate images**.
- **Objects:** the 80 COCO categories, and whether captions mention the objects in the image.

### 3. Preprocessing

The design follows the standard COCO captioning pipelines (Karpathy, ruotianluo, Show-Attend-Tell, CLIP and BLIP code).

| | Steps | Why |
|---|---|---|
| **Text** | Unicode NFKC → lowercase → remove apostrophes → punctuation and hyphens to spaces → conservative typo correction (train-only, logged) → drop exact duplicate captions | Removes formatting noise that splits one word into many vocabulary entries |
| **Kept on purpose** | Stopwords, all captions (quality is only flagged), original caption files | A caption model must generate stopwords; evaluation uses the original captions |
| **Vocabulary** | Train only, words seen ≥ 5 times + `<pad> <start> <end> <unk>` = 5,718 tokens; max length 19 words | Standard thresholds, no val/test leakage |
| **Images** | RGB → resize shorter side to 224 (bicubic) → centre-crop 224 → normalise (CLIP or ImageNet mean/std). Training: random resized crop + horizontal flip, except the 560 images whose captions mention left/right | CLIP/BLIP standard. Originals kept because different models need different sizes. |

### 4. EDA on the cleaned data and comparison

The **same statistics** were re-computed on the cleaned data. Highlights:

| | Before | After |
|---|---:|---:|
| Distinct words (train) | 27,588 | 14,469 (−48%) |
| Hapax legomena | 12,862 | 5,117 (−60%) |
| Vocabulary at min-freq 5 | 8,122 | 5,714 |
| Val words unknown to the train vocabulary | 2.2% | 1.1% (−49%) |
| Val captions with an unknown word | 17.9% | 10.3% |
| Image sizes | 1,534 different | all 224×224×3, mean ≈ 0, std ≈ 1 |

Details: [`eda/before_vs_after.md`](eda/before_vs_after.md).

### 5. Key findings

Full report: [`eda/report.md`](eda/report.md).

- **Captions:** short (10.5 words), 45% stopwords, object-centric (37% nouns, 11% verbs). They mention big objects ~90% of the time and small ones only 33%.
- **Images:** uniform (mostly 640 px, 4:3 or 3:2), 66 grayscale, none corrupt.
- **Duplicates:** the original Karpathy split had **10 photos in two splits** (6 train↔val, 2 train↔test, 2 val↔test). They're now removed, and an independent re-check finds 0 cross-split duplicates.

## Conceptual Captions (second dataset)

**What it is:** Conceptual Captions (CC3M, Google, 2018) is image–caption pairs taken from the alt-text of web images.
- **Splits:** train 3,318,333 · validation 15,840 · test hidden.
- **Format:** one caption per image. Names are replaced by generic words ("actor", "person").
- **Images:** only distributed as web links, and many have died since 2018.
- **Our use:** the **validation set as an out-of-domain test** for models trained on COCO. This is the standard practice.

### How to run

Run this after the COCO steps, because the comparison uses the COCO results:

```bash
pip install pyarrow                   # in addition to the libraries above
python prepare_cc.py                  # ~20 min: validation images + captions, train captions (text only)
jupyter nbconvert --to notebook --execute --inplace eda/cc_eda.ipynb   # ~5-10 min
```

- **Download:** ~360 MB of caption files plus ~1.4 GB of images.
- **Check:** the script must end with `ALL CHECKS PASSED`.
- **Interrupted?** Just run it again: finished images are kept, and known dead links aren't retried (`--retry-failed` retries them).
- **Teammates' numbers may differ slightly:** links keep dying, so a teammate may get a few images fewer than our 10,038. `data/cc/val/download_log.csv` records what was fetched and why each failure happened.

### Key results

Full report: [`eda/cc_report.md`](eda/cc_report.md).

| | COCO | Conceptual Captions |
|---|---:|---:|
| Images analysed | 40,494 | 10,038 (63% of validation links still work) |
| Captions per image | 5 | 1 |
| Mean caption length (words) | 10.5 | 9.6 |
| Lexical diversity (MTLD) | 25 | 154 |
| Training vocabulary | 5,718 | 26,990 |
| Val captions with a word the **other** dataset's vocabulary doesn't know | 6.9% | **64.3%** |
| Duplicate images shared with the other dataset | 0 | 0 |

- **One-sided domain gap:** CC's vocabulary covers COCO, but COCO's doesn't cover CC, so a COCO-only model will struggle on web images.
- **CC is web text,** not descriptions: stock-photo language ("isolated on a white background"), generic names ("actor attends the premiere"), and some machine-generated alt-text.
- **CC preprocessing changes little:** captions are already lowercased and tokenised, so the main step is removing punctuation tokens. The vocabulary follows the CC paper's settings and is built from all 3.3M train captions.

## 10K COCO CLIP task: zero-shot vs fine-tuned retrieval

**Question:** how much does fine-tuning on a small, cleaned, class-balanced COCO set improve CLIP for **image → text** and **text → image** retrieval?

**Answer:** retrieval rsum on the held-out test set goes from **397.1 to 447.1 ± 0.8** (out of 600). Text → image gains the most. Notebooks: [main](clip-10k-task/notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb) · [live demo](clip-10k-task/notebooks/coco10k_clip_live_demo.ipynb).

### Setup

| | |
|---|---|
| **Model** | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), shared 512-d image-text space, cosine similarity |
| **Data** | A cleaned 10K COCO 2017 split with 5 human captions per image: **train 5,000** (class-balanced, Gini 0.082 against 0.496 for a random draw), **validation 2,000**, **test 3,000** (natural class mix, never trained on). Cleaning: 123,287 COCO images examined, 118,361 candidates kept. |
| **Test protocol** | Image → text: each photo is ranked against all **15,000** test captions (5 are correct). Text → image: each caption is ranked against all **3,000** photos (1 is correct). Zero-shot and fine-tuned use the same test set, gallery and metric code. |
| **Fine-tuning** | Contrastive loss where all 5 captions of a photo count as matches. AdamW, weight decay 0.1, cosine schedule with 10% warm-up, bf16, batch 64, **lr 1e-6**, up to 10 epochs, stop after 2 epochs without a validation gain, best-validation epoch kept. Seeds 0, 1, 2 (best epochs 6, 8, 6). |
| **Why CLIP, not BLIP** | BLIP was pretrained on COCO train images, so its "before" score would not be zero-shot. CLIP's web data is private and cannot be proven COCO-free; this is stated as a limit below. |

### Results (test set)

Zero-shot is the released CLIP. Fine-tuned is the mean ± std over 3 training seeds.

| Metric | Zero-shot | Fine-tuned | Change |
|---|---:|---:|---:|
| **rsum** (max 600) | 397.1 (95% CI 392.7 to 401.2) | **447.1 ± 0.8** | **+49.9** |
| Image → text R@1 / R@5 / R@10 | 56.5 / 80.1 / 87.5 | 62.8 / 86.0 / 92.5 | +6.3 / +5.8 / +5.0 |
| Text → image R@1 / R@5 / R@10 | 35.8 / 63.2 / 74.0 | 47.1 / 74.5 / 84.1 | +11.3 / +11.3 / +10.1 |
| Image → text MRR / mAP | 67.0 / 42.9 | 73.0 / 51.5 | +6.0 / +8.5 |
| Text → image MRR (= mAP) | 48.6 | 59.6 | +11.0 |
| Image → text nDCG@10 | 49.6 | 58.1 | +8.6 |
| Text → image nDCG@10 | 53.8 | 64.9 | +11.1 |
| Text → image mean rank (lower is better) | 14.18 | 8.41 | -5.8 |
| 80-class zero-shot classification mAP (a different metric) | 52.8 | 57.3 ± 0.3 | +4.6 |

<details>
<summary>All retrieval metrics, both directions</summary>

**Image → text** (3,000 queries, 5 correct captions each)

| Metric | Zero-shot | Fine-tuned | Change |
|---|---:|---:|---:|
| R@1 / R@5 / R@10 | 56.5 / 80.1 / 87.5 | 62.8 / 86.0 / 92.5 | +6.3 / +5.8 / +5.0 |
| MRR | 67.0 | 73.0 | +6.0 |
| mAP | 42.9 | 51.5 | +8.5 |
| nDCG@5 / @10 / @50 | 42.7 / 49.6 / 59.5 | 50.1 / 58.1 / 67.5 | +7.4 / +8.6 / +8.0 |
| Recall-all@5 / @10 / @50 | 38.8 / 51.2 / 77.3 | 46.1 / 60.7 / 85.2 | +7.3 / +9.5 / +7.9 |
| Precision@10 (max 50) | 25.6 | 30.3 | +4.7 |
| Median / mean rank | 1 / 6.00 | 1 / 4.12 | 0 / -1.9 |

**Text → image** (15,000 queries, 1 correct photo each)

| Metric | Zero-shot | Fine-tuned | Change |
|---|---:|---:|---:|
| R@1 / R@5 / R@10 | 35.8 / 63.2 / 74.0 | 47.1 / 74.5 / 84.1 | +11.3 / +11.3 / +10.1 |
| MRR (= mAP) | 48.6 | 59.6 | +11.0 |
| nDCG@5 / @10 / @50 | 50.3 / 53.8 / 58.5 | 61.8 / 64.9 / 68.1 | +11.5 / +11.1 / +9.6 |
| Median / mean rank | 3 / 14.18 | 2 / 8.41 | -1 / -5.8 |

Zero-shot intervals are 95% bootstrap intervals over test queries. Fine-tuned values are the mean ± std over 3 seeds (std at most 0.4 per metric, 0.8 for rsum). These are two different kinds of uncertainty and are not directly comparable. The notebook has the full tables with intervals.

</details>

### Robustness to gallery size

The gain holds at every gallery size (rsum, zero-shot → fine-tuned): **1K gallery** 473.3 → 512.0, **3K gallery (full test)** 397.1 → 447.1, **5K gallery (validation + test)** 358.6 → 411.9. A bigger gallery has more distractors and lower recall, but the gain stays at about 39 to 53 rsum points.

### Hyperparameter study (validation only, seed 0)

The learning rate and stopping epoch were chosen on validation rsum only; the test set was not used for any choice. Zero-shot validation rsum is 425.7.

| Batch / learning rate | Best epoch | Epochs run | Best val rsum | Minutes |
|---|---:|---:|---:|---:|
| 64 / 1e-5 | 1 | 3 | 447.3 | 5.5 |
| 64 / 3e-6 | 4 | 6 | 469.3 | 11.3 |
| **64 / 1e-6 (used)** | 6 | 8 | **472.1** | 14.6 |
| 64 / 3e-7 | 8 | 10 | 469.4 | 14.5 |
| 256 / 4e-6 (rate scaled with batch) | 3 | 5 | 472.6 | 5.8 |

- **1e-6 is a real optimum:** both a higher rate (overfits after 1 epoch at 1e-5) and a lower rate (3e-7) are worse.
- **Batch 256 ties** (472.6 against 472.1, inside seed noise) and gets there in 3 epochs instead of 6, so it is a faster alternative. The delivered setting stays at batch 64.
- **Compared with the public [ITRA CLIP fine-tuning guide](https://itra.readthedocs.io/en/latest/Contents/example-usage/clip-finetuning.html)** (ResNet50, batch 256 to 800, lr 1e-5 to 3.1e-5): our recipe is the same family with a smaller learning rate, which suited ViT-B/32 here.
- **Not tried:** weight EMA, weight decay 1.0, batch 800. Single seed, validation only.

### Why it works

- **Wrong pairs are pushed apart, right pairs stay close:** mean cosine of matched pairs 0.305 → 0.299, of unmatched pairs 0.151 → 0.102, so the margin grows from 0.154 to 0.198.
- **The modality gap does not close** (0.827 → 0.852): image and text embeddings still sit in separate regions, but ranking inside the shared space gets sharper.
- **Classification gains concentrate on frequent classes:** 80-class mAP head 51.4 → 59.8, mid 60.7 → 64.6, tail 46.0 → 47.3.

### What each metric means in this codebase

Defined in [`task10k/metrics.py`](clip-10k-task/task10k/metrics.py) and [`task10k/metrics_ext.py`](clip-10k-task/task10k/metrics_ext.py). Every retrieval metric is computed per query, averaged over the queries of one direction, and reported as a percentage (0 to 100). The rank of an item is the number of candidates scoring **strictly higher**, so ties favour the correct item.

| Metric | Meaning |
|---|---|
| **R@K** | Hit rate: % of queries with at least one correct item in the top K (for image → text, at least one of the 5 captions) |
| **rsum** | R@1 + R@5 + R@10 in both directions, so the maximum is 600; each recall is averaged over queries before summing |
| **MRR** | Mean of 1 / (position of the first correct item) |
| **mAP** | Average precision over all correct items on the full ranking. For text → image it equals MRR |
| **nDCG@K** | Binary relevance (a photo's own 5 captions count, nothing else), discount 1 / log2(position + 1), divided by the ideal ranking |
| **Recall-all@K** (image → text) | The **fraction** of the photo's 5 captions found in the top K (not "all five present"). Recall-all@5 is also R-Precision |
| **Precision@10** (image → text) | Correct captions in the top 10 divided by 10, so the maximum is 50 |

### Limits

- **CLIP's pretraining data is private,** so it cannot be proven free of COCO images.
- **Retrieval and classification only:** caption generation is not covered yet.
- **Exact-match relevance:** only a photo's own 5 captions count as correct, so similar captions of other photos score as wrong and all metrics are somewhat pessimistic.
- **Intervals are approximate:** the zero-shot bootstrap resamples captions and photos as if independent.
- **The hyperparameter study** is validation only and uses one seed.

### Notebooks and code

| File | What it is |
|---|---|
| [`coco10k_clip_zeroshot_vs_finetuned.ipynb`](clip-10k-task/notebooks/coco10k_clip_zeroshot_vs_finetuned.ipynb) | Main notebook: data cleaning and split, zero-shot baseline, fine-tuning, all metrics, plots. Executed with saved outputs (it renders on GitHub). Setting `RERUN = True` in its second cell recomputes everything (about 1.5 hours on one GPU) |
| [`coco10k_clip_live_demo.ipynb`](clip-10k-task/notebooks/coco10k_clip_live_demo.ipynb) | Live demo: type any caption or upload any photo and see the top results of both models side by side. Runs offline on a laptop CPU. Needs `demo_assets/` (671 MB, not in this repo), built with `python -m task10k.make_demo_assets` |
| [`clip-10k-task/task10k/`](clip-10k-task/task10k) | Code: `clean.py`, `split.py`, `zeroshot.py`, `finetune.py`, `metrics.py`, `metrics_ext.py`, `report.py`, `demo.py` and the notebook builders |

```bash
# from clip-10k-task/ (needs the COCO 2017 images and annotations, and a CUDA GPU)
python -m task10k.split                 # builds data/task10k (train 5000 / val 2000 / test 3000)
python -m task10k.zeroshot              # results/task10k/zeroshot.json
python -m task10k.finetune --lr 1e-6 --seed 0 --final --out results/task10k/final/s0.json   # seeds 0, 1, 2
python -m task10k.make_notebook         # rebuilds the main notebook
```

`results/` and `demo_assets/` are git-ignored. The code in `task10k/` also imports small helper packages (`evaluation/`, `training/`, `selection/`) from the full project, which are not part of this repository.

## Data folder (not pushed to GitHub)

```
data/
├── coco/
│   ├── train/                    30,496 images (default)
│   │   ├── images/               the photos (COCO_val2014_<id>.jpg), original, untouched
│   │   ├── captions.json         original captions (5+ per image)  ← use for evaluation
│   │   ├── captions_clean.json   cleaned captions + tokens          ← use for training
│   │   └── objects.json          labelled objects per image
│   ├── val/                      4,998 images  (same files)
│   ├── test/                     5,000 images  (same files)
│   ├── vocab.json                vocabulary (train only)
│   ├── preprocess_config.json    every text/image preprocessing setting
│   ├── removed_duplicates.json   the 10 cross-split duplicate photos that were dropped
│   └── analysis/                 spelling corrections, duplicate images, caption flags, image info
├── cc/                           Conceptual Captions
│   ├── val/
│   │   ├── images/               downloaded validation images (cc_val_<row>.<ext>)
│   │   ├── captions.json         one caption per downloaded image
│   │   ├── captions_clean.json   cleaned captions + tokens
│   │   ├── all_captions.json     all 15,840 validation captions (incl. dead links)
│   │   └── download_log.csv      status of every link
│   ├── train_captions.txt        all 3,318,333 train captions (text only)
│   ├── vocab.json, preprocess_config.json
│   └── analysis/
└── downloads/                    original downloaded files (can be deleted)
```

- `prepare_data.py` creates `images/`, `captions.json`, `objects.json` and `removed_duplicates.json`.
- `eda/eda.ipynb` creates the rest, so re-run it after switching train size.
- Every file lists images in the same order, matched by COCO image ID.
