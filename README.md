# Multimodal Media Retrieval and Captioning

IIITH capstone project (CV-06).

Given an image, find matching captions. Given text, find matching images. Generate captions for images that have none.

Problem statement: [problem-statement.md](problem-statement.md)

## Repository

| Path | What it is |
|---|---|
| `prepare_data.py` | Downloads COCO and builds the train/val/test split (step 1) |
| `eda/eda.ipynb` | Raw EDA → preprocessing → clean EDA → comparison (step 2) |
| `eda/report.md` | **Full data report**: findings and figures |
| `eda/before_vs_after.md` | **Before vs after preprocessing** comparison |
| `eda/figures/` | All plots used in the reports |
| `eda/summary_raw.json`, `eda/summary_clean.json`, `eda/comparison.csv` | All numbers, machine-readable |

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
└── downloads/                    original downloaded files (can be deleted)
```

- `prepare_data.py` creates `images/`, `captions.json`, `objects.json` and `removed_duplicates.json`.
- `eda/eda.ipynb` creates the rest, so re-run it after switching train size.
- Every file lists images in the same order, matched by COCO image ID.
