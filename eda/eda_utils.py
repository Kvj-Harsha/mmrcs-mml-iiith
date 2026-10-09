"""Shared analysis functions for the COCO and Conceptual Captions EDA notebooks.

Both notebooks call the same functions, so their statistics are directly comparable.
"""

import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image

for _res in ["stopwords", "words", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(_res, quiet=True)
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder  # noqa: E402
from nltk.corpus import stopwords, wordnet  # noqa: E402
from nltk.corpus import words as nltk_words  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "eda" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

STOP = set(stopwords.words("english"))
ENGLISH = set(w.lower() for w in nltk_words.words())
LEMMATIZER = WordNetLemmatizer()


def savefig(name, dpi=110):
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=dpi)


def known_word(w):
    """True if w is an English word (dictionary or WordNet, which handles plurals/verb forms) or a number."""
    return w in ENGLISH or w.isdigit() or bool(wordnet.morphy(w))


def pct(x):
    return f"{x:.2%}"


# ----------------------------------------------------------------------------------------------- text statistics
def mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), mean of forward and backward passes."""
    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for t in seq:
            count += 1
            types.add(t)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        return len(seq) / factors if factors else float("nan")
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def heaps_curve(token_lists, seed=SEED):
    """Vocabulary size V after reading N tokens (captions in random order)."""
    order = list(range(len(token_lists)))
    random.Random(seed).shuffle(order)
    seen, n, xs, ys = set(), 0, [], []
    checkpoints = set(np.unique(np.logspace(2, np.log10(sum(map(len, token_lists))), 40).astype(int)))
    for i in order:
        for t in token_lists[i]:
            n += 1
            seen.add(t)
            if n in checkpoints:
                xs.append(n); ys.append(len(seen))
    xs.append(n); ys.append(len(seen))
    beta, logk = np.polyfit(np.log(xs), np.log(ys), 1)
    return np.array(xs), np.array(ys), float(np.exp(logk)), float(beta)


def norm_form(t):
    """Lowercase letters/digits only: used to compare raw and clean tokens on equal footing."""
    return re.sub(r"[^a-z0-9]", "", t.lower())


def text_stats(df, label, pos_sample=20000, train_name="train", eval_name="val", splits=("train", "val", "test")):
    """All text statistics for a caption table with columns split, text, tokens.

    Vocabulary-type statistics are computed on the `train_name` split; OOV is measured on `eval_name`.
    """
    S, F = {"label": label}, {}  # S: numbers, F: series/tables for plots
    toks_all = df.tokens.tolist()
    train = df[df.split == train_name]
    val = df[df.split == eval_name]
    train_toks = train.tokens.tolist()
    flat_train = [t for ts in train_toks for t in ts]
    counts = Counter(flat_train)
    all_counts = Counter(t for ts in toks_all for t in ts)
    F["counts"] = counts

    # size and richness
    S["captions"] = len(df)
    S["tokens"] = int(sum(map(len, toks_all)))
    S["types_all_splits"] = len(all_counts)
    S["train_tokens"] = len(flat_train)
    S["train_types"] = len(counts)
    S["ttr"] = S["train_types"] / S["train_tokens"]
    S["root_ttr"] = S["train_types"] / math.sqrt(S["train_tokens"])
    S["mtld"] = mtld(flat_train)

    # rare words
    spectrum = Counter(counts.values())
    hapax = spectrum[1]
    S["hapax"] = hapax
    S["hapax_pct_types"] = hapax / len(counts)
    S["hapax_pct_tokens"] = hapax / len(flat_train)
    S["dis_legomena"] = spectrum[2]
    S["dis_pct_types"] = spectrum[2] / len(counts)
    S["hapax_dis_ratio"] = hapax / max(spectrum[2], 1)
    F["spectrum"] = pd.Series({m: spectrum[m] for m in range(1, 11)})

    # Zipf and Heaps
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    sl = slice(9, 1000)
    S["zipf_slope"] = float(np.polyfit(np.log(ranks[sl]), np.log(freqs[sl]), 1)[0])
    F["zipf"] = (ranks, freqs)
    hx, hy, k, beta = heaps_curve(train_toks)
    S["heaps_K"], S["heaps_beta"] = k, beta
    F["heaps"] = (hx, hy)

    # length
    n_words = df.tokens.map(len)
    F["n_words"] = n_words
    S["len_mean"] = float(n_words.mean())
    S["len_std"] = float(n_words.std())
    S["len_median"] = float(n_words.median())
    S["len_p95"] = float(n_words.quantile(.95))
    S["len_p99"] = float(n_words.quantile(.99))
    S["len_min"] = int(n_words.min())
    S["len_max"] = int(n_words.max())
    S["chars_mean"] = float(df.text.str.len().mean())
    S["word_len_mean"] = float(np.mean([len(t) for t in flat_train]))

    # stopwords and n-grams
    S["stopword_share"] = float(np.mean([norm_form(t) in STOP for t in flat_train]))
    F["top_words"] = counts.most_common(20)
    F["top_content"] = [(w, c) for w, c in counts.most_common() if norm_form(w) not in STOP and norm_form(w)][:20]
    bi = Counter(g for ts in train_toks for g in zip(ts, ts[1:]))
    tri = Counter(g for ts in train_toks for g in zip(ts, ts[1:], ts[2:]))
    F["top_bigrams"] = bi.most_common(12)
    F["top_trigrams"] = tri.most_common(12)
    finder = BigramCollocationFinder.from_documents(train_toks)
    finder.apply_freq_filter(20)
    F["collocations"] = finder.score_ngrams(BigramAssocMeasures.pmi)[:15]

    # grammar
    sample = train.tokens.sample(min(pos_sample, len(train)), random_state=SEED)
    POS_MAP = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ", "RB": "ADV", "IN": "ADP", "DT": "DET", "CC": "CONJ",
               "PR": "PRON", "CD": "NUM", "TO": "PRT", "RP": "PRT", "WD": "DET", "WP": "PRON"}
    pos_counts, by_pos = Counter(), defaultdict(Counter)
    for sent in sample:
        for w, tag in nltk.pos_tag(list(sent)):
            p = POS_MAP.get(tag[:2], "OTHER")
            pos_counts[p] += 1
            by_pos[p][w] += 1
    F["pos"] = pd.Series(pos_counts).sort_values(ascending=False) / sum(pos_counts.values())
    F["pos_top"] = {p: by_pos[p].most_common(12) for p in ["NOUN", "VERB", "ADJ"]}
    for p in ["NOUN", "VERB", "ADJ", "DET", "ADP"]:
        S[f"pos_{p.lower()}"] = float(F["pos"].get(p, 0))
    lemmas = {LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(w, "n"), "v") for w in counts}
    S["train_lemmas"] = len(lemmas)
    S["types_per_lemma"] = len(counts) / len(lemmas)

    # spelling
    S["types_not_dictionary_as_written"] = sum(1 for w in counts if not known_word(w))
    norm_counts = Counter()
    for w, c in counts.items():
        norm_counts[norm_form(w)] += c
    typos = sorted(w for w, c in norm_counts.items() if w.isalpha() and c <= 2 and not known_word(w))
    S["likely_typo_types"] = len(typos)
    F["typos"] = typos

    # generalisation: vocabulary, coverage, OOV, split overlap
    for mf in (1, 5):
        vocab = {w for w, c in counts.items() if c >= mf}
        val_tok = [t for ts in val.tokens for t in ts]
        S[f"vocab_min{mf}"] = len(vocab)
        S[f"train_coverage_min{mf}"] = sum(c for w, c in counts.items() if c >= mf) / len(flat_train)
        S[f"val_oov_tokens_min{mf}"] = float(np.mean([t not in vocab for t in val_tok]))
        S[f"val_oov_captions_min{mf}"] = float(np.mean([any(t not in vocab for t in ts) for ts in val.tokens]))
    types = {s: set(t for ts in df[df.split == s].tokens for t in ts) for s in splits}
    for s in splits:
        if s != train_name:
            S[f"jaccard_{train_name}_{s}"] = len(types[train_name] & types[s]) / len(types[train_name] | types[s])
    return S, F


def char_stats(texts):
    """Surface / character-level statistics of caption strings."""
    t = pd.Series(texts)
    punct = Counter(ch for s in t for ch in s if not ch.isalnum() and not ch.isspace())
    return {
        "has_uppercase": float(t.str.contains(r"[A-Z]").mean()),
        "starts_uppercase": float(t.str.strip().str.match(r"^[A-Z]").mean()),
        "all_caps_word": float(t.str.contains(r"\b[A-Z]{2,}\b").mean()),
        "ends_with_period": float(t.str.strip().str.endswith(".").mean()),
        "has_punctuation": float(t.map(lambda s: any(not c.isalnum() and not c.isspace() for c in s)).mean()),
        "has_digit": float(t.str.contains(r"\d").mean()),
        "has_non_ascii": float(t.map(lambda s: any(ord(c) > 127 for c in s)).mean()),
        "has_double_space": float(t.str.contains("  ").mean()),
        "has_edge_whitespace": float((t != t.str.strip()).mean()),
        "distinct_characters": len(set("".join(t))),
    }, punct


def show_text_stats(S, F, prefix, splits=("train", "val", "test")):
    """Plots + tables for one text_stats result."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for s in splits:
        vals = F["n_words"][F["split"] == s].value_counts(normalize=True).sort_index()
        ax[0].plot(vals.index, vals.values, marker="o", ms=3, label=s)
    ax[0].set(xlabel="words per caption", ylabel="share of captions", title=f"[{prefix}] caption length by split", xlim=(0, 30))
    ax[0].legend()
    for a, data, title in [(ax[1], F["top_words"], "top 20 words"), (ax[2], F["top_content"], "top 20 content words")]:
        w, c = zip(*data)
        a.barh([str(x) for x in w[::-1]], c[::-1], color="#4C72B0")
        a.set(title=f"[{prefix}] {title}", xlabel="count (train)")
    plt.tight_layout(); savefig(f"{prefix}_text_overview"); plt.show()

    table = pd.Series({k: v for k, v in S.items() if not isinstance(v, (list, dict, str))}).to_frame(prefix)
    display(table.style.format("{:,.4g}"))
    display(pd.DataFrame({
        "top bigrams": [f"{' '.join(g)} ({c:,})" for g, c in F["top_bigrams"]],
        "top trigrams": [f"{' '.join(g)} ({c:,})" for g, c in F["top_trigrams"]],
        "collocations (PMI, freq ≥ 20)": [f"{a} {b} ({s:.1f})" for (a, b), s in F["collocations"][:12]],
    }))
    display(pd.DataFrame({p: [f"{w} ({c})" for w, c in v] for p, v in F["pos_top"].items()}))


def plot_rare(F, S, prefix):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
    F["spectrum"].plot.bar(ax=ax[0], color="#C44E52", rot=0)
    ax[0].set(title=f"[{prefix}] frequency spectrum", xlabel="word occurs exactly m times", ylabel="number of words")
    r, f = F["zipf"]
    ax[1].loglog(r, f)
    ax[1].set(title=f"[{prefix}] Zipf: slope {S['zipf_slope']:.2f}", xlabel="rank", ylabel="frequency")
    x, y = F["heaps"]
    ax[2].loglog(x, y, "o-", ms=3, label="observed")
    ax[2].loglog(x, S["heaps_K"] * x ** S["heaps_beta"], "--", label=f"V = {S['heaps_K']:.1f}·N^{S['heaps_beta']:.3f}")
    ax[2].set(title=f"[{prefix}] Heaps: vocabulary growth", xlabel="tokens read (N)", ylabel="vocabulary (V)")
    ax[2].legend()
    plt.tight_layout(); savefig(f"{prefix}_rare_zipf_heaps"); plt.show()
    print(f"hapax: {S['hapax']:,} words = {S['hapax_pct_types']:.1%} of the vocabulary, {S['hapax_pct_tokens']:.2%} of all tokens; "
          f"dis legomena: {S['dis_legomena']:,} ({S['dis_pct_types']:.1%})")


# ---------------------------------------------------------------------------------------------- image statistics
def laplacian_var(g):
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def dhash(gray_img):
    a = np.asarray(gray_img.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(">u8")[0])


def image_pass(items):
    """Decode every image once and measure it. items: iterable of (path, dict of id columns).

    Returns (DataFrame with one row per image, pixel mean per channel, pixel std per channel).
    """
    rows, px_sum, px_sq, n_px = [], np.zeros(3), np.zeros(3), 0
    for path, ids in items:
        info = {**ids, "bytes": path.stat().st_size}
        try:
            with Image.open(path) as im:
                info.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                im.load()
                rgb = im.convert("RGB")
            rgb.thumbnail((256, 256), Image.BILINEAR)
            a = np.asarray(rgb, dtype=np.float64) / 255
            g = np.asarray(rgb.convert("L"), dtype=np.float64) / 255
            px = a.reshape(-1, 3)
            px_sum += px.sum(0); px_sq += (px ** 2).sum(0); n_px += len(px)
            info.update(ok=True, mean_r=px[:, 0].mean(), mean_g=px[:, 1].mean(), mean_b=px[:, 2].mean(),
                        brightness=g.mean(), contrast=g.std(), sharpness=laplacian_var(g),
                        colourfulness=float(np.abs(a - a.mean(2, keepdims=True)).mean()),
                        dhash=dhash(rgb.convert("L")))
        except Exception as e:
            info.update(ok=False, error=str(e))
        rows.append(info)
    df = pd.DataFrame(rows)
    # keep the 64-bit hashes as exact Python ints (a float column would lose precision if any image failed)
    df["dhash"] = pd.Series([r.get("dhash") for r in rows], dtype=object)
    df["aspect"] = df.width / df.height
    df["orientation"] = np.select([df.aspect > 1.05, df.aspect < 0.95], ["landscape", "portrait"], "square-ish")
    mean = px_sum / n_px
    return df, mean, np.sqrt(px_sq / n_px - mean ** 2)


def small_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255


def find_duplicates(ok, key, group, path_of, max_bits=3, min_corr=0.95, max_aspect_diff=0.05):
    """Duplicate images among rows of `ok` (output of image_pass with ok=True).

    Stage 1: dHash candidates differing in <= max_bits of 64 bits (pigeonhole over four 16-bit chunks).
    Stage 2: confirmed if 64x64 grayscale thumbnails correlate > min_corr and aspect ratios match.
    Calibration on hand-checked COCO pairs: true duplicates 0.97-1.00, false hash matches 0.17-0.69.
    Returns one row per candidate pair: a, b (values of `key`), split_a, split_b (values of `group`), hamming,
    aspect_diff, pixel_corr, confirmed, cross_split.
    """
    H = ok.dhash.astype("uint64").to_numpy()
    idx = ok.index.to_numpy()
    cand = set()
    for k in range(4):
        chunk = (H >> np.uint64(16 * k)) & np.uint64(0xFFFF)
        for g in pd.Series(np.arange(len(H))).groupby(chunk):
            members = g[1].to_numpy()
            if 1 < len(members) <= 300:
                cand.update(combinations(members, 2))
    pairs = [(a, b, bin(int(H[a] ^ H[b])).count("1")) for a, b in cand]
    pairs = [(idx[a], idx[b], d) for a, b, d in pairs if d <= max_bits]
    dup = pd.DataFrame([{"a": ok.loc[a, key], "b": ok.loc[b, key], "split_a": ok.loc[a, group],
                         "split_b": ok.loc[b, group], "hamming": d,
                         "aspect_diff": abs(ok.loc[a, "aspect"] - ok.loc[b, "aspect"])} for a, b, d in pairs],
                       columns=["a", "b", "split_a", "split_b", "hamming", "aspect_diff"])
    dup["pixel_corr"] = [float(np.corrcoef(small_gray(path_of(a)).ravel(), small_gray(path_of(b)).ravel())[0, 1])
                         for a, b in zip(dup.a, dup.b)]
    dup["confirmed"] = (dup.pixel_corr > min_corr) & (dup.aspect_diff < max_aspect_diff)
    dup["cross_split"] = dup.split_a != dup.split_b
    return dup


def show_images(items, ncols, name, title, img_h=4.2, text_lines=None):
    """Grid of images with text underneath. items: list of (path, panel title, text)."""
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, (img_h + 0.22 * (text_lines or 6)) * nrows), squeeze=False)
    for ax, (path, panel_title, text) in zip(axes.ravel(), items):
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(panel_title, fontsize=9)
        ax.text(0, -0.02, text, transform=ax.transAxes, va="top", fontsize=8, family="monospace")
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(h_pad=3)
    savefig(name, dpi=70)
    plt.show()


# ------------------------------------------------------------------------------------------- image transforms
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


def resize_shorter(im, size):
    w, h = im.size
    s = size / min(w, h)
    return im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def center_crop(im, size):
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def normalise(im, mean=CLIP_MEAN, std=CLIP_STD):
    return (np.asarray(im, dtype=np.float32) / 255 - mean) / std


def eval_transform(im):
    return normalise(center_crop(resize_shorter(im.convert("RGB"), SIZE), SIZE))


def random_resized_crop(im, r, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    w, h = im.size
    for _ in range(10):
        area = w * h * r.uniform(*scale)
        ar = math.exp(r.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = round(math.sqrt(area * ar)), round(math.sqrt(area / ar))
        if 0 < cw <= w and 0 < ch <= h:
            x, y = r.randint(0, w - cw), r.randint(0, h - ch)
            return im.crop((x, y, x + cw, y + ch)).resize((SIZE, SIZE), Image.BICUBIC)
    return center_crop(resize_shorter(im, SIZE), SIZE)
