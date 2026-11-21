"""Helpers of the live demo: zero-shot vs fine-tuned CLIP side by side on the 3,000 test photos.

Runs offline on CPU. The gallery embeddings are precomputed (demo_assets/test_embeddings.npz), so a query
costs one text or image encoding per model. Captions of test image i are rows 5i..5i+4 of the caption matrix.
"""
import base64
import html
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = (("zero-shot", "clip_zeroshot"), ("fine-tuned", "clip_ft_s0"))
GOOD, BAD, MUTED = "#15803d", "#b91c1c", "#64748b"


def _b64(img, size=150, quality=80):
    img = img.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class Demo:
    def __init__(self, assets="demo_assets", threads=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import CLIPModel, CLIPProcessor
        self.root = Path(assets)
        if threads:
            torch.set_num_threads(threads)
        with open(self.root / "test_items.json") as f:
            self.items = json.load(f)
        self.captions = [c for it in self.items for c in it["captions"]]
        self.cap_image = np.repeat(np.arange(len(self.items)), 5)
        z = np.load(self.root / "test_embeddings.npz")
        self.gal = {"zero-shot": (z["zs_img"].astype(np.float32), z["zs_txt"].astype(np.float32)),
                    "fine-tuned": (z["ft_img"].astype(np.float32), z["ft_txt"].astype(np.float32))}
        self.models = {}
        for name, folder in MODELS:
            model = CLIPModel.from_pretrained(self.root / folder).float().eval()
            self.models[name] = (model, CLIPProcessor.from_pretrained(self.root / folder))
        self._rank_cache = {}

    # ---------- encoding ----------
    @torch.no_grad()
    def encode_text(self, name, texts):
        model, proc = self.models[name]
        t = proc.tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt")
        v = model.get_text_features(**t)
        return torch.nn.functional.normalize(v, dim=-1).numpy()

    @torch.no_grad()
    def encode_image(self, name, pil):
        model, proc = self.models[name]
        px = proc.image_processor(pil.convert("RGB"), return_tensors="pt")["pixel_values"]
        v = model.get_image_features(pixel_values=px)
        return torch.nn.functional.normalize(v, dim=-1).numpy()

    def thumb(self, i):
        return Image.open(self.root / "thumbs" / self.items[i]["file"])

    # ---------- search ----------
    def text_to_image(self, query, k=5):
        """{model: (indices, scores)} of the top-k test photos for a typed caption."""
        out = {}
        for name in self.models:
            q = self.encode_text(name, [query])[0]
            s = self.gal[name][0] @ q
            idx = np.argsort(-s)[:k]
            out[name] = (idx, s[idx])
        return out

    def image_to_text(self, photo, k=5):
        """{model: (caption indices, scores)}; photo is a test index (uses the stored embedding) or a PIL image."""
        out = {}
        for name in self.models:
            q = self.gal[name][0][photo] if isinstance(photo, (int, np.integer)) else self.encode_image(name, photo)[0]
            s = self.gal[name][1] @ q
            idx = np.argsort(-s)[:k]
            out[name] = (idx, s[idx])
        return out

    # ---------- ranks and metrics from the stored embeddings ----------
    def t2i_ranks(self, name):
        """0-based rank of the correct photo for each of the 15,000 test captions."""
        key = ("t2i", name)
        if key not in self._rank_cache:
            img, txt = self.gal[name]
            r = np.empty(len(txt), dtype=np.int64)
            for a in range(0, len(txt), 1000):
                s = txt[a:a + 1000] @ img.T
                gt = s[np.arange(len(s)), self.cap_image[a:a + 1000]]
                r[a:a + 1000] = (s > gt[:, None]).sum(1)
            self._rank_cache[key] = r
        return self._rank_cache[key]

    def i2t_ranks(self, name):
        """0-based rank of the best correct caption for each of the 3,000 test photos."""
        key = ("i2t", name)
        if key not in self._rank_cache:
            img, txt = self.gal[name]
            r = np.empty(len(img), dtype=np.int64)
            for a in range(0, len(img), 500):
                s = img[a:a + 500] @ txt.T
                best = np.stack([s[j, 5 * (a + j):5 * (a + j) + 5].max() for j in range(len(s))])
                r[a:a + 500] = (s > best[:, None]).sum(1)
            self._rank_cache[key] = r
        return self._rank_cache[key]

    def metrics(self):
        """Test-set recall, rsum and MRR for both models, computed live from the stored embeddings."""
        rows = {}
        for name in self.models:
            i, t = self.i2t_ranks(name), self.t2i_ranks(name)
            row = {}
            for lab, r in (("image to text", i), ("text to image", t)):
                for k in (1, 5, 10):
                    row[f"{lab} R@{k}"] = 100 * float((r < k).mean())
                row[f"{lab} MRR"] = 100 * float((1.0 / (r + 1)).mean())
            row["rsum"] = sum(row[f"{d} R@{k}"] for d in ("image to text", "text to image") for k in (1, 5, 10))
            rows[name] = row
        return rows

    _VAGUE = re.compile(r"\b(picture|photo|image|object|thing|something|this is|there is|there are)\b", re.I)

    def descriptive(self, c):
        """A caption worth showing on screen: 6 to 16 words and not a generic filler sentence."""
        t = self.captions[c]
        return 6 <= len(t.split()) <= 16 and not self._VAGUE.search(t)

    def showcase(self, n=12):
        """Descriptive captions where fine-tuning helped most (zero-shot rank >= 5, fine-tuned rank 1) and hurt most
        (zero-shot rank 1). Counts over all 15,000 captions are reported separately."""
        z, f = self.t2i_ranks("zero-shot"), self.t2i_ranks("fine-tuned")
        ok = np.array([self.descriptive(c) for c in range(len(z))])
        better = np.where((f == 0) & (z >= 5) & ok)[0]
        better = better[np.argsort(-z[better])][:n]
        worse = np.where((z == 0) & ok)[0]
        worse = worse[np.argsort(-f[worse])][:n]
        return better, worse, z, f

    # ---------- HTML views ----------
    def _photo_cell(self, i, rank, score, correct=None):
        border = f"3px solid {GOOD}" if correct else "3px solid transparent"
        tag = f"<div style='color:{GOOD};font-weight:600'>correct photo</div>" if correct else ""
        return (f"<td style='padding:4px;text-align:center;vertical-align:top'>"
                f"<img src='data:image/jpeg;base64,{_b64(self.thumb(i))}' style='border:{border};border-radius:4px'>"
                f"<div style='font-size:12px;color:{MUTED}'>#{rank}  score {score:.3f}</div>{tag}</td>")

    def view_text_to_image(self, query, gt_image=None, k=5):
        res = self.text_to_image(query, k)
        rows = [f"<div style='font:16px sans-serif;margin:6px 0'>Query: <b>&ldquo;{html.escape(query)}&rdquo;</b></div>",
                "<table style='border-collapse:collapse'>"]
        for name, (idx, sc) in res.items():
            hit = [r for r, i in enumerate(idx, 1) if gt_image is not None and i == gt_image]
            note = (f"correct photo at #{hit[0]}" if hit else "correct photo not in top 5") if gt_image is not None else ""
            colour = GOOD if hit else BAD
            cells = "".join(self._photo_cell(i, r, s, gt_image is not None and i == gt_image)
                            for r, (i, s) in enumerate(zip(idx, sc), 1))
            rows.append(f"<tr><td style='font:600 15px sans-serif;padding-right:12px;vertical-align:middle;width:92px'>{name}"
                        f"<div style='font:12px sans-serif;color:{colour}'>{note}</div></td>{cells}</tr>")
        rows.append("</table>")
        return "".join(rows)

    def view_image_to_text(self, photo, k=5):
        res = self.image_to_text(photo, k)
        is_test = isinstance(photo, (int, np.integer))
        pil = self.thumb(int(photo)) if is_test else photo
        cols = []
        for name, (idx, sc) in res.items():
            lines = []
            for r, (c, s) in enumerate(zip(idx, sc), 1):
                ok = is_test and self.cap_image[c] == photo
                style = f"background:#dcfce7;border-left:4px solid {GOOD}" if ok else "border-left:4px solid transparent"
                lines.append(f"<div style='padding:4px 8px;margin:3px 0;{style}'>#{r} {html.escape(self.captions[c])}"
                             f" <span style='color:{MUTED};font-size:12px'>({s:.3f})</span></div>")
            hits = sum(1 for c in idx if is_test and self.cap_image[c] == photo)
            sub = f"{hits} of the top {k} are correct captions" if is_test else ""
            cols.append(f"<td style='vertical-align:top;text-align:left;padding:0 14px;width:420px'><div style='font:600 15px sans-serif'>{name}</div>"
                        f"<div style='font:12px sans-serif;color:{MUTED}'>{sub}</div>"
                        f"<div style='font:14px sans-serif'>{''.join(lines)}</div></td>")
        return (f"<table><tr><td style='vertical-align:top;text-align:left;padding-right:16px'>"
                f"<img src='data:image/jpeg;base64,{_b64(pil, 240)}' style='border-radius:4px'></td>{''.join(cols)}</tr></table>")
