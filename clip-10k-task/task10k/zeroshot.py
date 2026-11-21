"""Zero-shot CLIP ViT-B/32 on the 10K split.

    python -m task10k.zeroshot                       # full run -> results/task10k/zeroshot.json
    python -m task10k.zeroshot --limit 200 --out results/task10k/zeroshot_gate.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from task10k.common import MODEL_NAME, load_classes, load_clip, load_splits
from task10k.metrics import evaluate
from task10k.metrics_ext import save_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/task10k")
    p.add_argument("--limit", type=int, default=None, help="first N images of each split (gate run)")
    p.add_argument("--out", default="results/task10k/zeroshot.json")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--emb-out", default="results/task10k/emb/zeroshot.npz")
    args = p.parse_args()

    device = torch.device("cuda")
    splits = load_splits(args.split_dir, args.limit)
    cats, names = load_classes()
    model, processor = load_clip(device)
    t0 = time.time()
    with torch.no_grad():
        result, emb = evaluate(model, processor, splits, names, cats, device, n_boot=args.boot,
                               return_embeddings=True)
    save_embeddings(args.emb_out, emb)
    result.update({"model": MODEL_NAME, "stage": "zeroshot", "seconds": time.time() - t0,
                   "class_names": names, "category_ids": cats})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for k, r in result["retrieval"].items():
        print(f"{k:7s} i2t R@1 {r['i2t']['R@1']:.1f} R@5 {r['i2t']['R@5']:.1f}  t2i R@1 {r['t2i']['R@1']:.1f} "
              f"R@5 {r['t2i']['R@5']:.1f}  rsum {r['rsum']:.1f}  medR {r['i2t']['medr']:.0f}/{r['t2i']['medr']:.0f}  "
              f"CLIPScore {r['clipscore']:.3f}")
    for k, c in result["classification"].items():
        print(f"{k:7s} mAP {100 * c['mAP']:.1f} (prior {100 * c['mean_class_prior']:.1f})  "
              f"F1 micro {100 * c['f1_micro']:.1f} macro {100 * c['f1_macro']:.1f}")


if __name__ == "__main__":
    main()
