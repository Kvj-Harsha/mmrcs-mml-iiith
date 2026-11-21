"""Fine-tune CLIP ViT-B/32 on the 5K train split; pick the epoch by val retrieval.

    # learning-rate sweep: val curves only, test and train are not scored
    python -m task10k.finetune --lr 3e-6 --out results/task10k/sweep/lr3e-06_s0.json
    # final run: restore the best-val epoch, then score every split
    python -m task10k.finetune --lr 3e-6 --seed 0 --final --out results/task10k/final/s0.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from task10k.common import MODEL_NAME, load_classes, load_clip, load_splits
from task10k.metrics import embed, evaluate, ranks, retrieval_summary
from task10k.metrics_ext import save_embeddings
from training.clip_retrieval import multi_positive_loss
from training.data import PairDataset, sample_order, train_transform
from training.loop import lr_at, param_groups, set_seed


def val_rsum(model, processor, items, device):
    img, txt = embed(model, processor, items, device)
    s = retrieval_summary(*ranks(img, txt), n_boot=1)   # the interval is not used here
    return s["rsum"], s["i2t"]["R@1"], s["t2i"]["R@1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None, help="first N images of each split (gate run)")
    p.add_argument("--final", action="store_true", help="score all splits with the best-val epoch")
    p.add_argument("--split-dir", default="data/task10k")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--emb-out", default=None, help="embeddings of a final run (default results/task10k/emb/s<seed>.npz)")
    p.add_argument("--save-model", default=None,
                   help="final run only: save the best-val weights here (HF folder, fp16 safetensors) for the live demo")
    args = p.parse_args()
    if args.save_model and not args.final:
        p.error("--save-model needs --final (the best-val weights are only kept in a final run)")

    device = torch.device("cuda")
    set_seed(args.seed)
    splits = load_splits(args.split_dir, args.limit)
    train = splits["train"]
    train_ids = {it["id"] for it in train}
    assert not train_ids & {it["id"] for it in splits["val"]}, "train and val share an image"
    assert not train_ids & {it["id"] for it in splits["test"]}, "train and test share an image"
    cats, names = load_classes()

    model, processor = load_clip(device)
    tok = processor.tokenizer
    n_pairs = sum(len(it["captions"]) for it in train)
    steps_per_epoch = n_pairs // args.batch
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    transform = train_transform(224, 0.5)

    curve, best, best_state, bad, step = [], -1.0, None, 0, 0
    t0 = time.time()
    print(f"train {len(train)} images / {n_pairs} pairs, {steps_per_epoch} steps per epoch, lr {args.lr}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = sample_order(n_pairs, steps_per_epoch * args.batch, args.seed * 1000 + epoch)
        ds = PairDataset(train, order, transform, lambda c: c)
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, drop_last=True,
                                             num_workers=args.workers, generator=g)
        losses = []
        for image, caption, idx in loader:
            lr = lr_at(step, total_steps, args.lr, args.warmup_frac)
            for grp in opt.param_groups:
                grp["lr"] = lr
            t = tok(list(caption), padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                img = F.normalize(model.get_image_features(pixel_values=image.to(device)).float(), dim=-1)
                txt = F.normalize(model.get_text_features(**t).float(), dim=-1)
                loss = multi_positive_loss(img, txt, model.logit_scale.exp().clamp(max=100.0), idx.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
        model.eval()
        with torch.no_grad():
            rs, i1, t1 = val_rsum(model, processor, splits["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rsum": rs, "val_i2t_R@1": i1,
               "val_t2i_R@1": t1, "lr_end": lr, "seconds": time.time() - t0}
        curve.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} val rsum {rs:.1f} (i2t R@1 {i1:.1f}, t2i R@1 {t1:.1f}) "
              f"{row['seconds']:.0f}s", flush=True)
        if rs > best:
            best, bad = rs, 0
            if args.final:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val gain for {args.patience} epochs", flush=True)
                break

    best_epoch = max(curve, key=lambda r: r["val_rsum"])["epoch"]
    result = {"model": MODEL_NAME, "stage": "final" if args.final else "sweep", "lr": args.lr, "seed": args.seed,
              "batch": args.batch, "weight_decay": args.weight_decay, "epochs_run": len(curve),
              "best_epoch": best_epoch, "best_val_rsum": best, "curve": curve, "train_images": len(train),
              "train_seconds": time.time() - t0}
    if args.final:
        model.load_state_dict(best_state)
        if args.save_model:
            Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
            # fp16 copy for the demo; the model itself stays fp32 so the scores match the delivered runs
            model.save_pretrained(args.save_model, safe_serialization=True,
                                  state_dict={k: v.detach().half() for k, v in model.state_dict().items()})
            processor.save_pretrained(args.save_model)
            print(f"saved best-val weights (epoch {best_epoch}) to {args.save_model}", flush=True)
        with torch.no_grad():
            result["eval"], emb = evaluate(model, processor, splits, names, cats, device, return_embeddings=True)
            result["eval"].update({"class_names": names, "category_ids": cats})
        save_embeddings(args.emb_out or f"results/task10k/emb/s{args.seed}.npz", emb)
        for k, r in result["eval"]["retrieval"].items():
            print(f"{k:7s} rsum {r['rsum']:.1f}  i2t R@1 {r['i2t']['R@1']:.1f}  t2i R@1 {r['t2i']['R@1']:.1f}")
        for k, c in result["eval"]["classification"].items():
            print(f"{k:7s} mAP {100 * c['mAP']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
