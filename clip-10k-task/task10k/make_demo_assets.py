"""Build the offline assets of the live demo. Run on the machine that has the COCO images and the models:

    python -m task10k.make_demo_assets --out demo_assets

Writes demo_assets/ with 256 px thumbnails of the 3,000 test images, their captions, the saved test
embeddings of the zero-shot and the fine-tuned model (so the gallery needs no encoding), and both models
as fp16 Hugging Face folders. Copy the folder to the laptop; the demo then runs offline on CPU.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from task10k.common import MODEL_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/task10k")
    p.add_argument("--zero-emb", default="results/task10k/emb/zeroshot.npz")
    p.add_argument("--ft-emb", default="results/task10k/final_demo/emb_s0.npz")
    p.add_argument("--ft-model", default="ckpt/clip_ft_s0")
    p.add_argument("--out", default="demo_assets")
    p.add_argument("--thumb", type=int, default=256, help="longest side of a thumbnail in pixels")
    a = p.parse_args()

    out = Path(a.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    with open(Path(a.split_dir) / "test.json") as f:
        items = json.load(f)
    meta = []
    for it in items:
        assert len(it["captions"]) == 5, f"image {it['id']} has {len(it['captions'])} captions"
        img = Image.open(it["path"]).convert("RGB")
        img.thumbnail((a.thumb, a.thumb))
        name = f"{it['id']}.jpg"
        img.save(out / "thumbs" / name, quality=88)
        meta.append({"id": it["id"], "file": name, "captions": it["captions"]})
    with open(out / "test_items.json", "w") as f:
        json.dump(meta, f)
    print(f"{len(meta)} thumbnails")

    z, ft = np.load(a.zero_emb), np.load(a.ft_emb)
    for k in ("img_test", "txt_test"):
        assert z[k].shape == ft[k].shape, (k, z[k].shape, ft[k].shape)
    assert z["img_test"].shape[0] == len(meta) and z["txt_test"].shape[0] == 5 * len(meta)
    np.savez_compressed(out / "test_embeddings.npz", zs_img=z["img_test"], zs_txt=z["txt_test"],
                        ft_img=ft["img_test"], ft_txt=ft["txt_test"])

    from transformers import CLIPModel, CLIPProcessor
    base = CLIPModel.from_pretrained(MODEL_NAME)
    base.save_pretrained(out / "clip_zeroshot", safe_serialization=True,
                         state_dict={k: v.detach().half() for k, v in base.state_dict().items()})
    CLIPProcessor.from_pretrained(MODEL_NAME).save_pretrained(out / "clip_zeroshot")
    shutil.copytree(a.ft_model, out / "clip_ft_s0", dirs_exist_ok=True)
    print("models saved; total size:", sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // 2**20, "MB")


if __name__ == "__main__":
    main()
