"""Loading the 10K split and the CLIP model, shared by the zero-shot and fine-tuning scripts."""
import json
from pathlib import Path

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_splits(split_dir="data/task10k", limit=None):
    """{'train': [...], 'test': [...], 'val': [...]} of items with id, path, captions, labels."""
    out = {}
    for k in ("train", "test", "val"):
        with open(Path(split_dir) / f"{k}.json") as f:
            items = json.load(f)
        out[k] = items[:limit] if limit else items
    return out


def load_classes(annotations="data/annotations/instances_val2017.json"):
    """(category ids, names) of the 80 COCO classes, sorted by id."""
    with open(annotations) as f:
        cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    return [c["id"] for c in cats], [c["name"] for c in cats]


def load_clip(device, name=MODEL_NAME):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(name).to(device), CLIPProcessor.from_pretrained(name)
