"""Build notebooks/coco10k_clip_live_demo.ipynb, the live-demo continuation of the main notebook.

    python -m task10k.make_demo_notebook
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/coco10k_clip_live_demo.ipynb

Needs demo_assets/ (made by task10k.make_demo_assets). Prepared cases are read from presentation/demo_cases.json
if it exists: {"t2i": [caption indices], "i2t": [photo indices], "failure": caption index}.
"""
import json
from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks/coco10k_clip_live_demo.ipynb")
CASES = Path("presentation/demo_cases.json")
cases = json.load(open(CASES)) if CASES.exists() else {"t2i": [], "i2t": [], "failure": None}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Live demo: zero-shot vs fine-tuned CLIP, side by side

Continuation of `coco10k_clip_zeroshot_vs_finetuned.ipynb`: the same CLIP ViT-B/32, the same 3,000 test photos
(15,000 captions) that neither model saw in training. **Zero-shot** is the released model; **fine-tuned** is the model after
training on the 5,000 class-balanced train photos (seed 0, learning rate 1e-6, best-validation epoch).

**First run Kernel > Restart and Run All Cells (about 20 s); the search boxes and buttons only appear after the cells have run.**

Everything runs offline on CPU. The gallery of 3,000 photos and 15,000 captions is embedded once in advance, so a query costs one
encoding per model.

| Section | What it shows |
|---|---|
| 1 | The two models and a check that they match the saved results |
| Try it yourself | **Type any caption** or **upload any photo** and compare both models |
| 2 | **Text to image:** type a caption, get the top 5 photos from each model |
| 3 | **Image to text:** pick a test photo (or upload your own), get the top 5 captions from each model |
| 4 | Where fine-tuning helped most, and where it hurt |
| 5 | The headline numbers, recomputed live |
""")

code("""
import os, sys, io, json
root = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(root); sys.path.insert(0, root)
import numpy as np, pandas as pd
import ipywidgets as w
from PIL import Image
from IPython.display import HTML, display
from task10k.demo import Demo
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 110)
demo = Demo("demo_assets")
print(f"{len(demo.items)} test photos, {len(demo.captions)} captions, models: {list(demo.models)}")
""")

md("""
## 1. Check: the live models reproduce the saved results

Encode a test caption with each loaded model and compare with the stored gallery embedding (cosine close to 1 means the
saved weights are the ones that produced the reported numbers).
""")

code("""
for name in demo.models:
    live = demo.encode_text(name, [demo.captions[0], demo.captions[7]])
    stored = demo.gal[name][1][[0, 7]]
    print(f"{name:11s} cosine(live, stored) for 2 captions:", np.round((live * stored).sum(1), 4))
""")

md("""
## Try it yourself: type a query, or search with a photo

Two tabs. **Type a caption** (any sentence, nothing from the test set needed) and get the top photos from the 3,000-photo gallery
for both models. **Search with a photo** (upload any image from this laptop, or pick a random test photo) and get the best matching
captions from the 15,000-caption gallery. The first search after opening can take a few seconds.
""")

code("""
from PIL import ImageOps
try_rng = np.random.default_rng(7)
try_k = w.IntSlider(value=5, min=3, max=8, description="Show top", style={"description_width": "60px"}, layout=w.Layout(width="260px"))

# ---- tab 1: text -> image ----
try_text = w.Text(placeholder="type a caption, e.g. a dog sleeping on a couch  (Enter to search)", layout=w.Layout(width="620px"))
try_text.continuous_update = False
try_go = w.Button(description="Search photos", button_style="primary", icon="search")
try_out_t = w.Output()

def try_search_text(_=None):
    text = try_text.value.strip()
    with try_out_t:
        try_out_t.clear_output(wait=True)
        if not text:
            print("Type a caption first.")
            return
        print("searching ...")
        html_ = demo.view_text_to_image(text, k=try_k.value)
        try_out_t.clear_output(wait=True)
        display(HTML(html_))

try_text.observe(lambda ch: try_search_text(), "value")
try_go.on_click(try_search_text)

# ---- tab 2: image -> text ----
try_up = w.FileUpload(accept="image/*", multiple=False, description="Upload a photo", button_style="primary", icon="upload")
try_rand = w.Button(description="Random test photo", icon="random")
try_out_i = w.Output()

def try_show_photo(photo):
    with try_out_i:
        try_out_i.clear_output(wait=True)
        print("searching ...")
        html_ = demo.view_image_to_text(photo, k=try_k.value)
        try_out_i.clear_output(wait=True)
        display(HTML(html_))

def try_on_upload(change):
    files = try_up.value
    if not files:
        return
    f = files[0] if isinstance(files, (tuple, list)) else list(files.values())[0]
    try_show_photo(ImageOps.exif_transpose(Image.open(io.BytesIO(bytes(f["content"])))))

try_up.observe(try_on_upload, "value")
try_rand.on_click(lambda _: try_show_photo(int(try_rng.integers(len(demo.items)))))

tabs = w.Tab(children=[w.VBox([w.HBox([try_text, try_go]), try_out_t]),
                       w.VBox([w.HBox([try_up, try_rand]), try_out_i])])
tabs.set_title(0, "Type a caption")
tabs.set_title(1, "Search with a photo")
display(try_k, tabs)
""")

md("""
## 2. Text to image: type a caption

Each row is one model's top 5 photos for the same typed caption. If the caption comes from the test set, the photo it
was written for is outlined in green.
""")

# static prepared examples, then the widget
t2i = cases["t2i"]
if t2i:
    code(f"""
PREPARED_T2I = {t2i}
for c in PREPARED_T2I[:2]:
    display(HTML(demo.view_text_to_image(demo.captions[c], gt_image=int(demo.cap_image[c]))))
""")
else:
    code("""
better, worse, z_rank, f_rank = demo.showcase()
for c in better[:2]:
    display(HTML(demo.view_text_to_image(demo.captions[c], gt_image=int(demo.cap_image[c]))))
""")

code("""
rng = np.random.default_rng(0)
q = w.Text(placeholder="type a caption, then Enter or Search", layout=w.Layout(width="560px"))
go = w.Button(description="Search", button_style="primary")
rnd = w.Button(description="Random test caption")
out_t2i = w.Output()
state = {"gt": None}

def run_t2i(_=None):
    text = q.value.strip()
    with out_t2i:
        out_t2i.clear_output(wait=True)
        if text:
            display(HTML(demo.view_text_to_image(text, gt_image=state["gt"])))

def on_text(change):
    state["gt"] = None
    run_t2i()

def on_random(_):
    c = int(rng.integers(len(demo.captions)))
    state["gt"] = int(demo.cap_image[c])
    q.unobserve(on_text, "value"); q.value = demo.captions[c]; q.observe(on_text, "value")
    run_t2i()

q.continuous_update = False
q.observe(on_text, "value"); go.on_click(lambda _: (state.update(gt=None), run_t2i())); rnd.on_click(on_random)
display(w.HBox([q, go, rnd]), out_t2i)
""")

md("""
## 3. Image to text: pick a photo

Each column is one model's top 5 captions for the same photo. For a test photo, the captions written for it
(5 per photo) are highlighted in green. You can also upload your own photo.
""")

i2t = cases["i2t"]
code(f"""
PREPARED_I2T = {i2t if i2t else '[0, 1]'}
display(HTML(demo.view_image_to_text(PREPARED_I2T[0])))
""")

code("""
rng2 = np.random.default_rng(1)
out_i2t = w.Output()
pick = w.Dropdown(options=[(f"test photo {i}", i) for i in PREPARED_I2T], description="Prepared:")
rnd_i = w.Button(description="Random test photo")
upload = w.FileUpload(accept="image/*", multiple=False, description="Upload a photo")

def show_i2t(photo):
    with out_i2t:
        out_i2t.clear_output(wait=True)
        display(HTML(demo.view_image_to_text(photo)))

pick.observe(lambda ch: show_i2t(int(ch["new"])), "value")
rnd_i.on_click(lambda _: show_i2t(int(rng2.integers(len(demo.items)))))

def on_upload(change):
    files = upload.value
    if not files:
        return
    f = files[0] if isinstance(files, (tuple, list)) else list(files.values())[0]
    show_i2t(Image.open(io.BytesIO(bytes(f["content"]))))

upload.observe(on_upload, "value")
display(w.HBox([pick, rnd_i, upload]), out_i2t)
""")

md("""
## 4. Where fine-tuning helped most, and where it hurt

For every test caption we know the rank of its photo under each model (1 = top). First table: descriptive captions whose
photo was at rank 5 or worse zero-shot and is first after fine-tuning. Second: captions that were first zero-shot and got
worse. (Vague captions such as "an object is in the picture" are left out of both tables; the counts below use all 15,000.)
""")

code("""
better, worse, z_rank, f_rank = demo.showcase(12)
def table(idx):
    df = pd.DataFrame({"caption": [demo.captions[c] for c in idx], "zero-shot rank": z_rank[idx] + 1,
                       "fine-tuned rank": f_rank[idx] + 1}, index=[int(c) for c in idx]).rename_axis("caption id")
    return df.style.set_properties(subset=["caption"], **{"text-align": "left"}).set_table_styles(
        [{"selector": "th.col_heading", "props": [("text-align", "left")]}])
print("helped most:"); display(table(better))
print("hurt most:");   display(table(worse))
n_up, n_down = int((f_rank < z_rank).sum()), int((f_rank > z_rank).sum())
print(f"Of {len(z_rank):,} test captions: photo ranked higher after fine-tuning for {n_up:,}, lower for {n_down:,}, "
      f"unchanged for {len(z_rank) - n_up - n_down:,}.")
""")

fail = cases["failure"]
code(f"""
FAILURE = {fail if fail is not None else 'int(worse[0])'}
display(HTML(demo.view_text_to_image(demo.captions[FAILURE], gt_image=int(demo.cap_image[FAILURE]))))
""")

md("""
## 5. The headline numbers, recomputed live

Computed here from the stored embeddings of the 3,000 test photos and 15,000 captions. The deck reports the mean of three
training seeds (rsum 447.1); this notebook holds seed 0, so its fine-tuned rsum is the seed-0 value from the results
file (446.2).
""")

code("""
m = demo.metrics()
live = pd.DataFrame(m).round(1)
live.index.name = "test split"
display(live)
res = "results/task10k"
saved = {"zero-shot": json.load(open(f"{res}/zeroshot.json"))["retrieval"]["test"]["rsum"],
         "fine-tuned": json.load(open(f"{res}/final/s0.json"))["eval"]["retrieval"]["test"]["rsum"]}
for name in demo.models:
    print(f"{name:11s} rsum live {m[name]['rsum']:.1f}  vs results file {saved[name]:.1f}")
    assert abs(m[name]["rsum"] - saved[name]) < 0.5, "live numbers do not match the saved results"
""")

md("""
**Reading it.** Same photos, same captions, same code: the only difference is the weights. Text to image gains the most,
and a few captions do get worse (section 4), which is why we report averages over all 15,000 captions and three seeds
instead of picking examples.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
