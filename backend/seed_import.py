"""
Seed import — load the 'agro-mind' dataset (new/unknown classes) into the
Supabase similarity memory.

For each kept image:
  • compute CLIP (512-d) + ResNet50 penultimate (2048-d, trained backbone)
  • upload the JPEG to the `review-images` bucket
  • insert a row into `labeled_images` (source='seed')

Trained classes (and a few renamed duplicates) are skipped — the CNN already
handles those.
"""

import os, io, sys, re, json, time, hashlib, zipfile
import requests
import torch
import open_clip
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config_v4 as config
from inference_v4 import load_model, tf, DEVICE

ZIP = r"C:\Users\Aziz\Downloads\agro-mind-20260622T111026Z-3-001.zip"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ── env ───────────────────────────────────────────────────────────────────────
def load_env(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d

env = load_env(os.path.join(BASE, ".env"))
URL = env["SUPABASE_URL"].rstrip("/")
KEY = env["SUPABASE_SERVICE_KEY"]
if not KEY or KEY.startswith("PASTE_"):
    sys.exit("SUPABASE_SERVICE_KEY is not set in .env")
REST, STORAGE, BUCKET = f"{URL}/rest/v1", f"{URL}/storage/v1", "review-images"
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


# ── what counts as 'already trained' (skip these) ─────────────────────────────
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())

skip = {(norm(p), norm(d)) for p, ds in config.PLANT_DISEASES.items() for d in ds}
for src in config.CLASS_MERGES:                      # merge sources were trained too
    p, d = src.split(config.FOLDER_SEP, 1)
    skip.add((norm(p), norm(d)))
skip |= {                                            # renamed duplicates of trained classes
    ("tomato", "early blight"), ("tomato", "late blight"),
    ("corn-maize", "brown spot"), ("apple tree", "red spider mite"),
}


# ── models ────────────────────────────────────────────────────────────────────
print("loading CLIP ViT-B-32 ...", flush=True)
clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()

print("loading ResNet50 backbone (trained Stage-1) ...", flush=True)
with open(config.STAGE1_CLASSES) as f:
    plant_names = json.load(f)["plant_names"]
res = load_model(config.STAGE1_MODEL, len(plant_names))
res.fc = torch.nn.Identity()                          # drop classifier → 2048-d features
res = res.to(DEVICE).eval()


@torch.no_grad()
def embed(pil):
    c = clip_model.encode_image(clip_pre(pil).unsqueeze(0).to(DEVICE))
    c = (c / c.norm(dim=-1, keepdim=True))[0].cpu().tolist()
    r = res(tf(pil).unsqueeze(0).to(DEVICE))[0].cpu().tolist()
    return c, r

def vec(lst): return "[" + ",".join(f"{x:.6f}" for x in lst) + "]"


# ── run ───────────────────────────────────────────────────────────────────────
z = zipfile.ZipFile(ZIP)
items = [n for n in z.namelist() if n.lower().endswith(IMG_EXT)]
print(f"{len(items)} images in archive\n", flush=True)

seen = set()
kept = skipped = errs = 0
t0 = time.time()

for i, n in enumerate(items, 1):
    parent = n.split("/")[-2]                         # agro-mind/<class>/<file>
    plant, disease = parent.split("_", 1) if "_" in parent else (parent, "")
    if (norm(plant), norm(disease)) in skip:
        skipped += 1
        continue
    try:
        raw = z.read(n)
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)

        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        c, r = embed(pil)

        buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=90)
        path = f"seed/{sha}.jpg"
        up = requests.post(f"{STORAGE}/object/{BUCKET}/{path}",
                           headers={**H, "Content-Type": "image/jpeg", "x-upsert": "true"},
                           data=buf.getvalue(), timeout=60)
        up.raise_for_status()

        row = {
            "plant": plant, "disease": disease,
            "expert_label": f"{plant} | {disease}",
            "source": "seed",
            "image_url": f"{STORAGE}/object/public/{BUCKET}/{path}",
            "image_path": path, "image_sha256": sha,
            "annotator": "seed-import",
            "embed_clip": vec(c), "embed_resnet": vec(r),
        }
        ins = requests.post(f"{REST}/labeled_images",
                            headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"},
                            json=row, timeout=60)
        ins.raise_for_status()
        kept += 1
    except Exception as e:
        errs += 1
        if errs <= 12:
            print(f"  ERR [{parent}] {n.split('/')[-1]}: {str(e)[:140]}", flush=True)

    if i % 25 == 0:
        print(f"{i}/{len(items)}  kept={kept} skipped={skipped} errs={errs}  {time.time()-t0:.0f}s", flush=True)

print(f"\nDONE  kept={kept}  skipped={skipped}  errs={errs}  ({time.time()-t0:.0f}s)", flush=True)
