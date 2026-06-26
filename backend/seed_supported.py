"""
Seed the 16 SUPPORTED plants into the similarity library.

seed_import.py imported only the new/unknown classes (and skipped the trained
ones). This imports the opposite set — the supported-plant images — so the
similarity search can recognise supported plants too (fixes the case where the
true plant simply wasn't in the library).
"""

import io, os, re, sys, json, time, hashlib, zipfile
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


def load_env(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); d[k.strip()] = v.strip()
    return d

env = load_env(os.path.join(BASE, ".env"))
URL = env["SUPABASE_URL"].rstrip("/"); KEY = env["SUPABASE_SERVICE_KEY"]
REST, STORAGE, BUCKET = f"{URL}/rest/v1", f"{URL}/storage/v1", "review-images"
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


def norm(s): return re.sub(r"\s+", " ", s.strip().lower())

# supported set = the trained classes (+ merge sources + renamed duplicates)
supported = {(norm(p), norm(d)) for p, ds in config.PLANT_DISEASES.items() for d in ds}
for src in config.CLASS_MERGES:
    p, d = src.split(config.FOLDER_SEP, 1)
    supported.add((norm(p), norm(d)))
supported |= {("tomato", "early blight"), ("tomato", "late blight"),
              ("corn-maize", "brown spot"), ("apple tree", "red spider mite")}

print("loading CLIP ...", flush=True)
clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()
print("loading ResNet50 backbone ...", flush=True)
with open(config.STAGE1_CLASSES) as f:
    plant_names = json.load(f)["plant_names"]
res = load_model(config.STAGE1_MODEL, len(plant_names))
res.fc = torch.nn.Identity(); res = res.to(DEVICE).eval()


@torch.no_grad()
def embed(pil):
    c = clip_model.encode_image(clip_pre(pil).unsqueeze(0).to(DEVICE))
    c = (c / c.norm(dim=-1, keepdim=True))[0].cpu().tolist()
    r = res(tf(pil).unsqueeze(0).to(DEVICE))[0].cpu().tolist()
    return c, r

def vec(l): return "[" + ",".join(f"{x:.6f}" for x in l) + "]"

z = zipfile.ZipFile(ZIP)
items = [n for n in z.namelist() if n.lower().endswith(IMG_EXT)]
seen = set(); kept = skipped = errs = 0
t0 = time.time()
for i, n in enumerate(items, 1):
    parent = n.split("/")[-2]
    plant, disease = parent.split("_", 1) if "_" in parent else (parent, "")
    if (norm(plant), norm(disease)) not in supported:   # only supported plants
        skipped += 1; continue
    try:
        raw = z.read(n); sha = hashlib.sha256(raw).hexdigest()
        if sha in seen: continue
        seen.add(sha)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        c, r = embed(pil)
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=90)
        path = f"supported/{sha}.jpg"
        requests.post(f"{STORAGE}/object/{BUCKET}/{path}",
                      headers={**H, "Content-Type": "image/jpeg", "x-upsert": "true"},
                      data=buf.getvalue(), timeout=60).raise_for_status()
        requests.post(f"{REST}/labeled_images",
                      headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"},
                      json={"plant": plant, "disease": disease,
                            "expert_label": f"{plant} | {disease}", "source": "seed",
                            "image_url": f"{STORAGE}/object/public/{BUCKET}/{path}",
                            "image_path": path, "image_sha256": sha,
                            "annotator": "seed-supported", "embed_clip": vec(c), "embed_resnet": vec(r)},
                      timeout=60).raise_for_status()
        kept += 1
    except Exception as e:
        errs += 1
        if errs <= 12: print("  ERR", parent, str(e)[:120], flush=True)
    if i % 25 == 0:
        print(f"{i}/{len(items)} kept={kept} skipped={skipped} errs={errs} {time.time()-t0:.0f}s", flush=True)
print(f"\nDONE kept={kept} skipped={skipped} errs={errs} ({time.time()-t0:.0f}s)", flush=True)
