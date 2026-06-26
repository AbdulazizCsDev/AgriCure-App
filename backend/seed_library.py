"""
Seed the AgriCure similarity library (Supabase pgvector) with CLIP embeddings.

One script supersedes the old seed_import.py / seed_supported.py. It reads a
dataset (a .zip or a folder of `Plant_Disease/<image>` subfolders), computes a
CLIP embedding for each image *with the same code path used at inference*
(plant_gate.PlantGate.embed), and inserts it into `labeled_images` so the
similarity search can match it.

Usage
-----
  # See exactly what WOULD be seeded — no DB, no models, no writes:
  python seed_library.py --data path/to/dataset.zip --dry-run

  # Seed only the 16 supported classes (needs SUPABASE_* in .env):
  python seed_library.py --data path/to/training_dataset.zip --which supported

  # Seed the open-set / unknown classes (extends coverage beyond the 16):
  python seed_library.py --data path/to/agro-mind.zip --which unknown

Options
-------
  --data / $AGRICURE_SEED_ZIP   dataset .zip OR a folder (required)
  --which {all,supported,unknown}   default: all
        supported = the 16 trained classes (config.PLANT_DISEASES, + merges)
        unknown   = everything NOT in the trained set (open-set library)
  --dry-run     scan + print a coverage report, write nothing
  --limit N     cap images per class (quick partial seed)

Embeddings
----------
CLIP ViT-B-32 (512-d) only — the primary retrieval space queried by
`match_labeled_images()`. The old `embed_resnet` (2048-d) column was never read
by any query, so it is no longer written. See backend/README.md.

A real run needs SUPABASE_URL + SUPABASE_SERVICE_KEY in .env; --dry-run needs
neither (no torch / CLIP / Supabase required).
"""

import argparse
import hashlib
import io
import os
import re
import sys
import time
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config_v4 as config

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def supported_set() -> set:
    """The trained/supported (plant, disease) pairs — incl. merge sources & known renames."""
    s = {(norm(p), norm(d)) for p, ds in config.PLANT_DISEASES.items() for d in ds}
    for src in config.CLASS_MERGES:
        p, d = src.split(config.FOLDER_SEP, 1)
        s.add((norm(p), norm(d)))
    s |= {("tomato", "early blight"), ("tomato", "late blight"),
          ("corn-maize", "brown spot"), ("apple tree", "red spider mite")}
    return s


def parse_class(parent: str):
    """`Tomato--Leaf Mold` or `Tomato_Leaf Mold` -> ('Tomato', 'Leaf Mold').

    Accepts both the canonical `Plant--Disease` (config.FOLDER_SEP) and the
    open-set `Plant_Disease` folder conventions.
    """
    if config.FOLDER_SEP in parent:          # "Plant--Disease"
        plant, disease = parent.split(config.FOLDER_SEP, 1)
    elif "_" in parent:                       # "Plant_Disease"
        plant, disease = parent.split("_", 1)
    else:
        plant, disease = parent, ""
    return plant.strip(), disease.strip()


# ── Dataset readers: a .zip or a plain folder, same interface ──────────────────
def iter_zip(path):
    z = zipfile.ZipFile(path)
    for n in z.namelist():
        if not n.lower().endswith(IMG_EXT):
            continue
        parts = n.split("/")
        if len(parts) < 2:
            continue
        yield parts[-2], n, lambda n=n: z.read(n)


def iter_folder(root):
    for dirpath, _dirs, files in os.walk(root):
        parent = os.path.basename(dirpath)
        for fn in files:
            if not fn.lower().endswith(IMG_EXT):
                continue
            full = os.path.join(dirpath, fn)
            yield parent, full, lambda full=full: open(full, "rb").read()


def iter_dataset(path):
    if os.path.isdir(path):
        return iter_folder(path)
    if zipfile.is_zipfile(path):
        return iter_zip(path)
    sys.exit(f"--data must be a .zip or a folder: {path}")


def select(plant, disease, which, supported, only_plants=None):
    if only_plants is not None and norm(plant) not in only_plants:
        return False
    pair = (norm(plant), norm(disease))
    if which == "supported":
        return pair in supported
    if which == "unknown":
        return pair not in supported
    return True  # all


def scan(path, which, supported, only_plants=None):
    """Group selected images by class without loading models. Returns {class: [readers]}."""
    by_class = {}
    for parent, name, reader in iter_dataset(path):
        plant, disease = parse_class(parent)
        if not select(plant, disease, which, supported, only_plants):
            continue
        by_class.setdefault(parent, []).append((name, reader))
    return by_class


def coverage_report(by_class, supported):
    total_imgs = sum(len(v) for v in by_class.values())
    print(f"\nSelected: {len(by_class)} classes, {total_imgs} images\n")

    # How well do we cover the 16 supported plants?
    present = {}
    for parent, items in by_class.items():
        p, _ = parse_class(parent)
        present[norm(p)] = present.get(norm(p), 0) + len(items)
    print("Supported-plant coverage (16 trained species):")
    missing = []
    for plant in config.PLANT_DISEASES:
        n = present.get(norm(plant), 0)
        mark = "ok " if n else "-- "
        if not n:
            missing.append(plant)
        print(f"  [{mark}] {plant:<16} {n} image(s)")
    if missing:
        print(f"\n  {len(missing)} supported plant(s) have NO images here: {', '.join(missing)}")
        print("  -> point --data at the training dataset to seed those.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Seed the AgriCure similarity library.")
    ap.add_argument("--data", default=os.environ.get("AGRICURE_SEED_ZIP"),
                    help="dataset .zip or folder (or set $AGRICURE_SEED_ZIP)")
    ap.add_argument("--which", choices=["all", "supported", "unknown"], default="all")
    ap.add_argument("--plants", default=None,
                    help="comma-separated plant names to restrict to (e.g. 'Cassava,Squash')")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="cap images per class (0 = no cap)")
    args = ap.parse_args()

    if not args.data:
        ap.error("--data is required (path to a .zip or folder, or set $AGRICURE_SEED_ZIP)")
    if not os.path.exists(args.data):
        sys.exit(f"--data not found: {args.data}")

    only_plants = None
    if args.plants:
        only_plants = {norm(p) for p in args.plants.split(",") if p.strip()}

    supported = supported_set()
    print(f"Scanning {args.data}  (which={args.which}"
          f"{', plants=' + args.plants if args.plants else ''}) ...", flush=True)
    by_class = scan(args.data, args.which, supported, only_plants)
    if args.limit:
        by_class = {k: v[: args.limit] for k, v in by_class.items()}
    coverage_report(by_class, supported)

    if args.dry_run:
        print("Dry run - nothing written. Re-run without --dry-run to seed.")
        return

    # ── Real run: lazy-load DB + CLIP only now (keeps --dry-run dependency-free) ──
    import db
    if not db.ENABLED:
        sys.exit("Supabase is not configured — set SUPABASE_URL and SUPABASE_SERVICE_KEY "
                 "in backend/.env before a real seed run.")
    import torch
    import plant_gate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP (same embed path as inference) ...", flush=True)
    gate = plant_gate.PlantGate(device)

    from PIL import Image
    seen, kept, skipped, errs = set(), 0, 0, 0
    total = sum(len(v) for v in by_class.values())
    t0 = time.time()
    i = 0
    for parent, items in by_class.items():
        plant, disease = parse_class(parent)
        for name, reader in items:
            i += 1
            try:
                raw = reader()
                sha = hashlib.sha256(raw).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                if db.exists_sha("labeled_images", sha):  # idempotent: skip already-seeded
                    skipped += 1
                    continue
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                emb = gate.embed(pil)[0].detach().cpu().tolist()
                buf = io.BytesIO(); pil.save(buf, "JPEG", quality=90)
                path = f"seed/{sha}.jpg"
                url = db.upload_image(buf.getvalue(), path, "image/jpeg")
                db.insert("labeled_images", {
                    "plant": plant, "disease": disease,
                    "expert_label": f"{plant} | {disease}", "source": "seed",
                    "image_url": url, "image_path": path, "image_sha256": sha,
                    "annotator": "seed-library", "embed_clip": db._vec(emb),
                })
                kept += 1
            except Exception as e:
                errs += 1
                if errs <= 12:
                    print(f"  ERR [{parent}] {str(e)[:120]}", flush=True)
            if i % 25 == 0:
                print(f"{i}/{total}  kept={kept} skipped={skipped} errs={errs}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    print(f"\nDONE  kept={kept}  skipped(existing)={skipped}  errs={errs}  "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
