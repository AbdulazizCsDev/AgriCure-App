---
title: AgriCure App
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AgriCure — Backend (FastAPI + ML pipeline + Web UI)

Backend service for **AgriCure**. Serves the hierarchical **ResNet50** plant-disease
pipeline (Stage 1 plant → Stage 2 disease) behind a REST API, with a CLIP plant-gate,
confidence-based abstention to a human-review queue, Supabase `pgvector` similarity
search, and LLM-generated care advice.

> This is the backend reference. For the project overview, architecture diagram, and
> repo layout, see the [root README](../README.md).

> **Note:** the running app's internal name/version is still `AgroCure v4` (FastAPI
> `title`, `config_v4.py`). Only branding/docs were renamed to AgriCure — the code
> module names were left untouched to avoid breaking the trained pipeline.

## Run (full mode — models loaded locally)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
cp .env.example .env            # optional: enables Supabase / LLM features
uvicorn main:app --port 8000
```

Open **http://localhost:8000** — drag in a leaf photo and get the diagnosis.

> The model weights (`models/*.pth`) are tracked with **Git LFS** (~630 MB). Run
> `git lfs install` before cloning so they download automatically.
> First request after startup loads 7 models into memory — give it a few seconds.
> Runs on GPU automatically if available, otherwise CPU.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/`                  | Landing page |
| GET  | `/app`               | Upload / diagnosis web UI |
| GET  | `/dashboard`         | Expert reviewer dashboard |
| GET  | `/requests`          | Review-queue page |
| GET  | `/health`            | Model + gate status |
| GET  | `/classes`           | Supported plants → diseases |
| POST | `/predict`           | multipart `file=<image>` → JSON diagnosis |
| GET  | `/api/stats`         | Dashboard statistics |
| GET  | `/api/pending`       | Scans awaiting expert review |
| GET  | `/api/taxonomy`      | Known plants/diseases (annotation dropdowns) |
| GET  | `/api/neighbors`     | Nearest verified images for a review case |
| POST | `/api/label`         | Submit an expert label (approves a review) |
| POST | `/api/skip`          | Skip / set status on a review case |
| POST | `/api/claim`         | Mark a case as actively under review |
| GET  | `/api/review-status` | Track a handed-to-expert case (pending → reviewed) |
| POST | `/api/reviewer-auth` | Validate the reviewer access code |
| GET  | `/api/advice`        | Cached LLM care guidance for a plant+disease |
| POST | `/api/chat`          | Diagnosis-aware follow-up chat (assistant "Sage") |

### Example

```bash
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```
```json
{
  "plant": "Tomato", "disease": "Blight",
  "label": "Tomato | Blight",
  "confidence": 0.93, "stage1_conf": 0.99, "stage2_conf": 0.94,
  "latency_ms": 41.2
}
```

## Configuration (`.env`)

Copy `.env.example` to `.env`. Every feature degrades safely if its keys are missing —
the API still runs (similarity, review, advice simply become no-ops).

| Variable | Required | Purpose |
|----------|----------|---------|
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | for review + similarity | Supabase project (server-side key only) |
| `OPENAI_API_KEY` | optional | LLM care advice & chat (omitted if unset) |
| `OPENAI_MODEL` | optional | defaults to `gpt-4o-mini` |
| `REVIEWER_PIN` | recommended | reviewer dashboard access code (default `314159`) |
| `MODEL_API_URL` | optional | enables light **proxy** mode (see below) |

## Non-plant rejection (plant gate)

Before diagnosing, every image passes through `plant_gate.py`. All checks share **one**
CLIP image embedding (encode once, reuse):

1. **CLIP zero-shot gate** — CLIP (open_clip, ViT-B-32) scores the image against
   "plant leaf" prompts vs "animal / person / hand / car / object / screenshot"
   prompts. It only passes if the plant group wins (`PLANT_THRESHOLD`). This is a
   direct *"is this a leaf?"* signal, so it accepts hard/ambiguous real leaves and
   rejects cats, hands, cars, faces — things a confidence threshold cannot separate.
2. **Confidence floor** — a last-resort net; rejects only if the plant classifier
   is below `STAGE1_MIN_CONF` (kept very low so real leaves are never rejected here).

### Healthy check
If the image is a plant, CLIP scores it against "healthy leaf" vs "sick leaf / leaf
with spots" prompts. If the healthy group wins `HEALTH_THRESHOLD` (0.70, set high on
purpose), the API returns `healthy: true` and skips disease diagnosis.

> ⚠️ CLIP is **weak** at healthy-vs-diseased (a fine, spot-level distinction). The
> threshold is biased so a *sick* leaf is rarely called healthy, at the cost of
> sometimes calling a healthy leaf "diseased". For reliable health detection, train a
> dedicated healthy/diseased classifier.

### Framing tip
CLIP also flags **wide / whole-plant shots** (vs a close-up single leaf) and adds a
non-blocking `framing_tip` to the response, steering users toward the close-up
single-leaf photos the model was trained on. Tune with `FRAMING_THRESHOLD` (0.50).

### Open-set + similarity (trust over guessing)
Stage-1 signals (confidence, top-1/top-2 margin) plus a CLIP "known-species" check
flag scans that may **not** be one of the 16 supported species — these are returned as
a best guess *and* queued for expert review. A confident `pgvector` similarity match
(`match_labeled_images`) that disagrees with the CNN is treated as evidence the CNN is
wrong. Below the confidence floor, AgriCure **abstains** and routes the scan to the
reviewer dashboard rather than showing a confident wrong answer.

Tune strictness at the top of `plant_gate.py` (`PLANT_THRESHOLD`, `STAGE1_MIN_CONF`,
`HEALTH_THRESHOLD`, `FRAMING_THRESHOLD`, plus the prompt lists).

> The CLIP weights (~350 MB) download once on first startup and are cached.

## Deployment modes

- **Full** (default) — `torch` + CLIP load locally; all features available. Used in dev
  and on a GPU host.
- **Proxy / light** — set `MODEL_API_URL` to a separate model service (e.g. a Hugging
  Face Space). The web app then forwards `/predict` to that service and skips loading
  `torch`/CLIP, so it runs on a tiny instance. This is what the `Dockerfile` builds
  (using `requirements-app.txt`, which has no `torch`).

## Supabase setup

The review queue, similarity library, taxonomy, and cached advice live in Supabase.
Apply the SQL in `supabase/migrations/` (in order) to a fresh project, then set
`SUPABASE_URL` / `SUPABASE_SERVICE_KEY` in `.env`. See
[`supabase/README.md`](supabase/README.md) and [`SHARE_DATABASE_GUIDE.md`](SHARE_DATABASE_GUIDE.md)
for the schema and the `pgvector` similarity function.

### Seeding the similarity library

`seed_library.py` populates `labeled_images` with CLIP embeddings so the similarity
search has something to match against. It reads a `.zip` or a folder of
`Plant_Disease/<image>` subfolders, and uses the **same** CLIP embed path as inference.

```bash
# Preview coverage — no DB, no models, no writes:
python seed_library.py --data path/to/dataset.zip --dry-run

# Seed the 16 supported classes (point --data at the training dataset):
python seed_library.py --data path/to/training_dataset.zip --which supported

# Extend coverage with open-set / unknown classes:
python seed_library.py --data path/to/agro-mind.zip --which unknown
```

It is **idempotent** (skips images already seeded by SHA) and writes only
`embed_clip`. The `--dry-run` coverage report shows which of the 16 supported plants
are present in the dataset, so you can confirm coverage before a real run.

## Layout
```
backend/
├── main.py             FastAPI app + all routes
├── inference_v4.py     hierarchical ResNet50 pipeline (Stage 1 → Stage 2)
├── plant_gate.py       CLIP gate: non-plant rejection, health, framing, open-set
├── advice.py           LLM care guidance + "Sage" chat
├── db.py               Supabase REST helpers + .env loader
├── config_v4.py        class maps, plant→disease taxonomy, paths
├── finetune_stage1.py  Stage-1 fine-tuning script
├── seed_library.py     seed the similarity library (CLIP) — has --dry-run + --which
├── models/             7 .pth weights + calibration/info JSONs (Git LFS)
├── static/             web UI (landing, app, dashboard, requests)
├── supabase/migrations 7 SQL migrations (pgvector schema, taxonomy, review fns)
├── Dockerfile          light/proxy image (Hugging Face Spaces)
├── requirements.txt    full mode (torch + CLIP)
└── requirements-app.txt  proxy mode (no torch)
```
