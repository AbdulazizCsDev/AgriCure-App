<div align="center">

# 🌿 AgriCure

**AI-powered plant disease diagnosis — built to know when *not* to guess.**

A hierarchical deep-learning pipeline that identifies a plant from a single leaf
photo, diagnoses its disease, and—crucially—**abstains and routes to a human
expert when it isn't sure** instead of showing a confident wrong answer.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

> **Note** — AgriCure is the production successor to the earlier *Agricare* concept
> proposal. Agricare was the pitch; **AgriCure is the working product** — a trained
> model pipeline, a REST API, an expert-review workflow, and a web UI.

---

## ✨ Highlights

- **Two-stage diagnosis** — Stage 1 identifies the *plant* (ResNet50), Stage 2
  runs a plant-specific *disease* classifier. Specialized heads beat one giant
  flat classifier.
- **Knows its limits** — a CLIP "plant gate" rejects non-leaf images (cats, hands,
  screenshots), flags healthy leaves, and warns on bad framing. Below a confidence
  floor, AgriCure **abstains** and sends the scan to the expert review queue rather
  than guessing.
- **Human-in-the-loop** — uncertain or unknown scans land in a reviewer dashboard;
  experts label them, and those labels grow a verified-image similarity library.
- **Similarity search** — CLIP + ResNet embeddings stored in Supabase (`pgvector`)
  let the system recognize plants it was never explicitly trained on.
- **Actionable guidance** — for each diagnosis an LLM generates concise
  *what-it-is / treat / prevent* advice, cached in Supabase as a growing knowledge base.
- **Two deploy modes** — full (models loaded locally) or light "proxy" mode where a
  thin web app calls a separate model service (e.g. a Hugging Face Space).

## 🧠 How it works

```
        leaf photo
            │
            ▼
   ┌──────────────────┐   not a leaf / wrong frame
   │   Plant gate      │ ───────────────────────────► reject / framing tip
   │   (CLIP zero-shot)│
   └────────┬─────────┘
            │ is a plant
            ▼
   ┌──────────────────┐
   │ Stage 1: Plant ID │  ResNet50
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐   low confidence
   │ Stage 2: Disease  │ ───────────────────────────► abstain → expert review queue
   │ (plant-specific)  │
   └────────┬─────────┘
            │ confident
            ▼
   diagnosis + LLM care advice + closest library match
```

See [`docs/architecture.png`](docs/architecture.png) for the full system diagram and
[`notebooks/agrocure_v4_training.ipynb`](notebooks/agrocure_v4_training.ipynb) for how the
models were trained.

## 🗂️ Repository structure

```
AgriCure-App/
├── backend/                 FastAPI service + ML pipeline + web UI
│   ├── main.py              API + routes (predict, review, advice, chat)
│   ├── inference_v4.py      hierarchical ResNet50 pipeline
│   ├── plant_gate.py        CLIP non-plant rejection + healthy/framing checks
│   ├── advice.py            LLM-generated plant-care guidance
│   ├── db.py                Supabase REST helpers + .env loader
│   ├── config_v4.py         class maps + paths
│   ├── models/              trained weights (.pth, via Git LFS)
│   ├── static/              web UI (landing, upload, dashboard, reviewer)
│   ├── supabase/migrations/ pgvector schema, taxonomy seed, review fns
│   ├── Dockerfile           light/proxy image for Hugging Face Spaces
│   └── .env.example         configuration template
├── notebooks/               model training notebook
├── docs/                    architecture diagram & assets
├── LICENSE
└── README.md
```

## 🚀 Quick start

> **Prerequisite:** this repo uses **Git LFS** for the model weights. Install it once
> (`git lfs install`) before cloning, then the `.pth` files download automatically.

```bash
git clone https://github.com/<your-username>/AgriCure-App.git
cd AgriCure-App/backend

python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

cp .env.example .env            # then fill in your keys (see below)
uvicorn main:app --port 8000
```

Open **http://localhost:8000**, drag in a leaf photo, and get a diagnosis.

> First request after startup loads 7 models (~630 MB) into memory — give it a few
> seconds. Runs on GPU automatically if available, otherwise CPU. The CLIP weights
> (~350 MB) download once on first start and are cached.

### Configuration

Copy [`backend/.env.example`](backend/.env.example) to `backend/.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | for review/similarity | Supabase project (server-side only) |
| `OPENAI_API_KEY` | optional | LLM care advice (omitted if unset) |
| `REVIEWER_PIN` | yes | reviewer dashboard access code |
| `MODEL_API_URL` | optional | enables light proxy deployment mode |

## 🔌 API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health`   | model status |
| `GET`  | `/classes`  | supported plants → diseases |
| `POST` | `/predict`  | `file=<image>` → JSON diagnosis |
| `GET`  | `/api/stats` | dashboard statistics |
| `GET`  | `/api/pending` | scans awaiting expert review |
| `POST` | `/api/label` | submit an expert label |
| `GET`  | `/api/advice` | cached plant-care guidance |
| `POST` | `/api/chat` | assistant chat |

```bash
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```
```json
{
  "plant": "Tomato", "disease": "Blight", "label": "Tomato | Blight",
  "confidence": 0.93, "stage1_conf": 0.99, "stage2_conf": 0.94, "latency_ms": 41.2
}
```

The web UI also includes a **reviewer dashboard** (`/dashboard`) and a **review
queue** (`/requests`) for the human-in-the-loop workflow.

## 🌱 Supported plants

Apple, Cassava, Chinese Rose, Corn (Maize), Pear, and Tomato — each with a
dedicated Stage-2 disease classifier. The similarity library extends coverage to
unseen plants via embeddings. See [`backend/README.md`](backend/README.md) for
deployment and Supabase details.

## 🛠️ Tech stack

**ML:** PyTorch · ResNet50 (hierarchical) · OpenCLIP (ViT-B-32) ·
**API:** FastAPI · Uvicorn ·
**Data:** Supabase (Postgres + pgvector) ·
**LLM:** OpenAI (care advice & chat) ·
**Deploy:** Docker · Hugging Face Spaces

## 📄 License

[MIT](LICENSE) © 2026 Abdulaziz Alhaidan
