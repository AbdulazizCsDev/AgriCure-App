---
title: AgroCure App
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AgroCure v4 — FastAPI + Web UI

Serves the hierarchical **ResNet50** plant-disease pipeline (Stage 1 plant → Stage 2 disease)
behind a REST API with a polished upload page.

## Run

```bash
cd agrocure_api
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open **http://localhost:8000** — drag in a leaf photo and get the diagnosis.

> First request after startup loads 7 models (~630 MB) into memory. Give it a few seconds.
> Runs on GPU automatically if available, otherwise CPU.

## Endpoints

| Method | Path        | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/`         | Web UI                                        |
| GET    | `/health`   | Model status                                  |
| GET    | `/classes`  | Supported plants → diseases                    |
| POST   | `/predict`  | multipart `file=<image>` → JSON diagnosis      |

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

## Non-plant rejection (plant gate)

Before diagnosing, every image passes through `plant_gate.py`:

1. **CLIP zero-shot gate** — CLIP (open_clip, ViT-B-32) scores the image against
   "plant leaf" prompts vs "animal / person / hand / car / object / screenshot"
   prompts. It only passes if the plant group wins (`PLANT_THRESHOLD`). This is a
   direct *"is this a leaf?"* signal, so it accepts hard/ambiguous real leaves and
   rejects cats, hands, cars, faces — things a confidence threshold cannot separate.
2. **Confidence floor** — a last-resort net; rejects only if the plant classifier
   is below `STAGE1_MIN_CONF` (kept very low so real leaves are never rejected here).

### Healthy check
If the image is a plant, CLIP then scores it against "healthy leaf" vs
"sick leaf / leaf with spots" prompts. If the healthy group wins
`HEALTH_THRESHOLD` (0.70, set high on purpose), the API returns
`healthy: true` and skips disease diagnosis:
```json
{ "is_plant": true, "healthy": true, "plant": "Tomato",
  "label": "Tomato — Healthy", "health_score": 0.80 }
```
### Framing tip
CLIP also flags **wide / whole-plant shots** (vs a close-up single leaf) and adds a
non-blocking `framing_tip` to the response. The UI shows it as an amber banner and a
"best results / avoid" guide under the upload box — steering users to the close-up
single-leaf photos the model was trained on. Tune with `FRAMING_THRESHOLD` (0.50).

> ⚠️ CLIP is **weak** at healthy-vs-diseased (a fine, spot-level distinction) —
> the two score distributions nearly overlap. The threshold is biased so a *sick*
> leaf is rarely called healthy, at the cost of sometimes calling a healthy leaf
> "diseased". For reliable health detection, train a dedicated healthy/diseased
> classifier (PlantVillage has ready `*___healthy` folders).

Rejected responses look like:
```json
{ "is_plant": false,
  "message": "No plant leaf detected — please upload a clear single-leaf photo.",
  "reason": "looks like: cat (88%)" }
```

Tune strictness at the top of `plant_gate.py`:
- `PLANT_THRESHOLD` (default 0.55) — higher = stricter (more likely to reject).
- `STAGE1_MIN_CONF` (default 0.10) — last-resort confidence floor.
- `PLANT_PROMPTS` / `NONPLANT_PROMPTS` — add prompts to tune behavior.

> The CLIP weights (~350 MB) download once on first startup and are cached.

## Layout
```
agrocure_api/
├── main.py            FastAPI app
├── inference_v4.py    pipeline (ResNet50)
├── config_v4.py       class maps + paths
├── plant_gate.py      non-plant rejection (ImageNet gate + threshold)
├── models/            7 .pth + calibration/info JSONs
└── static/index.html  web UI
```
