"""
AgroCure v4 — FastAPI inference server
Serves the hierarchical ResNet50 pipeline (Stage 1 plant → Stage 2 disease).

Run:
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000
"""

import io
import os
import sys
import time

# Windows consoles default to cp1252, which crashes on the ✓ chars the
# inference module prints. Force UTF-8 so startup logging is portable.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import hashlib

import requests

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config_v4 as config
import db
import advice

# ── Deployment mode ───────────────────────────────────────────────────────────
# If MODEL_API_URL is set we run "light": the heavy CNN/CLIP models live in a
# separate service (e.g. a Hugging Face Space) and we just call it for /predict.
# If it's NOT set we load the models locally (full features, used in dev).
MODEL_API_URL = (os.environ.get("MODEL_API_URL") or db._env.get("MODEL_API_URL") or "").strip().rstrip("/")
PROXY_MODE = bool(MODEL_API_URL)

if not PROXY_MODE:
    import torch
    import inference_v4
    import plant_gate
    from PIL import Image, UnidentifiedImageError

SIMILARITY_THRESHOLD = 0.85    # cosine; show the closest library match above this
LIBRARY_PLANT_OVERRIDE = 0.88  # trust the library's plant over the CNN above this
CROSSCHECK_THRESHOLD = 0.90    # a match this strong that DISAGREES flags the scan uncertain
CONF_FLOOR = 0.70              # below this (diseased) we abstain and hand to an expert

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="AgroCure v4 API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup (models stay warm in memory)
pipeline = None
gate = None


@app.on_event("startup")
def load_pipeline():
    global pipeline, gate
    if PROXY_MODE:
        print(f"Proxy mode — diagnosis served by {MODEL_API_URL}")
        return
    pipeline = inference_v4.AgroCureV4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = plant_gate.PlantGate(device)
    print("Plant gate loaded ✓")


@app.get("/health")
def health():
    if PROXY_MODE:
        return {"status": "ok", "mode": "proxy", "model_api": MODEL_API_URL}
    return {
        "status": "ok" if pipeline is not None and gate is not None else "loading",
        "plants": len(pipeline.plant_names) if pipeline else 0,
        "stage2_models": len(pipeline.stage2_models) if pipeline else 0,
        "plant_gate": gate is not None,
    }


@app.get("/classes")
def classes():
    """Return the full plant → diseases map so the UI can show what's supported."""
    return {
        "plants": config.PLANT_DISEASES,
        "num_plants": len(config.PLANT_DISEASES),
        "num_classes": sum(len(d) for d in config.PLANT_DISEASES.values()),
    }


def open_set_flags(result: dict, known: dict) -> dict:
    """Combine Stage-1 (ResNet) and CLIP signals into an unknown-plant warning.

    We never reject here — the closest match is still returned as a best guess.
    """
    s1     = result.get("stage1_conf", 1.0)
    margin = result.get("stage1_margin", 1.0)
    ks     = known["known_score"]

    reasons = []
    if ks < plant_gate.KNOWN_PLANT_THRESHOLD:
        reasons.append("visual features don't strongly match a supported species")
    if s1 < plant_gate.S1_WARN_CONF:
        reasons.append(f"low plant-identity confidence ({s1:.0%})")
    if margin < plant_gate.S1_WARN_MARGIN:
        reasons.append("two species scored almost equally")

    unsupported = len(reasons) > 0
    return {
        "unsupported": unsupported,
        "warning": (
            "This may not be one of the 16 supported species. "
            "Showing the closest match as a best guess."
        ) if unsupported else None,
        "warning_reason": "; ".join(reasons) if reasons else None,
        "known_plant_score": known["known_score"],
        "clip_plant": known["clip_plant"],
    }


def similarity_lookup(feat):
    """Query the verified memory with the CLIP embedding. Return a 'similar' dict or None."""
    if not db.ENABLED:
        return None
    try:
        emb = feat[0].detach().cpu().tolist()
        matches = db.match(emb, k=5)
    except Exception:
        return None
    if not matches or (matches[0].get("similarity") or 0) < SIMILARITY_THRESHOLD:
        return None
    top = matches[0]
    agree = sum(1 for m in matches
                if m.get("plant") == top["plant"] and m.get("disease") == top["disease"])
    return {
        "similar_plant": top["plant"],
        "similar_disease": top["disease"],
        "similarity": round(float(top["similarity"]), 4),
        "neighbors_agree": agree,
        "neighbors_total": len(matches),
        "similar_image_url": top.get("image_url"),
    }


def enqueue_review(feat, contents, content_type, result, flags, healthy):
    """Best-effort: save a flagged scan into pending_reviews for expert labeling.

    Returns the review id (so the user can track its status) or None.
    """
    if not db.ENABLED:
        return None
    try:
        sha = hashlib.sha256(contents).hexdigest()
        if db.exists_sha("labeled_images", sha):
            return None  # already verified — nothing pending to track
        existing = db.pending_id_by_sha(sha)
        if existing:
            return existing  # same image already queued — reuse its id
        ext = "png" if (content_type or "").endswith("png") else "jpg"
        path = f"queue/{sha}.{ext}"
        url = db.upload_image(contents, path, content_type or "image/jpeg")
        row = db.insert_returning("pending_reviews", {
            "image_url": url, "image_path": path, "image_sha256": sha,
            "model_plant": result.get("plant"), "model_disease": result.get("disease"),
            "model_label": result.get("label"), "confidence": result.get("confidence"),
            "is_plant": True, "healthy": healthy,
            "known_plant_score": flags.get("known_plant_score"),
            "stage1_conf": result.get("stage1_conf"),
            "stage1_margin": result.get("stage1_margin"),
            "clip_plant": flags.get("clip_plant"),
            "trigger_reason": "unsupported", "source": "app",
            "embed_clip": db._vec(feat[0].detach().cpu().tolist()),
            "status": "pending",
        })
        return row.get("id") if row else None
    except Exception:
        return None


def enqueue_review_noembed(contents, content_type, plant, disease, label, confidence, healthy):
    """Queue an uncertain scan for expert review without a CLIP embedding (proxy mode).

    The visual-similarity features stay quiet (no embedding) but the case is still
    captured, tracked, and shown to experts. Returns the review id or None.
    """
    if not db.ENABLED:
        return None
    try:
        sha = hashlib.sha256(contents).hexdigest()
        if db.exists_sha("labeled_images", sha):
            return None
        existing = db.pending_id_by_sha(sha)
        if existing:
            return existing
        ext = "png" if (content_type or "").endswith("png") else "jpg"
        path = f"queue/{sha}.{ext}"
        url = db.upload_image(contents, path, content_type or "image/jpeg")
        row = db.insert_returning("pending_reviews", {
            "image_url": url, "image_path": path, "image_sha256": sha,
            "model_plant": plant, "model_disease": disease, "model_label": label,
            "confidence": confidence, "is_plant": True, "healthy": healthy,
            "trigger_reason": "low_confidence", "source": "app", "status": "pending",
        })
        return row.get("id") if row else None
    except Exception:
        return None


async def predict_proxy(file: UploadFile):
    """Forward the image to the external model service and shape its reply for the UI."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    contents = await file.read()
    t0 = time.time()
    try:
        r = requests.post(
            f"{MODEL_API_URL}/predict",
            files={"file": (file.filename or "leaf.jpg", contents, file.content_type)},
            timeout=60,
        )
        r.raise_for_status()
        m = r.json()
    except Exception as e:
        raise HTTPException(502, f"Model service unavailable: {e}")

    if not m.get("is_plant", True):
        return JSONResponse({
            "is_plant": False,
            "message": m.get("message") or "No plant leaf detected — please upload a clear single-leaf photo.",
            "reason": m.get("reason"),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })

    plant = m.get("plant")
    disease = m.get("disease")
    conf = m.get("confidence")
    healthy = bool(disease and str(disease).strip().lower() == "healthy")
    level = str(m.get("confidence_level") or "").lower()
    # Trust-over-guessing: low confidence (or the model says so) → hand to an expert.
    uncertain = (not healthy) and (((conf is not None) and conf < CONF_FLOOR) or level == "low")

    out = {
        "is_plant": True,
        "healthy": healthy,
        "plant": plant,
        "disease": None if healthy else disease,
        "label": m.get("label"),
        "confidence": conf,
        "stage1_conf": m.get("stage1_conf"),
        "stage2_conf": m.get("stage2_conf"),
        "unsupported": uncertain,
        "queued": False,
        "review_id": None,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    if healthy:
        out["health_score"] = conf
        out["message"] = "Looks healthy — no disease detected."
    if uncertain:
        rid = enqueue_review_noembed(contents, file.content_type, plant, disease, m.get("label"), conf, healthy)
        out["review_id"] = rid
        out["queued"] = bool(rid)
    return JSONResponse(out)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if PROXY_MODE:
        return await predict_proxy(file)
    if pipeline is None:
        raise HTTPException(503, "Model still loading, try again in a moment.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(400, "Could not read that image.")

    t0 = time.time()

    # Encode the image once; all CLIP checks reuse this embedding.
    feat = gate.embed(pil)

    # ── Gate 1: is this even a plant? ─────────────────────────────────────────
    g = gate.check(feat)
    if not g["is_plant"]:
        return JSONResponse({
            "is_plant": False,
            "message": "No plant leaf detected — please upload a clear single-leaf photo.",
            "reason": g["reason"],
            "top_label": g["top_label"],
            "top_conf": g["top_conf"],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })

    # ── Diagnose (also gives us the plant name) ───────────────────────────────
    try:
        result = pipeline.predict(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # ── Gate 2: confidence floor (secondary net) ──────────────────────────────
    if result.get("stage1_conf", 1.0) < plant_gate.STAGE1_MIN_CONF:
        return JSONResponse({
            "is_plant": False,
            "message": "Not confident this is a supported plant — please upload a clearer leaf photo.",
            "reason": f"low plant confidence ({result['stage1_conf']:.0%})",
            "top_label": g["top_label"],
            "top_conf": g["top_conf"],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })

    # ── Framing tip: nudge toward a close-up single leaf (non-blocking) ────────
    framing = gate.check_framing(feat)
    framing_tip = (
        None if framing["closeup"]
        else "Tip: for best accuracy, photograph a single leaf up close — "
             "wide shots of whole plants can confuse the model."
    )

    # ── Open-set check: is this actually one of the 16 supported species? ──────
    known = gate.check_known_plant(feat)
    flags = open_set_flags(result, known)
    h = gate.check_health(feat)
    healthy = bool(h["healthy"])

    # ── Similarity memory: second opinion + capture ────────────────────────────
    # Run the library lookup on every scan. A *very confident* match that names a
    # different plant than the CNN is strong evidence the CNN is wrong (its closed
    # set of 16 can't represent this species) — so we flag the scan uncertain and
    # prefer the library. A weak/agreeing match changes nothing (no false alarms
    # on confidently-identified supported plants).
    extra = dict(flags)
    extra["queued"] = False
    sim = similarity_lookup(feat)

    crosscheck = bool(sim and sim["similarity"] >= CROSSCHECK_THRESHOLD
                      and sim["similar_plant"] != result["plant"])
    low_conf = (not healthy) and (result.get("confidence", 1.0) < CONF_FLOOR)
    uncertain = flags["unsupported"] or crosscheck or low_conf
    extra["unsupported"] = uncertain
    extra["wide_shot"] = framing_tip is not None
    if crosscheck and not extra.get("warning"):
        extra["warning"] = "The closest known match is a different species — showing it as a best guess."
        extra["warning_reason"] = (f"library match '{sim['similar_plant']}' "
                                    f"({sim['similarity']:.0%}) differs from the model's guess")

    # Trustworthy default: when unsure we DON'T guess — we hand it to an expert.
    # So every uncertain scan is captured for review (with or without a match).
    if uncertain:
        if sim:
            extra.update(sim)
        review_id = enqueue_review(feat, contents, file.content_type, result, flags, healthy)
        extra["review_id"] = review_id
        extra["queued"] = bool(review_id)

    # Prefer the library's plant identity when uncertain and the match is confident.
    use_lib = bool(uncertain and sim and sim["similarity"] >= LIBRARY_PLANT_OVERRIDE
                   and sim["similar_plant"] != result["plant"])
    extra["plant_from_library"] = use_lib
    display_plant = sim["similar_plant"] if use_lib else result["plant"]

    # ── Healthy leaf → stop here, skip disease diagnosis ───────────────────────
    if healthy:
        return JSONResponse({
            "is_plant": True,
            "healthy": True,
            "plant": display_plant,
            "label": f"{display_plant} | Healthy",
            "health_score": h["healthy_score"],
            "stage1_conf": result.get("stage1_conf"),
            "message": "Looks healthy — no disease detected.",
            "framing_tip": framing_tip,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            **extra,
        })

    result["is_plant"] = True
    result["healthy"] = False
    result["health_score"] = h["healthy_score"]
    result["framing_tip"] = framing_tip
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    result.update(extra)
    return JSONResponse(result)


# ── Frontend ────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def landing():
    return FileResponse(os.path.join(STATIC_DIR, "landing.html"))


@app.get("/app")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Expert review dashboard ───────────────────────────────────────────────────
@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.post("/api/reviewer-auth")
def reviewer_auth(body: dict = Body(...)):
    """Validate the reviewer access code (set REVIEWER_PIN in .env; default 314159)."""
    import hmac
    pin = (db._env.get("REVIEWER_PIN") or os.environ.get("REVIEWER_PIN") or "314159").strip()
    given = str(body.get("code", "")).strip()
    if given and hmac.compare_digest(given, pin):
        return {"ok": True}
    return JSONResponse({"ok": False, "error": "Incorrect access code."}, status_code=401)


@app.get("/requests")
def requests_page():
    return FileResponse(os.path.join(STATIC_DIR, "requests.html"))


@app.get("/api/pending")
def api_pending():
    return db.get_pending()


@app.get("/api/stats")
def api_stats():
    return db.get_stats()


@app.get("/api/advice")
def api_advice(plant: str, disease: str):
    """Care guidance for a plant+disease (cached, LLM-generated on first ask)."""
    try:
        return advice.get_or_create(plant, disease) or {}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/taxonomy")
def api_taxonomy():
    return db.get_taxonomy()


@app.get("/api/neighbors")
def api_neighbors(id: str):
    return db.rpc("match_review", {"p_id": id, "match_count": 5}) or []


@app.post("/api/label")
def api_label(body: dict = Body(...)):
    if not (body.get("id") and body.get("plant") and body.get("disease")):
        raise HTTPException(400, "id, plant and disease are required.")
    db.rpc("approve_review", {
        "p_id": body["id"], "p_plant": body["plant"].strip(),
        "p_disease": body["disease"].strip(), "p_annotator": body.get("annotator", "expert"),
    })
    return {"ok": True}


@app.post("/api/skip")
def api_skip(body: dict = Body(...)):
    db.update_status(body["id"], body.get("status", "skipped"))
    return {"ok": True}


@app.get("/api/review-status")
def api_review_status(id: str):
    """Track a user's handed-to-expert case: pending → reviewing → reviewed."""
    return db.get_review_status(id) or {"status": "unknown"}


@app.post("/api/claim")
def api_claim(body: dict = Body(...)):
    """Dashboard marks a case as actively being looked at (powers 'Under review')."""
    if body.get("id"):
        try:
            db.claim_review(body["id"])
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/chat")
def api_chat(body: dict = Body(...)):
    """Diagnosis-aware follow-up chat. Degrades cleanly when no AI key is set."""
    if not advice.ENABLED:
        return {"reply": None, "error": "AI chat is not available right now."}
    messages = body.get("messages") or []
    context = body.get("context") or {}
    if not messages:
        raise HTTPException(400, "messages are required.")
    try:
        return {"reply": advice.chat(context, messages)}
    except Exception as e:
        return {"reply": None, "error": str(e)}
