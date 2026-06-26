"""
Supabase helper for the similarity-check feature.

Reads credentials from `.env` (SUPABASE_URL, SUPABASE_SERVICE_KEY). If they're
missing the module stays DISABLED and every call is a safe no-op, so the API
still runs without a database.
"""

import os
import requests

BASE = os.path.dirname(os.path.abspath(__file__))


def _load_env(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


_env = _load_env(os.path.join(BASE, ".env"))
URL = (_env.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")).rstrip("/")
KEY = _env.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY", "")

ENABLED = bool(URL and KEY and not KEY.startswith("PASTE_"))
REST, STORAGE, BUCKET = f"{URL}/rest/v1", f"{URL}/storage/v1", "review-images"
_H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


def _vec(lst):
    return "[" + ",".join(f"{x:.6f}" for x in lst) + "]"


def match(embedding, k=5, plant=None):
    """Return the k nearest verified images: [{plant, disease, image_url, similarity}]."""
    if not ENABLED:
        return []
    r = requests.post(
        f"{REST}/rpc/match_labeled_images",
        headers={**_H, "Content-Type": "application/json"},
        json={"query_embedding": _vec(embedding), "match_count": k, "filter_plant": plant},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def upload_image(file_bytes, filename, content_type="image/jpeg"):
    if not ENABLED:
        return None
    r = requests.post(
        f"{STORAGE}/object/{BUCKET}/{filename}",
        headers={**_H, "Content-Type": content_type, "x-upsert": "true"},
        data=file_bytes, timeout=30,
    )
    r.raise_for_status()
    return f"{STORAGE}/object/public/{BUCKET}/{filename}"


def insert(table, row):
    if not ENABLED:
        return None
    r = requests.post(
        f"{REST}/{table}",
        headers={**_H, "Content-Type": "application/json", "Prefer": "return=minimal"},
        json=row, timeout=15,
    )
    r.raise_for_status()
    return True


def insert_returning(table, row):
    """Insert a row and return it (so callers can read the generated id)."""
    if not ENABLED:
        return None
    r = requests.post(
        f"{REST}/{table}",
        headers={**_H, "Content-Type": "application/json", "Prefer": "return=representation"},
        json=row, timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def rpc(name, payload):
    if not ENABLED:
        return None
    r = requests.post(f"{REST}/rpc/{name}",
                      headers={**_H, "Content-Type": "application/json"},
                      json=payload, timeout=15)
    r.raise_for_status()
    return r.json() if r.text else None


def get_pending(status="pending"):
    """Queued scans for the dashboard (embeddings excluded to keep it light).

    Includes cases an expert has opened ('reviewing') so a claimed card stays on
    screen — without that, marking a case 'reviewing' would make it vanish.
    """
    if not ENABLED:
        return []
    cols = ("id,image_url,model_plant,model_disease,model_label,confidence,"
            "known_plant_score,stage1_conf,clip_plant,status,created_at")
    r = requests.get(f"{REST}/pending_reviews", headers=_H,
                     params={"status": "in.(pending,reviewing)", "select": cols,
                             "order": "created_at.desc"}, timeout=15)
    r.raise_for_status()
    return r.json()


def pending_id_by_sha(sha):
    """Return the id of an existing pending/reviewing case for this image, or None."""
    if not ENABLED or not sha:
        return None
    r = requests.get(f"{REST}/pending_reviews", headers=_H,
                     params={"image_sha256": f"eq.{sha}",
                             "status": "in.(pending,reviewing)",
                             "select": "id", "limit": "1"}, timeout=15)
    r.raise_for_status()
    rows = r.json()
    return rows[0]["id"] if rows else None


def claim_review(review_id):
    """Move a case pending → reviewing (best-effort; never overrides a final state)."""
    if not ENABLED or not review_id:
        return False
    r = requests.patch(f"{REST}/pending_reviews",
                       headers={**_H, "Content-Type": "application/json", "Prefer": "return=minimal"},
                       params={"id": f"eq.{review_id}", "status": "eq.pending"},
                       json={"status": "reviewing"}, timeout=15)
    r.raise_for_status()
    return True


def get_review_status(review_id):
    """Lifecycle of a user's case: pending | reviewing | reviewed (+ expert verdict)."""
    if not ENABLED or not review_id:
        return None
    r = requests.get(f"{REST}/pending_reviews", headers=_H,
                     params={"id": f"eq.{review_id}",
                             "select": "id,status,image_sha256,model_plant,model_disease",
                             "limit": "1"}, timeout=15)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    row = rows[0]
    out = {
        "status": row.get("status") or "pending",
        "model_plant": row.get("model_plant"),
        "model_disease": row.get("model_disease"),
    }
    if out["status"] == "reviewed" and row.get("image_sha256"):
        lr = requests.get(f"{REST}/labeled_images", headers=_H,
                          params={"image_sha256": f"eq.{row['image_sha256']}",
                                  "select": "plant,disease,annotator,labeled_at",
                                  "order": "labeled_at.desc", "limit": "1"}, timeout=15)
        lr.raise_for_status()
        ld = lr.json()
        if ld:
            out["expert_plant"] = ld[0].get("plant")
            out["expert_disease"] = ld[0].get("disease")
            out["annotator"] = ld[0].get("annotator")
            out["reviewed_at"] = ld[0].get("labeled_at")
    return out


def get_recommendation(plant, disease):
    """Cached care guidance for a (plant, disease), or None."""
    if not ENABLED:
        return None
    r = requests.get(f"{REST}/recommendations", headers=_H,
                     params={"plant": f"eq.{plant}", "disease": f"eq.{disease}", "limit": "1"},
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def get_stats():
    """Aggregate dashboard metrics (single JSON via RPC)."""
    if not ENABLED:
        return {}
    return rpc("dashboard_stats", {}) or {}


def get_taxonomy():
    """Known plants/diseases for the annotation dropdowns."""
    if not ENABLED:
        return []
    r = requests.get(f"{REST}/taxonomy", headers=_H,
                     params={"select": "plant,disease,status,support_count",
                             "order": "plant.asc"}, timeout=15)
    r.raise_for_status()
    return r.json()


def update_status(row_id, status):
    if not ENABLED:
        return False
    r = requests.patch(f"{REST}/pending_reviews",
                       headers={**_H, "Content-Type": "application/json", "Prefer": "return=minimal"},
                       params={"id": f"eq.{row_id}"}, json={"status": status}, timeout=15)
    r.raise_for_status()
    return True


def exists_sha(table, sha):
    """True if a row with this image hash already exists (avoid duplicate queue entries)."""
    if not ENABLED or not sha:
        return False
    r = requests.get(
        f"{REST}/{table}",
        headers=_H, params={"image_sha256": f"eq.{sha}", "select": "id", "limit": "1"},
        timeout=15,
    )
    r.raise_for_status()
    return len(r.json()) > 0
