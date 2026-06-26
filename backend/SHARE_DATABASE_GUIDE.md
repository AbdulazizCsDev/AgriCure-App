# AgroCure — Shared Database Guide

This explains how to connect an app to the shared **AgroCure similarity** Supabase
database so it can **read and write** the same data. Give this file to your
collaborator.

---

## 1. What this database is

A plant-disease "similarity memory". Each verified leaf image is stored with a
vector embedding (via **pgvector**), so any app can ask *"what known images look
like this one?"*. There's also a review queue for scans an expert hasn't labeled yet.

Tables:
- **`labeled_images`** — verified images + embeddings (the searchable library)
- **`pending_reviews`** — scans waiting for an expert to label (the work queue)
- **`taxonomy`** — the class registry (every known plant + disease, with status)
- **`review-images`** — a Storage bucket holding the actual photo files

---

## 2. Credentials

| Name | Value | Secret? |
|------|-------|---------|
| `SUPABASE_URL` | `https://iiyzgdipuivypzhwhbzt.supabase.co` | No (public) |
| `SUPABASE_SERVICE_KEY` | *(the `service_role` key — sent to you privately)* | **YES** |

> ⚠️ **The `service_role` key is full admin access.** It bypasses all security
> rules. Use it **only on a server / backend**. Never put it in a browser, a
> mobile app bundle, or a public git repo. Store it in an environment variable.
>
> If your app is **client-side only** (browser/mobile with no backend), do **not**
> use this key. Ask the owner to add row-level-security policies + a publishable
> key instead.

Set them before running:
```bash
export SUPABASE_URL="https://iiyzgdipuivypzhwhbzt.supabase.co"
export SUPABASE_SERVICE_KEY="<service_role key sent to you>"
```

---

## 3. Schema

### `labeled_images` — verified library (searchable)
| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid | auto |
| `image_url` | text | public URL of the photo |
| `image_path` | text | filename inside the `review-images` bucket |
| `plant` | text | **required** |
| `disease` | text | **required** |
| `expert_label` | text | `"Plant \| Disease"` (back-compat) |
| `model_label` | text | what a model originally predicted |
| `was_model_correct` | bool | |
| `confidence` | real | 0–1 |
| `source` | text | `seed` \| `expert` \| `app` |
| `embed_clip` | vector(512) | CLIP ViT-B-32 image embedding |
| `embed_resnet` | vector(2048) | ResNet50 features (optional) |
| `image_sha256` | text | dedup hash |
| `annotator` | text | who labeled it |
| `taxonomy_id` | uuid | FK → `taxonomy` |
| `labeled_at` | timestamptz | auto |

### `pending_reviews` — work queue
| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid | auto |
| `image_url`, `image_path` | text | photo location |
| `model_plant`, `model_disease`, `model_label` | text | the model's guess |
| `confidence` | real | 0–1 |
| `is_plant`, `healthy` | bool | |
| `known_plant_score`, `stage1_conf`, `stage1_margin` | real | open-set signals |
| `clip_plant` | text | CLIP's plant guess |
| `trigger_reason` | text | why it was queued |
| `source` | text | `app` \| `user_report` |
| `embed_clip` | vector(512) | so similarity works on approval |
| `embed_resnet` | vector(2048) | optional |
| `image_sha256` | text | dedup |
| `status` | text | `pending` \| `skipped` \| `reviewed` \| `rejected` |
| `created_at` | timestamptz | auto |

### `taxonomy` — class registry
| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid | auto |
| `plant`, `disease` | text | unique together |
| `status` | text | `trained` \| `provisional` \| `trusted` |
| `source` | text | `trained` \| `seed` \| `discovered` |
| `support_count` | int | number of verified images |
| `created_at` | timestamptz | auto |

### `review-images` Storage bucket (public read)
```
https://iiyzgdipuivypzhwhbzt.supabase.co/storage/v1/object/public/review-images/<image_path>
```

---

## 4. Embeddings (important)

`embed_clip` / `embed_resnet` are pgvector columns. Over the REST API you send a
vector as a **string**: `"[0.0123,-0.045,...]"` (512 numbers for CLIP, 2048 for
ResNet). To get embeddings, encode the image with **CLIP ViT-B-32** (`open_clip`,
pretrained `openai`) and L2-normalize. Rows can be inserted **without** embeddings,
but similarity search won't find them until an embedding is present.

---

## 5. REST API (PostgREST)

```
Base:    https://iiyzgdipuivypzhwhbzt.supabase.co/rest/v1
Headers: apikey: <SERVICE_KEY>
         Authorization: Bearer <SERVICE_KEY>
         Content-Type: application/json
```

| Goal | Request |
|------|---------|
| New pending scans | `GET /pending_reviews?status=eq.pending&order=created_at.desc` |
| All known classes | `GET /taxonomy?select=plant,disease,status` |
| Insert a queued scan | `POST /pending_reviews` (JSON body) |
| Insert a verified image | `POST /labeled_images` (JSON body) |
| Update a status | `PATCH /pending_reviews?id=eq.<id>` body `{"status":"reviewed"}` |

Filter syntax: `column=eq.value`, `order=column.desc`.

### Similarity search & approval (RPC)
```
POST /rest/v1/rpc/match_labeled_images
  body: {"query_embedding":"[...512 floats...]","match_count":5,"filter_plant":null}
  -> [{id, plant, disease, image_url, similarity}]

POST /rest/v1/rpc/match_review
  body: {"p_id":"<pending id>","match_count":5}
  -> nearest verified images to a queued scan

POST /rest/v1/rpc/approve_review
  body: {"p_id":"<id>","p_plant":"Avocado Tree","p_disease":"Brown Spot","p_annotator":"alex"}
  -> moves the review into labeled_images and grows the class
```

---

## 6. Quick start — JavaScript (supabase-js, server-side)

```js
import { createClient } from "@supabase/supabase-js";
const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);

// read the queue
const { data: pending } = await sb
  .from("pending_reviews").select("*").eq("status", "pending");

// add a scan to the queue
await sb.from("pending_reviews").insert({
  image_url: url, image_path: path, model_plant: "Tomato",
  model_disease: "Blight", confidence: 0.58, status: "pending",
  embed_clip: "[" + clipVector.join(",") + "]",   // optional
});

// similarity search
const { data: matches } = await sb.rpc("match_labeled_images", {
  query_embedding: "[" + clipVector.join(",") + "]", match_count: 5, filter_plant: null,
});

// upload a photo to the bucket
await sb.storage.from("review-images").upload(`app/${name}.jpg`, fileBuffer,
  { contentType: "image/jpeg", upsert: true });
```

## 7. Quick start — Python

```python
import os, requests
URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SERVICE_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# insert a queued scan
requests.post(f"{URL}/rest/v1/pending_reviews", headers=H, json={
    "image_url": url, "image_path": path, "model_plant": "Tomato",
    "model_disease": "Blight", "confidence": 0.58, "status": "pending",
})

# similarity search
matches = requests.post(f"{URL}/rest/v1/rpc/match_labeled_images", headers=H,
    json={"query_embedding": "[" + ",".join(map(str, vec)) + "]", "match_count": 5}).json()
```

---

## 8. Coordination & safety

- **Multiple apps share these tables.** Use the `status` field so two people don't
  label the same scan: a labeled scan becomes `reviewed` and leaves the `pending`
  queue. Use `skipped` for "review later".
- **Dedup** with `image_sha256` (SHA-256 of the image bytes) to avoid duplicates.
- **Label format** is `"Plant | Disease"` (spaces around the pipe) to match existing data.
- **Don't delete** `taxonomy` rows with `status='trained'` — those mirror the model's classes.
- Keep the `service_role` key server-side. Rotate it (Supabase → Settings → API)
  if it is ever exposed.
