# AgriCure — Supabase (similarity + review)

Schema for the human-in-the-loop + similarity system: a class registry/taxonomy, a
verified-image store with CLIP embeddings (pgvector), the expert review queue, and
cached care recommendations.

## Apply (one replay)
Run the migrations in order against a fresh project (psql, the Supabase SQL editor,
or the MCP). Everything is additive, so applying them is just running the files:

```
001_similarity_schema.sql       # extensions, tables, indexes, match RPC, RLS, bucket
002_seed_trained_taxonomy.sql   # the already-trained classes (taxonomy registry)
003_harden_match_function.sql   # pin match_labeled_images search_path
004_review_functions.sql        # approve/label review -> labeled_images
005_dashboard_stats.sql         # aggregate dashboard metrics RPC
006_recommendations.sql         # cached LLM care-advice table
007_review_in_progress.sql      # 'reviewing' (claimed) state
```

Then set `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` in `backend/.env` and seed the
similarity library with [`../seed_library.py`](../seed_library.py).

## Notes
- Tables have **RLS enabled with no policies**: access is server-side only via the
  `service_role` key. If a browser client ever needs direct access, add policies.
- `embed_clip` is `vector(512)` (CLIP ViT-B-32), HNSW-indexed — the **only** retrieval
  space queried by `match_labeled_images`.
- `embed_resnet` (`vector(2048)`) is **deprecated / unused**: it was kept for an
  optional exact re-rank that was never wired up, so the app and `seed_library.py` no
  longer write it. The nullable column can stay (harmless) or be dropped. To actually
  use ResNet features later, add a second match function (pgvector's ANN index caps at
  2000 dims, so index it via `halfvec`).
- Similarity query: `select * from match_labeled_images(<embedding>, 5, '<plant or null>');`
