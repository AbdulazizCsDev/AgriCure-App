# AgroCure — Supabase (similarity check)

Schema for the unknown-plant similarity system: a class registry, a verified-image
store with CLIP + ResNet embeddings (pgvector), and the expert review queue.

## Projects
- **Test:** `agrocure-similarity` (`iiyzgdipuivypzhwhbzt`, eu-central-1) — build/iterate here.
- **Origin:** the shared AgroCure DB (different owner) — promote to it later.

## Apply / promote (one replay)
Run the migrations in order against the target project:

```
001_similarity_schema.sql       # extensions, tables, indexes, RPC, RLS, bucket
002_seed_trained_taxonomy.sql   # the 30 already-trained classes
003_harden_match_function.sql   # pin function search_path
```

Because everything is additive migration SQL, promoting to the origin DB is just
running these files there (psql, the Supabase SQL editor, or the MCP).

## Notes
- Tables have **RLS enabled with no policies**: access is server-side only via the
  `service_role` key. If a browser client ever needs direct access, add policies.
- `embed_clip` is `vector(512)` (CLIP ViT-B-32) and HNSW-indexed.
  `embed_resnet` is `vector(2048)` (ResNet50) — kept for exact compare; pgvector's
  ANN index caps at 2000 dims, so use `halfvec` later if you want it indexed.
- Similarity query: `select * from match_labeled_images(<embedding>, 5, '<plant or null>');`
