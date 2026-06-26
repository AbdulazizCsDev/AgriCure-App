-- AgroCure similarity-check schema
-- Applies pgvector + the class registry, verified-image store, and review queue.
create extension if not exists vector;

-- ── Class registry / controlled vocabulary (two-axis: plant + disease) ──
create table if not exists taxonomy (
  id            uuid primary key default gen_random_uuid(),
  plant         text not null,
  disease       text not null,
  status        text not null default 'provisional'
                  check (status in ('trained','provisional','trusted')),
  source        text not null default 'discovered'
                  check (source in ('trained','seed','discovered')),
  support_count int  not null default 0,
  notes         text,
  created_at    timestamptz not null default now(),
  unique (plant, disease)
);

-- ── Verified dataset = the similarity memory ──
create table if not exists labeled_images (
  id                uuid primary key default gen_random_uuid(),
  image_url         text,
  image_path        text,
  plant             text not null,
  disease           text not null,
  expert_label      text,            -- "Plant | Disease" (back-compat)
  model_label       text,
  was_model_correct boolean,
  confidence        real,
  source            text not null default 'expert'
                      check (source in ('seed','expert','app')),
  embed_clip        vector(512),     -- CLIP ViT-B-32 image embedding
  embed_resnet      vector(2048),    -- ResNet50 penultimate features
  image_sha256      text,            -- dedup
  annotator         text,
  quality_flags     jsonb,
  taxonomy_id       uuid references taxonomy(id) on delete set null,
  labeled_at        timestamptz not null default now()
);

-- ── Review queue (open-set / unknown-plant triggered) ──
create table if not exists pending_reviews (
  id                uuid primary key default gen_random_uuid(),
  image_url         text,
  image_path        text,
  model_plant       text,
  model_disease     text,
  model_label       text,
  confidence        real,
  is_plant          boolean,
  healthy           boolean,
  known_plant_score real,
  stage1_conf       real,
  stage1_margin     real,
  clip_plant        text,
  trigger_reason    text,            -- 'unsupported' | 'low_confidence' | 'user_report'
  source            text default 'app',
  embed_clip        vector(512),
  embed_resnet      vector(2048),
  image_sha256      text,
  status            text not null default 'pending'
                      check (status in ('pending','skipped','reviewed','rejected')),
  created_at        timestamptz not null default now()
);

-- ── Indexes ──
-- CLIP (512-d) is the primary retrieval space → ANN index.
-- ResNet (2048-d) exceeds pgvector's 2000-d ANN limit; kept for exact compare.
create index if not exists labeled_images_clip_hnsw
  on labeled_images using hnsw (embed_clip vector_cosine_ops);
create index if not exists labeled_images_plant_idx  on labeled_images (plant);
create index if not exists labeled_images_sha_idx    on labeled_images (image_sha256);
create index if not exists pending_reviews_status_idx on pending_reviews (status, created_at desc);

-- ── Similarity search (callable from the backend via RPC) ──
create or replace function match_labeled_images(
  query_embedding vector(512),
  match_count     int  default 5,
  filter_plant    text default null
)
returns table (id uuid, plant text, disease text, image_url text, similarity real)
language sql stable
set search_path = public
as $$
  select l.id, l.plant, l.disease, l.image_url,
         (1 - (l.embed_clip <=> query_embedding))::real as similarity
  from labeled_images l
  where l.embed_clip is not null
    and (filter_plant is null or l.plant = filter_plant)
  order by l.embed_clip <=> query_embedding
  limit match_count;
$$;

-- ── RLS: lock to server-side (service_role bypasses RLS) ──
alter table taxonomy        enable row level security;
alter table labeled_images  enable row level security;
alter table pending_reviews enable row level security;

-- ── Storage bucket for the photos (public read URLs) ──
insert into storage.buckets (id, name, public)
values ('review-images', 'review-images', true)
on conflict (id) do nothing;
