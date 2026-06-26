-- LLM-generated (later expert-editable) care guidance per disease. Cached so each
-- plant+disease is generated once, then served from here — a growing knowledge base.
create table if not exists recommendations (
  id          uuid primary key default gen_random_uuid(),
  plant       text not null,
  disease     text not null,
  what_it_is  text,
  treat       text,
  prevent     text,
  source      text not null default 'llm',   -- 'llm' | 'expert'
  model       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (plant, disease)
);
alter table recommendations enable row level security;
