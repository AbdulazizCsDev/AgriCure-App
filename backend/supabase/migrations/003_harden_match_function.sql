-- Pin the function's search_path (security advisor 0011).
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
