-- Approve a pending review: move it into labeled_images, register/grow its class.
create or replace function approve_review(
  p_id uuid, p_plant text, p_disease text, p_annotator text default 'expert'
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare r pending_reviews%rowtype;
begin
  select * into r from pending_reviews where id = p_id;
  if not found then raise exception 'review % not found', p_id; end if;

  insert into taxonomy (plant, disease, status, source)
  values (p_plant, p_disease, 'provisional', 'discovered')
  on conflict (plant, disease) do nothing;

  insert into labeled_images (image_url, image_path, plant, disease, expert_label,
      model_label, was_model_correct, confidence, source, embed_clip, embed_resnet,
      image_sha256, annotator, taxonomy_id)
  select r.image_url, r.image_path, p_plant, p_disease, p_plant || ' | ' || p_disease,
      r.model_label, (r.model_label = p_plant || ' | ' || p_disease), r.confidence,
      'expert', r.embed_clip, r.embed_resnet, r.image_sha256, p_annotator, t.id
  from taxonomy t where t.plant = p_plant and t.disease = p_disease;

  update pending_reviews set status = 'reviewed' where id = p_id;

  update taxonomy t
  set support_count = (select count(*) from labeled_images l
                       where l.plant = t.plant and l.disease = t.disease)
  where t.plant = p_plant and t.disease = p_disease;

  update taxonomy set status = 'trusted'
  where plant = p_plant and disease = p_disease and status = 'provisional' and support_count >= 10;
end;
$$;

-- Nearest verified images to a queued review (for context in the dashboard).
create or replace function match_review(p_id uuid, match_count int default 5)
returns table (plant text, disease text, image_url text, similarity real)
language sql stable
set search_path = public
as $$
  select l.plant, l.disease, l.image_url,
         (1 - (l.embed_clip <=> r.embed_clip))::real as similarity
  from pending_reviews r, labeled_images l
  where r.id = p_id and l.embed_clip is not null and r.embed_clip is not null
  order by l.embed_clip <=> r.embed_clip
  limit match_count;
$$;
