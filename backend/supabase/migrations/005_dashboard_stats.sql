-- Aggregate metrics for the expert dashboard cards.
create or replace function dashboard_stats()
returns json
language sql stable
set search_path = public
as $$
  select json_build_object(
    'pending',     (select count(*) from pending_reviews where status = 'pending'),
    'skipped',     (select count(*) from pending_reviews where status = 'skipped'),
    'verified',    (select count(*) from labeled_images),
    'reviewed',    (select count(*) from labeled_images where source = 'expert'),
    'correct',     (select count(*) from labeled_images where was_model_correct is true),
    'incorrect',   (select count(*) from labeled_images where was_model_correct is false),
    'classes',     (select count(*) from taxonomy),
    'trusted',     (select count(*) from taxonomy where status = 'trusted'),
    'provisional', (select count(*) from taxonomy where status = 'provisional'),
    'top_disease', (select disease from labeled_images
                    where disease is not null and disease <> ''
                    group by disease order by count(*) desc limit 1)
  );
$$;
