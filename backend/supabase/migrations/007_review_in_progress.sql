-- Add a 'reviewing' lifecycle state to pending_reviews so the user-facing app can
-- show "Under review" while an expert is actively looking at their case.
-- (pending → reviewing → reviewed)  Additive, non-destructive.
alter table pending_reviews drop constraint if exists pending_reviews_status_check;
alter table pending_reviews
  add constraint pending_reviews_status_check
  check (status in ('pending','reviewing','skipped','reviewed','rejected'));
