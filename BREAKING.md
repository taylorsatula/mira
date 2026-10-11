# Breaking: segment sentinel uniqueness widened to paused

`idx_messages_active_segment_unique` is replaced by
`idx_messages_live_segment_unique`, which covers `status IN ('active','paused')`
instead of `active` alone. Reason (2026-10-10 incident): a save landing while
the segment sat paused minted a second live sentinel — the old index only
guarded the active half of the one-live-sentinel invariant — and every later
message entry then violated the index, permanently locking chat out.

`mira update` refuses this release. Manual path on the existing database:

```sql
DROP INDEX IF EXISTS idx_messages_active_segment_unique;
CREATE UNIQUE INDEX idx_messages_live_segment_unique ON messages(continuum_id)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' IN ('active', 'paused');
```

The CREATE fails if a continuum already carries two live sentinels — that is
the same corruption this change prevents; resolve the duplicate rows first
(one live sentinel per continuum must remain; reassign or remove the stray
segment's messages, then delete the stray sentinel row).

Fresh installs get the new index from `deploy/mira_service_schema.sql`
directly; no other schema objects reference the old name.
