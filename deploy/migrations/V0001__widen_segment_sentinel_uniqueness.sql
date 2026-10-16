-- store: postgres
-- notice: Fixes a rule that could lock a conversation after a message was
-- saved while it was paused. Nothing is deleted. If a conversation is
-- already in that state, MIRA stops and names it before changing anything.
DO $precondition$
DECLARE
    bad RECORD;
BEGIN
    SELECT continuum_id, count(*) AS n INTO bad
      FROM messages
     WHERE metadata->>'is_segment_boundary' = 'true'
       AND metadata->>'status' IN ('active', 'paused')
     GROUP BY continuum_id
    HAVING count(*) > 1
     LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'This upgrade stopped before changing anything: one conversation has % live segment boundaries (conversation %), and only one is allowed. Ask MIRA to remove the duplicate — it needs that conversation id — then run: mira update', bad.n, bad.continuum_id;
    END IF;
END
$precondition$;
DROP INDEX IF EXISTS idx_messages_active_segment_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_live_segment_unique
    ON messages(continuum_id)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' IN ('active', 'paused');
