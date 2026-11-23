-- store: postgres
-- notice: Moves the links between memories and the entities they mention
-- into their own table. Each memory keeps pointing at the same entities,
-- with the timestamp of the memory it came from. Nothing is deleted.
-- expand-contract: contract-for V0004
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id, memory_id);

-- Backfill: uuid-shaped elements whose entity exists and belongs to the same
-- user. Malformed ids and links to missing entities are skipped, so a stray
-- entry in the old list cannot abort the step. Duplicate (memory, entity)
-- pairs collapse to one row.
INSERT INTO memory_entities (memory_id, entity_id, user_id, created_at)
SELECT DISTINCT ON (m.id, (el->>'uuid')::uuid)
       m.id, (el->>'uuid')::uuid, m.user_id, m.created_at
FROM memories m
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(m.entity_links, '[]'::jsonb)) AS el
WHERE (el->>'uuid') ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
  AND EXISTS (
      SELECT 1 FROM entities e
      WHERE e.id = (el->>'uuid')::uuid AND e.user_id = m.user_id
  )
ON CONFLICT DO NOTHING;

ALTER TABLE memories DROP COLUMN IF EXISTS entity_links;

ALTER TABLE memory_entities ENABLE ROW LEVEL SECURITY;
CREATE POLICY memory_entities_user_policy ON memory_entities
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

GRANT SELECT, INSERT, UPDATE, DELETE ON memory_entities TO mira_dbuser;
