-- The remote global memory store: one table in a Postgres database that every
-- MIRA instance sharing a person's global memories points at through Vault
-- mira/services/global_memory_url; that field being present is what turns remote mode on.
--
-- Run it once against that database as its owner:
--   psql "postgresql://OWNER@HOST:5432/DBNAME" -f deploy/global_memory_remote_schema.sql
--
-- No pgvector here: each instance embeds the text with its own model at boot.
-- Each instance's boot mirrors this table whole, so a row deleted here is
-- deleted from every instance at its next boot, and a row archived here stops
-- surfacing there. lt_memory/global_store.py reads and writes exactly these
-- columns.

CREATE TABLE IF NOT EXISTS global_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL CHECK (btrim(text) <> ''),
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 1.0
        CHECK (importance_score BETWEEN 0 AND 1),
    happens_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    archived_at TIMESTAMPTZ
);

-- The role named in global_memory_url needs exactly this; replace mira_global
-- with that role's name.
-- GRANT SELECT, INSERT ON global_memories TO mira_global;
