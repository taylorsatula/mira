-- MIRA database ownership bootstrap
--
-- Makes mira_admin the owner of everything MIRA created in the public schema
-- and creates the step ledger, so that every later database step
-- (deploy/migrations/, run by scripts/migrate.py) runs as mira_admin over TCP
-- with no superuser. Idempotent: safe to run on a fresh database right after
-- deploy/mira_service_schema.sql and on an installed database of any earlier
-- release. Stamps nothing — the step runner decides what the ledger records.
--
-- Run only as the PostgreSQL superuser, through psql:
--   psql -v ON_ERROR_STOP=1 -d mira_service -f deploy/bootstrap_adopt.sql

-- BEGIN GENERATED LEDGER DDL (utils/schema_migrations.py:ledger_ddl)
CREATE TABLE IF NOT EXISTS schema_migrations (
    revision    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('versioned', 'repeatable')),
    store       TEXT NOT NULL CHECK (store IN ('postgres', 'vault', 'env', 'valkey')),
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_by  TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
GRANT SELECT ON schema_migrations TO mira_dbuser;
-- END GENERATED LEDGER DDL

-- Extension members (pgcrypto, vector, pg_trgm objects living in public) stay
-- with the superuser: an extension's objects must keep the extension's owner
-- or ALTER EXTENSION UPDATE and dump/restore refuse them.
DO $adopt_ownership$
DECLARE
    obj RECORD;
BEGIN
    FOR obj IN
        SELECT c.oid, c.relkind, format('%I.%I', n.nspname, c.relname) AS ident
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
           AND c.relowner <> 'mira_admin'::regrole
           AND NOT EXISTS (
               SELECT 1 FROM pg_depend d
                WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype = 'e')
         ORDER BY c.relname
    LOOP
        -- ALTER TABLE carries the table's indexes, triggers, policies, and
        -- column-owned sequences with it.
        EXECUTE format(
            CASE obj.relkind
                WHEN 'v' THEN 'ALTER VIEW %s OWNER TO mira_admin'
                WHEN 'm' THEN 'ALTER MATERIALIZED VIEW %s OWNER TO mira_admin'
                WHEN 'f' THEN 'ALTER FOREIGN TABLE %s OWNER TO mira_admin'
                ELSE 'ALTER TABLE %s OWNER TO mira_admin'
            END,
            obj.ident);
    END LOOP;

    FOR obj IN
        SELECT c.oid, format('%I.%I', n.nspname, c.relname) AS ident
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relkind = 'S'
           AND c.relowner <> 'mira_admin'::regrole
           AND NOT EXISTS (
               SELECT 1 FROM pg_depend d
                WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype = 'e')
         ORDER BY c.relname
    LOOP
        EXECUTE format('ALTER SEQUENCE %s OWNER TO mira_admin', obj.ident);
    END LOOP;

    FOR obj IN
        SELECT p.oid, format('%I.%I(%s)', n.nspname, p.proname, pg_get_function_identity_arguments(p.oid)) AS ident,
               p.prokind
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public'
           AND p.proowner <> 'mira_admin'::regrole
           AND NOT EXISTS (
               SELECT 1 FROM pg_depend d
                WHERE d.classid = 'pg_proc'::regclass AND d.objid = p.oid AND d.deptype = 'e')
         ORDER BY p.proname, pg_get_function_identity_arguments(p.oid)
    LOOP
        EXECUTE format(
            CASE obj.prokind WHEN 'p' THEN 'ALTER PROCEDURE %s OWNER TO mira_admin'
                             WHEN 'a' THEN 'ALTER AGGREGATE %s OWNER TO mira_admin'
                             ELSE 'ALTER FUNCTION %s OWNER TO mira_admin' END,
            obj.ident);
    END LOOP;
END
$adopt_ownership$;

GRANT CREATE ON SCHEMA public TO mira_admin;
