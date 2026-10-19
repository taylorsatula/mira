-- Mira Blog Database Schema
-- Creates the roles, `posts` table, and default privileges for a separately
-- created `mira_blog` database.
--
-- No in-repo caller: nothing in MIRA reads or migrates this schema. An operator
-- runs it with psql against a mira_blog database they created; the app has no
-- code path to it. Do not build on it — a blog feature belongs in
-- deploy/mira_service_schema.sql plus a step in deploy/migrations/, not here.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_owner') THEN
        CREATE ROLE mira_blog_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_reader') THEN
        CREATE ROLE mira_blog_reader LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_writer') THEN
        CREATE ROLE mira_blog_writer LOGIN;
    END IF;
END
$$;

REVOKE CONNECT ON DATABASE mira_blog FROM PUBLIC;
GRANT CONNECT ON DATABASE mira_blog TO mira_blog_reader, mira_blog_writer;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO mira_blog_reader, mira_blog_writer;
GRANT ALL ON SCHEMA public TO mira_blog_owner;

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL DEFAULT 'GENERAL',
    content_markdown TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_posts_slug ON posts(slug);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);

ALTER TABLE posts OWNER TO mira_blog_owner;
ALTER SEQUENCE posts_id_seq OWNER TO mira_blog_owner;

REVOKE ALL ON posts FROM PUBLIC;
GRANT SELECT ON posts TO mira_blog_reader;
GRANT SELECT, INSERT, UPDATE, DELETE ON posts TO mira_blog_writer;

REVOKE ALL ON SEQUENCE posts_id_seq FROM PUBLIC;
GRANT USAGE, SELECT ON SEQUENCE posts_id_seq TO mira_blog_writer;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT SELECT ON TABLES TO mira_blog_reader;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mira_blog_writer;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO mira_blog_writer;
