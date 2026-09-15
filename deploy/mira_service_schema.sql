-- MIRA fresh-install database contract
--
-- Preconditions:
--   * Connect to an empty mira_service database as the schema owner.
--   * Provision mira_admin and mira_dbuser, including credentials and BYPASSRLS
--     for mira_admin, through Vault-backed deployment tooling before this file.
--     deploy/postgresql.sh performs both steps.
--
-- This is deliberately not a migration. It contains no compatibility DDL,
-- embedded credentials, database creation, or default privileges.
-- mira-OSS 2.0 is a fresh install: 1.x -> 2.0 is a reinstall, not an upgrade.

DO $schema_precondition$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
    ) THEN
        RAISE EXCEPTION 'mira_service_schema.sql requires an empty target database';
    END IF;
END
$schema_precondition$;

CREATE EXTENSION pgcrypto;
CREATE EXTENSION vector;
CREATE EXTENSION pg_trgm;

GRANT USAGE ON SCHEMA public TO mira_dbuser;

CREATE FUNCTION set_updated_at()
RETURNS TRIGGER AS $function$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END
$function$ LANGUAGE plpgsql;

CREATE FUNCTION set_search_vector()
RETURNS TRIGGER AS $function$
BEGIN
    NEW.search_vector = to_tsvector('english', NEW.text);
    RETURN NEW;
END
$function$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- Fixed model routing
-- ---------------------------------------------------------------------------

CREATE TABLE model_configs (
    name TEXT PRIMARY KEY CHECK (name IN ('primary', 'fast', 'batch', 'assessment', 'other')),
    model TEXT NOT NULL,
    dialect_name TEXT NOT NULL CHECK (dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq')),
    endpoint_url TEXT NOT NULL,
    api_key_name TEXT NOT NULL,
    effort TEXT CHECK (effort IN ('none', 'low', 'medium', 'high', 'xhigh', 'max')),
    max_tokens INTEGER NOT NULL CHECK (max_tokens > 0)
);

-- Seed values match the reference deployment: main chat (primary) is the
-- only consumer of the local llama-server (single KV-cache slot), so no
-- async subsystem call can evict the main conversation's cached prefix
-- between turns; every auxiliary route goes to poolside models on
-- OpenRouter, sharing the 'subcortical_key' credential at
-- secret/mira/api_keys. primary needs no credential ('' — the local
-- llama-server is unauthenticated). Truly offline installs can still rewrite
-- endpoint_url/model through the OFFLINE_SQL block in deploy/postgresql.sh.
-- Route assignments live in the cns/services call sites: peanut gallery,
-- forage overwatch, and domaindoc descriptor expansion ride 'fast';
-- summaries, compaction, persona/portrait/LoRA/user-model, memory curator,
-- and the repulsion rewriter ride 'batch'.
INSERT INTO model_configs (name, model, dialect_name, endpoint_url, api_key_name, effort, max_tokens)
VALUES
    ('primary', 'Qwen3.8 27B Uncensored Q8_0', 'openai', 'http://192.168.1.9:3090/v1/chat/completions', '', 'high', 16000),
    ('fast', 'poolside/laguna-xs-2.1', 'openai', 'https://openrouter.ai/api/v1/chat/completions', 'subcortical_key', 'none', 4096),
    ('batch', 'poolside/laguna-s-2.1', 'openai', 'https://openrouter.ai/api/v1/chat/completions', 'subcortical_key', 'high', 16000),
    ('assessment', 'poolside/laguna-s-2.1', 'openai', 'https://openrouter.ai/api/v1/chat/completions', 'subcortical_key', 'none', 10000),
    ('other', 'poolside/laguna-s-2.1', 'openai', 'https://openrouter.ai/api/v1/chat/completions', 'subcortical_key', 'high', 10000);

-- ---------------------------------------------------------------------------
-- Cost visibility
-- ---------------------------------------------------------------------------

-- Keyed by model_configs route name, not by model string. NULL prices
-- auto-resolve from the published fallbacks in utils/cost_accumulator.py;
-- NOT NULL is a manual override. __default__ is the reserved fallback key for
-- any endpoint/model pair without an explicit row.
CREATE TABLE usage_pricing (
    name VARCHAR(50) PRIMARY KEY,
    input_price_per_mtok DECIMAL(10,6),
    output_price_per_mtok DECIMAL(10,6),
    cache_read_price_per_mtok DECIMAL(10,6),
    cache_write_price_per_mtok DECIMAL(10,6),
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE
);

INSERT INTO usage_pricing (name, input_price_per_mtok, output_price_per_mtok)
VALUES ('__default__', 5.000000, 25.000000);
-- Poolside prices are OpenRouter's published rates (USD per Mtok, 2026-09):
-- laguna-xs-2.1 0.06/0.12, laguna-s-2.1 0.09/0.18. primary (local
-- llama-server) is seeded unpriced: marginal cost ~0, and NULL fields fall
-- through to the fallback tiers per utils/cost_accumulator.py.
INSERT INTO usage_pricing (name, input_price_per_mtok, output_price_per_mtok) VALUES
    ('primary', NULL, NULL),
    ('fast', 0.060000, 0.120000),
    ('batch', 0.090000, 0.180000),
    ('assessment', 0.090000, 0.180000),
    ('other', 0.090000, 0.180000);

-- ---------------------------------------------------------------------------
-- Accounts and authentication-ready contracts
-- ---------------------------------------------------------------------------

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    webauthn_credentials JSONB NOT NULL DEFAULT '{}'::jsonb,
    memory_manipulation_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    daily_manipulation_last_run TIMESTAMPTZ,
    timezone VARCHAR(100) NOT NULL,
    temperature_unit VARCHAR(20) NOT NULL DEFAULT 'fahrenheit'
        CHECK (temperature_unit IN ('fahrenheit', 'celsius')),

    subject_kind TEXT NOT NULL DEFAULT 'member' CHECK (subject_kind IN ('member', 'demo')),

    cumulative_activity_days INTEGER NOT NULL DEFAULT 0 CHECK (cumulative_activity_days >= 0),
    last_activity_date DATE,
    portrait TEXT,
    portrait_generated_at TIMESTAMPTZ,

    -- Account garbage collection. Replaces the 1.x users_trash table: a
    -- soft-deleted user keeps its row, and the deadline drives the purge job.
    deletion_requested_at TIMESTAMPTZ,
    soft_deleted_at TIMESTAMPTZ,
    purge_deadline TIMESTAMPTZ,

    -- Reserved for a future demo subject. Nullable and unpoliced by design:
    -- subject_kind admits 'demo' now so admitting it later is not a schema change.
    demo_start_at TIMESTAMPTZ,
    demo_expires_at TIMESTAMPTZ
);

CREATE TABLE magic_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_magic_links_user ON magic_links(user_id);

CREATE TABLE api_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL DEFAULT 'API Token',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE INDEX idx_api_tokens_hash_active ON api_tokens(token_hash)
    WHERE revoked_at IS NULL;
CREATE INDEX idx_api_tokens_user_active ON api_tokens(user_id)
    WHERE revoked_at IS NULL;
CREATE UNIQUE INDEX idx_api_tokens_user_name_active ON api_tokens(user_id, name)
    WHERE revoked_at IS NULL;

CREATE TABLE user_activity_days (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    activity_date DATE NOT NULL,
    first_message_at TIMESTAMPTZ NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 1 CHECK (message_count > 0),
    PRIMARY KEY (user_id, activity_date)
);

-- ---------------------------------------------------------------------------
-- Domain knowledge (Letta agent memory blocks)
-- ---------------------------------------------------------------------------

CREATE TABLE domain_knowledge_blocks (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    domain_label VARCHAR(100) NOT NULL,
    domain_name VARCHAR(255) NOT NULL,
    block_description TEXT NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_domain_knowledge_blocks_user
    ON domain_knowledge_blocks(user_id, domain_label);

CREATE TABLE domain_knowledge_block_content (
    id SERIAL PRIMARY KEY,
    block_id INTEGER NOT NULL UNIQUE REFERENCES domain_knowledge_blocks(id) ON DELETE CASCADE,
    block_value TEXT NOT NULL,
    letta_block_id VARCHAR(255),
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Domain-document sharing and federation
-- ---------------------------------------------------------------------------

CREATE TABLE domaindoc_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    domaindoc_label TEXT NOT NULL,
    collaborator_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'rejected', 'revoked')),
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accepted_at TIMESTAMPTZ,
    UNIQUE (owner_user_id, domaindoc_label, collaborator_user_id),
    CHECK (owner_user_id <> collaborator_user_id)
);

CREATE INDEX idx_domaindoc_shares_collaborator ON domaindoc_shares(collaborator_user_id, status);
CREATE INDEX idx_domaindoc_shares_owner ON domaindoc_shares(owner_user_id, status);
CREATE INDEX idx_domaindoc_shares_label ON domaindoc_shares(domaindoc_label);

-- Cross-user identity for domaindoc sharing. RLS on users is unconditional, so a
-- collaborator is not readable by the party sharing with them: the lookup by email
-- returns nothing, and because the share listings inner-join users the whole share row
-- disappears -- a share that reads as lost data rather than as a hidden profile.
-- These two functions publish only the three columns needed to name a counterparty.
-- portrait, webauthn_credentials and the soft-delete timestamps stay unreachable, and
-- an inactive account resolves to zero rows. SECURITY DEFINER with its own predicate
-- means the result does not depend on the caller's row-level-security context.
CREATE FUNCTION resolve_active_user_identity(p_email text)
RETURNS TABLE (id uuid, email varchar, first_name varchar)
STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
LANGUAGE sql
AS $function$
    SELECT u.id, u.email, u.first_name
    FROM public.users u
    WHERE u.email = p_email
      AND u.is_active = TRUE
$function$;

CREATE FUNCTION active_user_identity(p_user_id uuid)
RETURNS TABLE (id uuid, email varchar, first_name varchar)
STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
LANGUAGE sql
AS $function$
    SELECT u.id, u.email, u.first_name
    FROM public.users u
    WHERE u.id = p_user_id
      AND u.is_active = TRUE
$function$;

-- ---------------------------------------------------------------------------
-- Continuum and messages
-- ---------------------------------------------------------------------------

CREATE TABLE continuums (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (id, user_id)
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    continuum_id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT COMPRESSION lz4 NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_call_id TEXT,
    is_error BOOLEAN NOT NULL DEFAULT FALSE,
    segment_embedding vector(768),
    FOREIGN KEY (continuum_id, user_id)
        REFERENCES continuums(id, user_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_user ON messages(user_id);
CREATE INDEX idx_messages_continuum_time ON messages(continuum_id, created_at);
CREATE INDEX idx_messages_tool_call ON messages(tool_call_id) WHERE tool_call_id IS NOT NULL;
CREATE UNIQUE INDEX idx_messages_active_segment_unique ON messages(continuum_id)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' = 'active';
CREATE INDEX idx_messages_active_segments ON messages(continuum_id, created_at)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' IN ('active', 'paused');
CREATE INDEX idx_messages_segment_metadata ON messages USING gin(metadata)
    WHERE metadata->>'is_segment_boundary' = 'true';
CREATE INDEX idx_messages_segment_embedding ON messages
    USING hnsw (segment_embedding vector_cosine_ops)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND segment_embedding IS NOT NULL;

CREATE TRIGGER continuums_updated_at
BEFORE UPDATE ON continuums
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Long-term memory and entities
-- ---------------------------------------------------------------------------

CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text TEXT COMPRESSION lz4 NOT NULL,
    embedding vector(768),
    search_vector tsvector,
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 0.5
        CHECK (importance_score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    happens_at TIMESTAMPTZ,
    inbound_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    outbound_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    entity_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    archived_at TIMESTAMPTZ,
    last_tended_at TIMESTAMPTZ,
    activity_days_at_creation INTEGER,
    activity_days_at_last_access INTEGER,
    annotations JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_segment_id UUID
);

CREATE INDEX idx_memories_user ON memories(user_id);
CREATE INDEX idx_memories_search_vector ON memories USING gin(search_vector);
CREATE INDEX idx_memories_embedding_ivfflat ON memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_memories_source_segment ON memories(source_segment_id)
    WHERE source_segment_id IS NOT NULL;
CREATE INDEX idx_memories_floor_candidates ON memories(importance_score, last_tended_at)
    WHERE is_archived = FALSE;

CREATE TRIGGER memories_search_vector
BEFORE INSERT OR UPDATE OF text ON memories
FOR EACH ROW EXECUTE FUNCTION set_search_vector();
CREATE TRIGGER memories_updated_at
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Administrator-curated shared memory. Deliberately has no RLS: isolation is
-- enforced by the security-barrier view below, which is the only runtime read
-- path. Direct DML is revoked from the application role in the grants section,
-- so the shared table cannot be written from a request path.
CREATE TABLE global_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT COMPRESSION lz4 NOT NULL,
    embedding vector(768),
    search_vector tsvector,
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 1.0
        CHECK (importance_score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    happens_at TIMESTAMPTZ,
    entity_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    inbound_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    outbound_links JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    archived_at TIMESTAMPTZ
);

CREATE FUNCTION can_read_global_memories()
RETURNS BOOLEAN
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
LANGUAGE sql
AS $function$
    SELECT EXISTS (
        SELECT 1
        FROM public.users
        WHERE id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
          AND is_active = TRUE
    )
$function$;

CREATE VIEW global_memories_runtime
WITH (security_barrier = true)
AS
SELECT *
FROM global_memories
WHERE can_read_global_memories();

CREATE INDEX idx_global_memories_search_vector ON global_memories USING gin(search_vector);
CREATE INDEX idx_global_memories_embedding_ivfflat ON global_memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TRIGGER global_memories_search_vector
BEFORE INSERT OR UPDATE OF text ON global_memories
FOR EACH ROW EXECUTE FUNCTION set_search_vector();
CREATE TRIGGER global_memories_updated_at
BEFORE UPDATE ON global_memories
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    embedding vector(300),
    link_count INTEGER NOT NULL DEFAULT 0,
    last_linked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    archived_at TIMESTAMPTZ,
    UNIQUE (user_id, name, entity_type)
);

CREATE INDEX idx_entities_user ON entities(user_id);
CREATE INDEX idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);

CREATE TRIGGER entities_updated_at
BEFORE UPDATE ON entities
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Persona: immutable behavioral directives and evaluation evidence
-- ---------------------------------------------------------------------------

CREATE TABLE persona_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL CHECK (revision_number > 0),
    directives TEXT COMPRESSION lz4 NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('baseline', 'automatic', 'user', 'rollback')),
    parent_revision_id UUID,
    evidence_ids UUID[] NOT NULL DEFAULT '{}'::uuid[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, revision_number),
    UNIQUE (id, user_id),
    FOREIGN KEY (parent_revision_id, user_id)
        REFERENCES persona_revisions(id, user_id) ON DELETE RESTRICT
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_persona_revisions_user_time ON persona_revisions(user_id, revision_number DESC);

CREATE TABLE persona_state (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    current_revision_id UUID NOT NULL,
    latest_evaluated_segment_id UUID,
    refinement_checkpoint_activity_day INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (current_revision_id, user_id)
        REFERENCES persona_revisions(id, user_id) ON DELETE RESTRICT
);

-- Persona evaluation evidence. Distinct from feedback_signals, which serves the
-- user-model pipeline: the two tables share no columns beyond their keys.
CREATE TABLE persona_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    continuum_id UUID NOT NULL,
    segment_id UUID NOT NULL,
    behavioral_section TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('alignment', 'misalignment', 'contextual_pass')),
    strength TEXT NOT NULL CHECK (strength IN ('strong', 'moderate', 'mild')),
    evidence TEXT NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_by_revision_id UUID,
    FOREIGN KEY (continuum_id, user_id)
        REFERENCES continuums(id, user_id) ON DELETE CASCADE,
    FOREIGN KEY (consumed_by_revision_id, user_id)
        REFERENCES persona_revisions(id, user_id) ON DELETE SET NULL
);

CREATE INDEX idx_persona_signals_user ON persona_signals(user_id, evaluated_at);
CREATE INDEX idx_persona_signals_unconsumed ON persona_signals(user_id, evaluated_at)
    WHERE consumed_by_revision_id IS NULL;

CREATE TRIGGER persona_state_updated_at
BEFORE UPDATE ON persona_state
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Baseline provisioning is a DB trigger, not an application code path, so no
-- user can exist without revision 1 -- including the row the local-session
-- bootstrap provisions in single mode.
CREATE FUNCTION provision_baseline_persona()
RETURNS TRIGGER AS $function$
DECLARE
    baseline_revision_id UUID;
BEGIN
    INSERT INTO persona_revisions (
        user_id,
        revision_number,
        directives,
        source
    )
    VALUES (NEW.id, 1, '', 'baseline')
    RETURNING id INTO baseline_revision_id;

    INSERT INTO persona_state (user_id, current_revision_id)
    VALUES (NEW.id, baseline_revision_id);

    RETURN NEW;
END
$function$ LANGUAGE plpgsql;

CREATE TRIGGER users_provision_baseline_persona
AFTER INSERT ON users
FOR EACH ROW EXECUTE FUNCTION provision_baseline_persona();

-- ---------------------------------------------------------------------------
-- User model (DIY reinforcement loop)
-- ---------------------------------------------------------------------------

CREATE TABLE feedback_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    segment_id UUID NOT NULL,
    continuum_id UUID NOT NULL,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('alignment', 'misalignment', 'contextual_pass')),
    section_id TEXT NOT NULL,
    strength TEXT NOT NULL CHECK (strength IN ('strong', 'moderate', 'mild')),
    evidence TEXT NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    synthesized BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_signals_user_id ON feedback_signals(user_id);
CREATE INDEX idx_feedback_signals_user_type ON feedback_signals(user_id, signal_type);
CREATE INDEX idx_feedback_signals_unsynthesized ON feedback_signals(user_id)
    WHERE NOT synthesized;
CREATE INDEX idx_feedback_signals_section_id ON feedback_signals(user_id, section_id)
    WHERE NOT synthesized;

CREATE TABLE feedback_synthesis_tracking (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    activity_days_at_last_synthesis INTEGER NOT NULL DEFAULT 0,
    last_synthesis_at TIMESTAMPTZ,
    last_synthesis_output TEXT,
    needs_checkin BOOLEAN NOT NULL DEFAULT FALSE,
    checkin_response TEXT
);

-- ---------------------------------------------------------------------------
-- Lattice federation: global username registry
-- ---------------------------------------------------------------------------
-- Maps federated usernames to user_ids so the Lattice discovery daemon can
-- resolve inbound `username@server` addresses to a local recipient.
-- No RLS: this is a global routing/lookup table, queried contextlessly by
-- mira_resolve_username, which runs outside any user context. A trigger or
-- policy requiring matching user context would break the federation resolver.
-- Application logic in pager_tool._register_username only ever inserts the
-- current user's own row, and the UNIQUE constraint prevents duplicate
-- registrations.

CREATE TABLE global_usernames (
    username VARCHAR(20) PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_global_usernames_active ON global_usernames(username) WHERE active = TRUE;

-- ---------------------------------------------------------------------------
-- User feedback (developer-facing friction signals)
-- ---------------------------------------------------------------------------

CREATE TABLE user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('feature_request', 'bug_report', 'confusion', 'praise', 'other')),
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_feedback_category_time ON user_feedback(category, created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-level security. Missing user context resolves to NULL and sees no rows.
-- Privileged pre-auth and scheduler repositories use mira_admin/BYPASSRLS.
-- ---------------------------------------------------------------------------

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY users_user_policy ON users
    FOR ALL TO mira_dbuser
    USING (id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE magic_links ENABLE ROW LEVEL SECURITY;
CREATE POLICY magic_links_user_policy ON magic_links
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE api_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY api_tokens_user_policy ON api_tokens
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE user_activity_days ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_activity_days_user_policy ON user_activity_days
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE domain_knowledge_blocks ENABLE ROW LEVEL SECURITY;
CREATE POLICY domain_knowledge_blocks_user_policy ON domain_knowledge_blocks
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE domain_knowledge_block_content ENABLE ROW LEVEL SECURITY;
CREATE POLICY domain_knowledge_block_content_user_policy ON domain_knowledge_block_content
    FOR ALL TO mira_dbuser
    USING (block_id IN (
        SELECT id FROM domain_knowledge_blocks
        WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    ))
    WITH CHECK (block_id IN (
        SELECT id FROM domain_knowledge_blocks
        WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    ));

ALTER TABLE continuums ENABLE ROW LEVEL SECURITY;
CREATE POLICY continuums_user_policy ON continuums
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY messages_user_policy ON messages
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY memories_user_policy ON memories
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
CREATE POLICY entities_user_policy ON entities
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE persona_revisions ENABLE ROW LEVEL SECURITY;
CREATE POLICY persona_revisions_user_policy ON persona_revisions
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE persona_state ENABLE ROW LEVEL SECURITY;
CREATE POLICY persona_state_user_policy ON persona_state
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE persona_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY persona_signals_user_policy ON persona_signals
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE feedback_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY feedback_signals_user_policy ON feedback_signals
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE feedback_synthesis_tracking ENABLE ROW LEVEL SECURITY;
CREATE POLICY feedback_synthesis_tracking_user_policy ON feedback_synthesis_tracking
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_feedback_user_policy ON user_feedback
    FOR ALL TO mira_dbuser
    USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid);

ALTER TABLE domaindoc_shares ENABLE ROW LEVEL SECURITY;
CREATE POLICY domaindoc_shares_owner_policy ON domaindoc_shares
    FOR ALL TO mira_dbuser
    USING (
        owner_user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    )
    WITH CHECK (
        owner_user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    );
CREATE POLICY domaindoc_shares_collaborator_select_policy ON domaindoc_shares
    FOR SELECT TO mira_dbuser
    USING (
        collaborator_user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    );
CREATE POLICY domaindoc_shares_collaborator_update_policy ON domaindoc_shares
    FOR UPDATE TO mira_dbuser
    USING (
        collaborator_user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    )
    WITH CHECK (
        collaborator_user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
    );

-- ---------------------------------------------------------------------------
-- Explicit least-privilege runtime grants
-- ---------------------------------------------------------------------------

-- Deployment applies this schema as the PostgreSQL superuser while the
-- privileged control plane connects as mira_admin with BYPASSRLS.
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mira_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mira_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mira_admin;

GRANT SELECT ON model_configs TO mira_dbuser;
GRANT SELECT ON usage_pricing TO mira_dbuser;
GRANT SELECT, UPDATE ON users TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON magic_links, api_tokens TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_activity_days TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON continuums, messages, memories, entities TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON domain_knowledge_blocks, domain_knowledge_block_content TO mira_dbuser;
GRANT SELECT ON global_memories_runtime TO mira_dbuser;
GRANT SELECT, INSERT ON persona_revisions TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE ON persona_state TO mira_dbuser;
GRANT SELECT, INSERT ON persona_signals TO mira_dbuser;
GRANT UPDATE (consumed_by_revision_id) ON persona_signals TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE ON feedback_signals TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE ON feedback_synthesis_tracking TO mira_dbuser;
GRANT SELECT, INSERT, DELETE ON domaindoc_shares TO mira_dbuser;
GRANT UPDATE (status, accepted_at) ON domaindoc_shares TO mira_dbuser;
GRANT INSERT ON user_feedback TO mira_dbuser;
GRANT SELECT, INSERT, UPDATE ON global_usernames TO mira_dbuser;
GRANT EXECUTE ON FUNCTION can_read_global_memories() TO mira_dbuser;
GRANT EXECUTE ON FUNCTION resolve_active_user_identity(text) TO mira_dbuser;
GRANT EXECUTE ON FUNCTION active_user_identity(uuid) TO mira_dbuser;

-- The shared curated table is read through global_memories_runtime only, and is
-- curated out-of-band via psql as mira_admin. 1.x granted the runtime role full
-- DML here, which at N>1 lets any authenticated request path write every other
-- user's shared context.
REVOKE ALL ON global_memories FROM mira_dbuser;

REVOKE EXECUTE ON FUNCTION set_updated_at() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION set_search_vector() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION can_read_global_memories() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION provision_baseline_persona() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION resolve_active_user_identity(text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION active_user_identity(uuid) FROM PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------

COMMENT ON TABLE model_configs IS 'Exactly five required MIRA routes, each owning dialect, model, endpoint, Vault key, default effort, and output ceiling: primary (main chat only — the sole consumer of the local llama-server, so async subsystem calls cannot evict its KV-cache prefix), fast (latency-critical small turns: subcortical analysis, peanut gallery, forage overwatch, descriptor expansion), batch (bulk background work: segment summaries, live-context compaction, persona/portrait/LoRA/user-model synthesis, memory curator, forage, while-the-cat-is-away, repulsion rewriter), assessment (assessment extraction), other (a sidebar turn routed to an outside model, deliberately a different vendor from primary).';
COMMENT ON TABLE usage_pricing IS 'Per-route cost lookup keyed by model_configs name; __default__ is the reserved fallback pair.';
COMMENT ON TABLE users IS 'MIRA account. subject_kind admits member and demo; only member is provisioned today.';
COMMENT ON TABLE persona_revisions IS 'Immutable Persona directive history.';
COMMENT ON TABLE persona_state IS 'Current Persona pointer and automatic-refinement checkpoints.';
COMMENT ON TABLE persona_signals IS 'Evidence about MIRA behavior consumed by Persona revisions.';
COMMENT ON TABLE feedback_signals IS 'Assessment signals for the user-model pipeline.';
COMMENT ON TABLE feedback_synthesis_tracking IS 'User-model synthesis state; last_synthesis_output holds the user-model XML.';
COMMENT ON TABLE global_memories IS 'Administrator-curated global memory; runtime reads go through global_memories_runtime.';
COMMENT ON TABLE domaindoc_shares IS 'Cross-user domaindoc sharing with consent flow (pending to accepted).';
COMMENT ON TABLE user_feedback IS 'User-submitted friction signals captured proactively by Mira - developer queries this directly for rapid iteration.';
COMMENT ON TABLE domain_knowledge_blocks IS 'Domain-specific knowledge blocks for the Letta agent memory system.';
