"""Contract tests for the mira-OSS 2.0 greenfield schema.

The schema file is the single DDL source of truth: there are no migrations and
no 1.x upgrade path. These tests read the file statically so the contract is
guarded without a live PostgreSQL.

Two rules govern what is asserted here:

* Table matching tolerates both `CREATE TABLE foo (` and
  `CREATE TABLE IF NOT EXISTS foo (`. The bare form is what the file uses today;
  the optional form is matched anyway so that a style change can never make an
  assertion match nothing and pass vacuously.
* The absence list is mira-OSS's, not the CRM variant's. That variant
  omits `usage_pricing`,
  `domain_knowledge_blocks`, `domain_knowledge_block_content` and
  `feedback_synthesis_tracking`; mira-OSS retains all four (D5, the OSS domain
  knowledge feature, D1). See the backport plan 0.2 and 6.3.7.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "deploy" / "mira_service_schema.sql").read_text(encoding="utf-8")

# Optional `IF NOT EXISTS` in both helpers, so neither can silently stop matching.
_CREATE_TABLE = r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+"


def _defines_table(table: str) -> bool:
    return re.search(rf"{_CREATE_TABLE}{re.escape(table)}\b", SCHEMA) is not None


def _table_body(table: str) -> str:
    match = re.search(
        rf"{_CREATE_TABLE}{re.escape(table)}\s*\((.*?)\n\);",
        SCHEMA,
        re.DOTALL,
    )
    assert match, f"table {table} is missing"
    return match.group(1)


def _seed_rows() -> dict[str, dict[str, str]]:
    """Parse the model_configs seed into {route: row fields}."""
    pattern = (
        r"\('(primary|fast|batch|assessment|other)', '([^']+)', '([a-z]+)', "
        r"'([^']+)', '([a-z_]+)', '(none|low|medium|high|xhigh|max)', (\d+)\)"
    )
    rows = {}
    for m in re.finditer(pattern, SCHEMA):
        name, model, dialect, endpoint, key_name, effort, max_tokens = m.groups()
        rows[name] = {
            "model": model,
            "dialect_name": dialect,
            "endpoint_url": endpoint,
            "api_key_name": key_name,
            "effort": effort,
            "max_tokens": max_tokens,
        }
    return rows


# ---------------------------------------------------------------------------
# Install posture
# ---------------------------------------------------------------------------


def test_schema_is_a_clean_install_contract() -> None:
    assert "requires an empty target database" in SCHEMA
    assert "CREATE DATABASE" not in SCHEMA
    assert "CREATE ROLE" not in SCHEMA
    assert "PASSWORD '" not in SCHEMA
    assert "ALTER DEFAULT PRIVILEGES" not in SCHEMA
    assert "DROP TRIGGER" not in SCHEMA
    assert "CREATE EXTENSION pgcrypto;" in SCHEMA
    assert "CREATE EXTENSION vector;" in SCHEMA
    assert "CREATE EXTENSION pg_trgm;" in SCHEMA
    assert "uuid-ossp" not in SCHEMA
    # No upgrade-only idempotency scaffolding: 1.x -> 2.0 is a reinstall.
    assert "DO $$" not in SCHEMA
    assert "$schema_precondition$" in SCHEMA


def test_helpers_are_defined_before_use() -> None:
    assert re.search(r"CREATE FUNCTION set_updated_at\(\)", SCHEMA)
    assert re.search(r"CREATE FUNCTION set_search_vector\(\)", SCHEMA)
    assert "GRANT USAGE ON SCHEMA public TO mira_dbuser;" in SCHEMA


# ---------------------------------------------------------------------------
# Object set
# ---------------------------------------------------------------------------

# Retired by the 2.0 decisions, and CRM/billing objects mira-OSS never had.
REMOVED_TABLES = {
    # D4: replaced by model_configs.
    "conversation_llm",
    "internal_llm",
    # Replaced by the soft-delete columns on users.
    "users_trash",
    # D10: the Anthropic Batch API is gone.
    "extraction_batches",
    "post_processing_batches",
    # D7: billing.
    "stripe_customers",
    "stripe_subscriptions",
    "stripe_webhook_events",
    "entitlements",
    "billing_renewal_attempts",
    "billing_transactions",
    "billing_events",
    # Audit journal (crm-only; mira-OSS logs diagnostics instead).
    "audit_events",
    # CRM product surface.
    "crm_workspaces",
    "sms_threads",
    "sms_messages",
    "workphone_integrations",
    "workphone_messages",
    "workphone_opt_outs",
    "business_voice_directives",
    "conversation_outcomes",
    "voice_feedback_signals",
    "autonomy_assessment_sft_corpus",
    "push_subscriptions",
}

# mira-OSS retains these; the CRM variant dropped them. Asserted present so a future
# re-derivation from the CRM variant's file cannot silently lose a subsystem.
OSS_RETAINED_TABLES = {
    "usage_pricing",                    # D5
    "domain_knowledge_blocks",          # OSS feature
    "domain_knowledge_block_content",   # OSS feature
    "feedback_synthesis_tracking",      # D1
    "feedback_signals",                 # D1 (OSS column set)
    "user_feedback",                    # WP1 feedback_tool
}

CORE_TABLES = {
    "model_configs",
    "users",
    "magic_links",
    "api_tokens",
    "user_activity_days",
    "continuums",
    "messages",
    "memories",
    "global_memories",
    "entities",
    "persona_revisions",
    "persona_state",
    "persona_signals",
    "domaindoc_shares",
    "global_usernames",
}


def test_removed_tables_are_absent() -> None:
    for table in sorted(REMOVED_TABLES):
        assert not _defines_table(table), f"{table} must not exist in the 2.0 schema"


def test_oss_retained_tables_are_present() -> None:
    for table in sorted(OSS_RETAINED_TABLES):
        assert _defines_table(table), f"{table} is an OSS-retained table and is missing"


def test_core_tables_are_present() -> None:
    for table in sorted(CORE_TABLES):
        assert _defines_table(table), f"{table} is missing"


def test_persona_evidence_and_user_model_signals_are_separate_tables() -> None:
    """crm overloaded `feedback_signals` with a disjoint Persona column set.

    mira-OSS keeps `feedback_signals` for the user model and names Persona's
    table `persona_signals`. Both must exist, and neither may be a hybrid.
    """
    assert _defines_table("persona_signals")
    assert _defines_table("feedback_signals")

    persona = _table_body("persona_signals")
    assert "outcome TEXT NOT NULL CHECK (outcome IN ('alignment', 'misalignment', 'contextual_pass'))" in persona
    assert "behavioral_section TEXT NOT NULL" in persona
    assert "consumed_by_revision_id" in persona
    assert "REFERENCES persona_revisions(id, user_id)" in persona
    assert "signal_type" not in persona
    assert "synthesized" not in persona

    user_model = _table_body("feedback_signals")
    assert "signal_type TEXT NOT NULL CHECK (signal_type IN ('alignment', 'misalignment', 'contextual_pass'))" in user_model
    assert "section_id TEXT NOT NULL" in user_model
    assert "synthesized BOOLEAN NOT NULL DEFAULT FALSE" in user_model
    assert "behavioral_section" not in user_model
    assert "consumed_by_revision_id" not in user_model

    # Every FK, index, policy and grant crm hung on its feedback_signals now
    # points at persona_signals.
    assert "ON persona_signals" in SCHEMA
    assert "idx_persona_signals_user" in SCHEMA
    assert "idx_persona_signals_unconsumed" in SCHEMA
    assert "GRANT SELECT, INSERT ON persona_signals TO mira_dbuser;" in SCHEMA
    assert "GRANT UPDATE (consumed_by_revision_id) ON persona_signals TO mira_dbuser;" in SCHEMA
    assert "ALTER TABLE persona_signals ENABLE ROW LEVEL SECURITY;" in SCHEMA


def test_schema_table_count_matches_the_decided_set() -> None:
    declared = set(re.findall(rf"{_CREATE_TABLE}([a-z_]+)", SCHEMA))
    assert declared == CORE_TABLES | OSS_RETAINED_TABLES, sorted(
        declared ^ (CORE_TABLES | OSS_RETAINED_TABLES)
    )


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------


def test_users_exposes_auth_lifecycle_and_subject_contract() -> None:
    body = _table_body("users")
    auth_columns = {
        "id",
        "email",
        "first_name",
        "last_name",
        "is_active",
        "created_at",
        "last_login_at",
        "webauthn_credentials",
        "memory_manipulation_enabled",
        "daily_manipulation_last_run",
        "timezone",
    }
    lifecycle_columns = {
        "subject_kind",
        "temperature_unit",
        "cumulative_activity_days",
        "last_activity_date",
        "portrait",
        "portrait_generated_at",
        "deletion_requested_at",
        "soft_deleted_at",
        "purge_deadline",
        "demo_start_at",
        "demo_expires_at",
    }
    for column in auth_columns | lifecycle_columns:
        assert re.search(rf"^\s*{column}\s", body, re.MULTILINE), column

    assert "subject_kind IN ('member', 'demo')" in body
    # D12 ships member-only, but the column and its CHECK stay so admitting demo
    # later is not a schema change. The soft-delete columns replace users_trash.
    assert "webauthn_credentials JSONB NOT NULL DEFAULT '{}'::jsonb" in body
    assert "timezone VARCHAR(100) NOT NULL" in body

    for column in ("conversation_llm", "balance_usd"):
        assert not re.search(rf"^\s*{column}\s", body, re.MULTILINE), (
            f"users.{column} must be gone (D4/D7)"
        )


def test_users_carries_no_crm_product_columns() -> None:
    body = _table_body("users")
    for column in ("business_name", "phone", "primary_address", "sms_default_autonomous"):
        assert not re.search(rf"^\s*{column}\s", body, re.MULTILINE), column


def test_users_contract_constraints_are_not_carried_over() -> None:
    """crm's demo admission is not ported (D12), and its contract constraint
    hardcodes a demo email regex plus 24-hour expiry arithmetic."""
    body = _table_body("users")
    assert "users_subject_contract" not in body
    assert "users_lifecycle_contract" not in body
    assert "@no\\.email\\.add" not in SCHEMA
    assert "demo_start_at + INTERVAL '24 hours'" not in SCHEMA


# ---------------------------------------------------------------------------
# Authentication tables
# ---------------------------------------------------------------------------


def test_auth_tables_match_follow_up_contract() -> None:
    expected = {
        "magic_links": {
            "id", "user_id", "email", "token_hash", "expires_at", "used_at", "created_at",
        },
        "api_tokens": {
            "id", "user_id", "token_hash", "name", "created_at", "expires_at",
            "last_used_at", "revoked_at",
        },
    }
    for table, columns in expected.items():
        body = _table_body(table)
        for column in columns:
            assert re.search(rf"^\s*{column}\s", body, re.MULTILINE), f"{table}.{column}"
    assert "idx_api_tokens_hash_active" in SCHEMA
    assert "idx_api_tokens_user_active" in SCHEMA
    assert "idx_api_tokens_user_name_active" in SCHEMA
    assert SCHEMA.count("WHERE revoked_at IS NULL") >= 3


# ---------------------------------------------------------------------------
# model_configs
# ---------------------------------------------------------------------------


ROUTES = {"primary", "fast", "batch", "assessment", "other"}

# Every name deploy/postgresql.sh writes to secret/mira/api_keys. A route may
# not invent a new secret name.
VAULT_KEY_NAMES = {
    "provider_key",
    "subcortical_key",
    "anthropic_key",
    "anthropic_batch_key",
}


def test_model_configs_are_exact_and_fixed() -> None:
    body = _table_body("model_configs")
    assert "name IN ('primary', 'fast', 'batch', 'assessment', 'other')" in body
    assert "dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq')" in body
    assert "effort IN ('none', 'low', 'medium', 'high', 'xhigh', 'max')" in body
    assert "max_tokens INTEGER NOT NULL CHECK (max_tokens > 0)" in body
    assert "name TEXT PRIMARY KEY" in body


def test_model_configs_seed_covers_all_five_routes() -> None:
    rows = _seed_rows()
    assert set(rows) == ROUTES, sorted(set(rows) ^ ROUTES)
    assert {r["dialect_name"] for r in rows.values()} <= {
        "anthropic", "openai", "openrouter", "groq",
    }


def test_model_configs_seed_uses_only_existing_vault_key_names() -> None:
    rows = _seed_rows()
    assert rows, "model_configs seed rows could not be parsed"
    for name, row in rows.items():
        assert row["api_key_name"] in VAULT_KEY_NAMES, (
            f"route {name} invents secret name {row['api_key_name']}"
        )


def test_other_route_is_a_different_model_from_primary() -> None:
    """D14: `other` exists to consult an outside model, so it must not be a
    synonym for the chat model."""
    rows = _seed_rows()
    assert rows["other"]["model"] != rows["primary"]["model"]
    assert rows["other"]["model"] != rows["fast"]["model"]


def test_difficult_route_renamed_to_other_everywhere() -> None:
    assert "'difficult'" not in SCHEMA
    assert "difficult" not in SCHEMA


def test_model_configs_comment_matches_the_five_route_reality() -> None:
    comment = re.search(
        r"COMMENT ON TABLE model_configs IS '([^']*)';", SCHEMA
    )
    assert comment, "model_configs has no table comment"
    text = comment.group(1)
    assert "three" not in text.lower()
    for route in sorted(ROUTES):
        assert route in text, f"comment omits route {route}"


# ---------------------------------------------------------------------------
# usage_pricing (D5)
# ---------------------------------------------------------------------------


def test_usage_pricing_is_keyed_by_route_name() -> None:
    body = _table_body("usage_pricing")
    assert "name VARCHAR(50) PRIMARY KEY" in body
    assert "input_price_per_mtok" in body
    # The __default__ fallback row survives the migration deletion.
    assert "__default__" in SCHEMA
    for route in sorted(ROUTES):
        assert re.search(rf"\('{re.escape(route)}'\)", SCHEMA), f"no pricing row for {route}"
    # The 1.x tier-qualified keys are gone with conversation_llm/internal_llm.
    assert ":cof" not in SCHEMA
    assert ":free" not in SCHEMA


# ---------------------------------------------------------------------------
# Memory and federation
# ---------------------------------------------------------------------------


def test_embedding_dimensions_and_segment_column_are_direct() -> None:
    assert "segment_embedding vector(768)" in _table_body("messages")
    assert "embedding vector(768)" in _table_body("memories")
    assert "embedding vector(768)" in _table_body("global_memories")
    assert "embedding vector(300)" in _table_body("entities")


def test_global_memories_is_reachable_only_through_the_runtime_view() -> None:
    assert (
        "CREATE FUNCTION can_read_global_memories()\nRETURNS BOOLEAN\nSTABLE\n"
        "SECURITY DEFINER\nSET search_path = pg_catalog, public\nLANGUAGE sql" in SCHEMA
    )
    assert "CREATE VIEW global_memories_runtime\nWITH (security_barrier = true)" in SCHEMA
    assert "GRANT SELECT ON global_memories_runtime TO mira_dbuser;" in SCHEMA
    assert "GRANT EXECUTE ON FUNCTION can_read_global_memories() TO mira_dbuser;" in SCHEMA
    assert "REVOKE EXECUTE ON FUNCTION can_read_global_memories() FROM PUBLIC;" in SCHEMA
    # 1.x gave the runtime role full DML on the shared table, which at N>1 is a
    # cross-user prompt-injection vector. The view only pays off if revoked.
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON global_memories TO mira_dbuser;" not in SCHEMA
    assert "GRANT SELECT ON global_memories TO mira_dbuser;" not in SCHEMA
    assert "REVOKE ALL ON global_memories FROM mira_dbuser;" in SCHEMA


def test_global_usernames_has_no_rls_and_no_member_trigger() -> None:
    """The federation resolver queries this table with no user context; a policy
    or trigger requiring matching context breaks mira_resolve_username."""
    assert "ALTER TABLE global_usernames ENABLE ROW LEVEL SECURITY;" not in SCHEMA
    assert not _policies_for("global_usernames")
    assert "enforce_member_global_username" not in SCHEMA


def _policies_for(table: str) -> str:
    return "\n".join(
        line for line in SCHEMA.split("\n")
        if re.search(rf"CREATE POLICY\s+\w+\s+ON\s+{re.escape(table)}\b", line)
    )


# ---------------------------------------------------------------------------
# Persona (WP5 DDL)
# ---------------------------------------------------------------------------


def test_persona_contracts_exist() -> None:
    revisions = _table_body("persona_revisions")
    assert "revision_number INTEGER NOT NULL CHECK (revision_number > 0)" in revisions
    assert "source TEXT NOT NULL CHECK (source IN ('baseline', 'automatic', 'user', 'rollback'))" in revisions
    assert "UNIQUE (user_id, revision_number)" in revisions
    assert "UNIQUE (id, user_id)" in revisions
    state = _table_body("persona_state")
    assert "current_revision_id UUID NOT NULL" in state
    assert "refinement_checkpoint_activity_day INTEGER NOT NULL DEFAULT 0" in state

    assert "users_provision_baseline_persona" in SCHEMA
    assert "AFTER INSERT ON users" in SCHEMA
    assert "VALUES (NEW.id, 1, '', 'baseline')" in SCHEMA
    assert "INSERT INTO persona_state (user_id, current_revision_id)" in SCHEMA

    # Append-only: no UPDATE/DELETE on the revision history.
    assert "GRANT SELECT, INSERT ON persona_revisions TO mira_dbuser;" in SCHEMA
    assert not re.search(r"GRANT[^;]*(UPDATE|DELETE)[^;]*persona_revisions", SCHEMA)


# ---------------------------------------------------------------------------
# Row level security
# ---------------------------------------------------------------------------

# Every per-user table must be RLS-covered, including the OSS add-backs.
RLS_COVERED_TABLES = {
    "users",
    "magic_links",
    "api_tokens",
    "user_activity_days",
    "continuums",
    "messages",
    "memories",
    "entities",
    "persona_revisions",
    "persona_state",
    "persona_signals",
    "feedback_signals",
    "feedback_synthesis_tracking",
    "domain_knowledge_blocks",
    "domain_knowledge_block_content",
    "domaindoc_shares",
    "user_feedback",
}

# Global lookup tables (no owner column) plus the two tables deliberately left
# open: global_memories (gated by global_memories_runtime) and global_usernames
# (federation lookups run contextlessly).
RLS_EXEMPT_TABLES = {
    "model_configs",
    "usage_pricing",
    "global_memories",
    "global_usernames",
}


def test_rls_covers_every_per_user_table() -> None:
    enabled = set(re.findall(r"ALTER TABLE\s+([a-z_]+)\s+ENABLE ROW LEVEL SECURITY;", SCHEMA))
    assert enabled == RLS_COVERED_TABLES, sorted(enabled ^ RLS_COVERED_TABLES)
    for table in sorted(RLS_COVERED_TABLES):
        assert _defines_table(table), f"{table} has a policy but no table"
        assert _policies_for(table), f"{table} has RLS enabled but no policy"


def test_rls_exempt_tables_are_not_policies() -> None:
    enabled = set(re.findall(r"ALTER TABLE\s+([a-z_]+)\s+ENABLE ROW LEVEL SECURITY;", SCHEMA))
    assert not (enabled & RLS_EXEMPT_TABLES), sorted(enabled & RLS_EXEMPT_TABLES)


def test_rls_never_uses_the_throwing_context_predicate() -> None:
    """`current_setting('app.current_user_id')::uuid` raises on an unset GUC.
    Every policy must use the NULLIF form, which fails closed to zero rows."""
    assert "NULLIF(current_setting('app.current_user_id', true), '')::uuid" in SCHEMA
    assert "current_setting('app.current_user_id')::uuid" not in SCHEMA
    assert SCHEMA.count(
        "NULLIF(current_setting('app.current_user_id', true), '')::uuid"
    ) >= 2 * len(RLS_COVERED_TABLES)
    # Policies are scoped to the runtime role; mira_admin keeps BYPASSRLS.
    assert "FOR ALL TO mira_dbuser" in SCHEMA
    assert "TO PUBLIC" not in SCHEMA


def test_admin_and_runtime_grants() -> None:
    assert "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mira_admin;" in SCHEMA
    assert "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mira_admin;" in SCHEMA
    assert "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mira_admin;" in SCHEMA
    # Runtime role may not write the routing or pricing tables.
    assert "GRANT SELECT ON model_configs TO mira_dbuser;" in SCHEMA
    assert not re.search(r"GRANT[^;]*(INSERT|UPDATE|DELETE)[^;]*model_configs", SCHEMA)


def test_no_grant_or_policy_names_an_unknown_role() -> None:
    roles = set()
    for m in re.finditer(r"(?:GRANT|REVOKE)[^;]*?\b(?:TO|FROM)\s+([A-Za-z_][A-Za-z0-9_, ]*?);", SCHEMA):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                roles.add(part.split()[0])
    assert roles <= {"mira_admin", "mira_dbuser", "PUBLIC"}, sorted(
        roles - {"mira_admin", "mira_dbuser", "PUBLIC"}
    )


# ---------------------------------------------------------------------------
# Billing / CRM / member-gating excision (D7, D12, 8.1, 8.2)
# ---------------------------------------------------------------------------


def test_billing_and_member_functions_are_absent() -> None:
    for obj in (
        "add_calendar_month",
        "provision_member_entitlement",
        "users_provision_member_entitlement",
        "is_active_member",
        "resolve_active_member_by_email",
        "active_member_identity",
    ):
        assert obj not in SCHEMA, f"{obj} is billing/CRM gating machinery"


def test_crm_and_billing_sql_does_not_reappear() -> None:
    assert "stripe" not in SCHEMA.lower()
    assert "entitlement" not in SCHEMA.lower()
    assert "sms_" not in SCHEMA.lower()
    assert "workphone" not in SCHEMA.lower()
    assert "crm" not in SCHEMA.lower()
    assert "audit_event" not in SCHEMA.lower()
    assert "push_subscription" not in SCHEMA.lower()


# ---------------------------------------------------------------------------
# Scrub gate (plan 11)
# ---------------------------------------------------------------------------

# Route endpoints must point at a public provider. An allowlist is used rather
# than a list of known-private values: naming the private host, model or Vault
# key in this file would itself publish what the gate exists to keep out, and
# the allowlist rejects any unlisted value, including ones nobody has thought to
# write down yet.
PUBLIC_ROUTE_HOSTS = {"openrouter.ai", "api.anthropic.com", "api.groq.com"}


def test_schema_has_no_private_infrastructure() -> None:
    rows = _seed_rows()
    assert rows, "model_configs seed rows could not be parsed"
    for name, row in rows.items():
        host = urlparse(row["endpoint_url"]).hostname
        assert host in PUBLIC_ROUTE_HOSTS, (
            f"route {name} points at an unrecognised host: {host}"
        )

    # Cloud credentials only, over TLS. Local endpoints belong to the offline
    # path in deploy/postgresql.sh, never to committed DDL.
    assert "http://" not in SCHEMA

    # Private address space and personal install paths.
    assert re.search(r"\b(?:192\.168\.|10\.\d+\.\d+\.|172\.(?:1[6-9]|2\d|3[01])\.)", SCHEMA) is None
    for needle in ("/Users/", "/home/", "/opt/", ".gguf"):
        assert needle not in SCHEMA, f"private path escaped into the schema: {needle}"


def test_route_seeds_are_public_catalog_ids() -> None:
    """Seeded models are public provider catalog IDs: lowercase, no spaces, no
    quantiser suffixes from a personal fine-tune."""
    for name, row in _seed_rows().items():
        model = row["model"]
        assert re.fullmatch(r"[a-z0-9][a-z0-9._/+-]*", model), (
            f"route {name} has a non-catalog model id: {model!r}"
        )


# ---------------------------------------------------------------------------
# No migration path
# ---------------------------------------------------------------------------


def test_migrations_directory_is_gone() -> None:
    assert not (ROOT / "deploy" / "migrations").exists()
