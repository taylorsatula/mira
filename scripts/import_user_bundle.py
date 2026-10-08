"""MIRA user bundle importer (fresh-install side). Spec, not yet implementation.

Counterpart to export_user_bundle.py. Reads the zip produced by the hosted
exporter and restores the person into THIS install:

  1. Verify manifest.json sha256 checksums for every file. Stop on any mismatch.
  2. Ask the local user which local account receives the data (never guess).
  3. Postgres side, in order: user_profile, continuums, messages, entities,
     memories (keep original row UUIDs so entity_links and memory cross-links
     resolve; remap only user_id), feedback_signals, user_activity_days.
     Embeddings/search_vectors were excluded at export; regenerate all with
     THIS install's embedding model after rows land.
  4. Sqlite side, per table, in dependency order (domaindocs before
     domaindoc_sections before domaindoc_versions; contacts before reminders
     where reminders carry contact_uuid): insert rows, re-encrypting
     encrypted__ columns with THIS install's per-user key via the app's own
     UserDataManager (derive_session_key), never a hand-rolled cipher.
  5. Preserve timestamps. Bump sqlite_sequence to max(id) per table so
     autoincrement does not collide.
  6. Re-verify: every manifest row count present. Report a diff if not.
  7. On the person's next conversation, acknowledge the hosting gap rather
     than performing unbroken continuity (per the bundle's AGENTS.md).

Run inside the fresh install's environment (PYTHONPATH at app root, its venv)
so the app's own clients/managers do the crypto and DB writes.

Usage (once implemented):
  python3 import_user_bundle.py <bundle.zip> [--account <email>]
"""
print(__doc__)
