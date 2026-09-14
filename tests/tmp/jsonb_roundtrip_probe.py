"""
Exemplary disposable probe — kept for display only, per the policy in this
folder's AGENTS.md. It was executed ONCE in this exact form against live
infrastructure and is not part of any test suite; it will never run again.

What it verified (2026-09-14): the `Jsonb()` wrapping fix in
`lt_memory/db_access.py:update_memory`, after the autonomous memory curator
died to `psycopg.ProgrammingError: cannot adapt type 'dict'` on a real merge.
The probe round-trips a memory's real JSONB columns (inbound_links,
outbound_links, entity_links, annotations) through the app's own `LTMemoryDB`,
using the production Vault credentials plumbing, writing each value back
unchanged and asserting the read-back is identical. Live Postgres, real
memories with real links — no mocks, no fixtures, nothing simulated.

Why it was disposable: it answers one question about one fix. Once executed
and the result observed, re-running it adds no information — so it lives here
as an example of the shape, not as a check.

Provenance note: the original executed run identified candidates via a psql
pull passed as argv. This display version selects them through the app's own
sanctioned surface (`get_memories_paginated`) so the exhibit is self-contained;
this form was then executed once as displayed, with the same result.
"""
import os
import sys

os.environ.setdefault("VAULT_ADDR", "http://127.0.0.1:8200")
os.environ["VAULT_ROLE_ID"] = open("/opt/vault/role-id.txt").read().strip()
os.environ["VAULT_SECRET_ID"] = open("/opt/vault/secret-id.txt").read().strip()
sys.path.insert(0, "/opt/mira/app")
os.chdir("/opt/mira/app")

from utils.user_context import set_current_user_id

UID = "8e9c16ae-b033-4f40-a560-48c8c85840fc"  # the deployed install's local account
set_current_user_id(UID)

from utils.database_session_manager import get_shared_session_manager
from lt_memory.db_access import LTMemoryDB

db = LTMemoryDB(get_shared_session_manager())

JSONB_FIELDS = ("inbound_links", "outbound_links", "entity_links", "annotations")

# Candidates come through the sanctioned listing surface, bounded out of
# respect for the live system: three memories with non-empty JSONB fields
# prove the point. Empty fields were never the failure mode — the curator's
# crash needed lists of dicts for psycopg to choke on.
probed = 0
page = db.get_memories_paginated(limit=50, user_id=UID)
for row in page["memories"]:
    if probed >= 3:
        break
    mem = db.get_memory(row["id"], user_id=UID)  # psycopg already types uuid columns as UUID
    if not any(getattr(mem, field) for field in JSONB_FIELDS):
        continue
    before = tuple(getattr(mem, field) for field in JSONB_FIELDS)
    # Write every JSONB column back unchanged — pure path validation, no data change.
    updated = db.update_memory(
        mem.id, {field: getattr(mem, field) for field in JSONB_FIELDS}, user_id=UID
    )
    after = tuple(getattr(updated, field) for field in JSONB_FIELDS)
    assert before == after, f"round-trip mismatch on {mem.id}"
    print(f"OK {str(mem.id)[:8]}: ({len(mem.inbound_links)} in, {len(mem.outbound_links)} out, "
          f"{len(mem.entity_links)} ent, {len(mem.annotations)} ann) round-tripped intact")
    probed += 1

if probed == 0:
    print("no memories with non-empty JSONB fields were reachable")
    sys.exit(1)
print(f"PROBE PASSED: {probed} memories, update_memory accepts JSONB columns")
