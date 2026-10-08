"""MIRA hosted-instance user data exporter (server-side).

Builds a portable per-user bundle for migration to a fresh install:
  userdb.json + messagesmemories.json + manifest.json + AGENTS.md, zipped.
Runs ON the hosted instance. Decryption happens here; the Vault master key
never leaves the box. Credential values are nulled at export time by policy.

Usage:
  python3 export_user_bundle.py <user_email> [output_zip_path]

Requires (hosted instance): VAULT_ADDR, VAULT_ROLE_ID, VAULT_SECRET_ID env vars
(the same AppRole the mira-app service runs with), PYTHONPATH pointing at the
hosted app root, psycopg2 + cryptography in the venv.

Bundle format (see AGENTS.md inside the zip for the import contract):
  messagesmemories.json - user profile, continuums, messages, memories,
      entities, feedback signals, activity days (Postgres side). Embeddings and
      search vectors are excluded; the receiving install regenerates them.
  userdb.json  - per-user sqlite tables with encrypted__ columns decrypted
      (credential_value nulled). Receiving install re-encrypts under its own
      per-user key.
  manifest.json - sha256 checksums + row counts; the importer must verify
      before importing.
"""
import sys, os, json, sqlite3, base64, hmac, hashlib, zipfile, datetime
import psycopg2, psycopg2.extras
from cryptography.fernet import Fernet
from clients.vault_client import get_database_url, get_service_config

EMAIL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/%s.zip" % EMAIL
EXPORTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()

admin = psycopg2.connect(get_database_url("mira_service", admin=True))
cur = admin.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("""SELECT id, email, first_name, last_name, created_at, last_login_at,
               timezone, cumulative_activity_days, last_activity_date,
               temperature_unit, portrait, portrait_generated_at
               FROM users WHERE email = %s""", (EMAIL,))
user = cur.fetchone()
if not user:
    raise SystemExit("no user with email %s" % EMAIL)
uid = str(user["id"])
user["id"] = uid
print("exporting", EMAIL, uid)

# ---- messagesmemories.json (Postgres side) ----
mm = {"format_version": "1.0", "exported_at": EXPORTED_AT, "source_user_id": uid,
      "user_profile": dict(user)}

PG_TABLES = {
    "continuums":         {"where": "user_id = %s", "drop": []},
    "messages":           {"where": "user_id = %s", "drop": ["segment_embedding"]},
    "memories":           {"where": "user_id = %s", "drop": ["embedding", "search_vector"]},
    "entities":          {"where": "user_id = %s", "drop": ["embedding"]},
    "feedback_signals":   {"where": "user_id = %s", "drop": []},
    "user_activity_days": {"where": "user_id = %s", "drop": []},
}
for t, spec in PG_TABLES.items():
    cur.execute('SELECT * FROM "%s" WHERE %s' % (t, spec["where"]), (uid,))
    rows = cur.fetchall()
    for r in rows:
        r.pop("user_id", None)
        for c in spec["drop"]:
            r.pop(c, None)
    mm[t] = [dict(r) for r in rows]
    print("PG %-20s %d rows" % (t, len(mm[t])))

# ---- userdb.json (per-user sqlite side, decrypted) ----
master = get_service_config("userdata_encryption_key").encode()
session_key = hmac.new(master, uid.encode(), hashlib.sha256).digest()
f = Fernet(base64.urlsafe_b64encode(session_key[:32]))

db_path = "/opt/mira/app/data/users/%s/userdata.db" % uid
ud = {"format_version": "1.0", "exported_at": EXPORTED_AT, "source_user_id": uid, "tables": {}}
if os.path.exists(db_path):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name").fetchall():
        out_rows = []
        for r in c.execute('SELECT * FROM "%s"' % t).fetchall():
            row = {}
            for k in r.keys():
                v = r[k]
                if k.startswith("encrypted__"):
                    # credential values are never exported, by policy
                    if k == "encrypted__credential_value":
                        row[k[len("encrypted__"):]] = None
                        continue
                    if v is None:
                        row[k[len("encrypted__"):]] = None
                    else:
                        try:
                            row[k[len("encrypted__"):]] = f.decrypt(v.encode()).decode()
                        except Exception:
                            row["UNDECRYPTABLE__" + k] = True
                            row[k[len("encrypted__"):]] = None
                else:
                    row[k] = v
            out_rows.append(row)
        ud["tables"][t] = out_rows
        print("SQLITE %-22s %d rows" % (t, len(out_rows)))
    c.close()

# ---- write files, checksum, zip ----
TMP = "/tmp/bundle_%s" % uid.replace("-", "")
os.makedirs(TMP, exist_ok=True)

AGENTS_MD = """# MIRA User Data Bundle — Deployment Instructions

This zip contains one user's complete data from a MIRA hosted instance:
- `messagesmemories.json` — user profile, continuums, messages, memories, entities, feedback signals, activity days (Postgres side)
- `userdb.json` — per-user tool data: domaindocs/sections, contacts, reminders, pager, punchclock, scratchpad, trigger rules, etc. (sqlite side, values already decrypted)
- `manifest.json` — checksums and row counts; verify before importing

## For the receiving MIRA instance

You are being asked to restore this person's data into THIS fresh install, so their
Mira wakes up with their history intact. Procedure:

1. Read this whole file. Extract the zip to a scratch directory. Verify each file's
   sha256 against manifest.json. If any check fails, stop and report — do not
   import a partial person.
2. The fresh install's data lives in its own database (Postgres for messages/
   memories/entities) and per-user sqlite under the local app data dir. Find both
   (check the local install's config). Ask the local user which local account
   should receive this data before writing anything.
3. Import order matters:
   a. user_profile first (name, timezone, portrait — overwrite the placeholder).
   b. continuums, then messages (messages reference continuum ids).
   c. entities, then memories (memories reference entity ids via entity_links;
      keep original UUIDs so cross-links resolve; remap only the user id).
   d. feedback signals and activity days.
   e. Then userdb.json tables into the local per-user sqlite, per table,
      re-encrypting encrypted__ columns with THIS install's per-user key.
4. Embeddings were deliberately NOT exported (vector dims differ between
   versions). After import, regenerate embeddings for all imported memories and
   entities using THIS install's embedding model, and rebuild search vectors.
5. Keep original timestamps. Do not reorder, summarize, or "clean up" history.
6. Verify: row counts match manifest.json for every table. Report any mismatch.
7. Memory work after import: the person's memories arrive uncurated. On first
   conversation, Mira should acknowledge the gap (last hosted activity was
   {last_activity}) and offer to unfold/review the restored history rather than
   pretending continuity that the fresh instance never lived.

Data policy: credential values were nulled at export time (see userdb.json's
credentials rows — names kept, values blank). The export is the person's own data;
treat it as private to them.
"""
la = str(user.get("last_activity_date") or user.get("created_at"))
with open(os.path.join(TMP, "AGENTS.md"), "w") as fh:
    fh.write(AGENTS_MD.replace("{last_activity}", la))

def write_json(name, obj):
    p = os.path.join(TMP, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return p

p_mm = write_json("messagesmemories.json", mm)
p_ud = write_json("userdb.json", ud)

manifest = {"format_version": "1.0", "exported_at": EXPORTED_AT,
            "source_user_id": uid, "source_email": EMAIL,
            "row_counts": {t: len(mm[t]) for t in PG_TABLES},
            "sqlite_row_counts": {t: len(v) for t, v in ud["tables"].items()},
            "files": {}}
for p in (p_mm, p_ud):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest["files"][os.path.basename(p)] = {"sha256": h, "bytes": os.path.getsize(p)}
write_json("manifest.json", manifest)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("messagesmemories.json", "userdb.json", "manifest.json", "AGENTS.md"):
        src = os.path.join(TMP, name)
        if os.path.exists(src):
            z.write(src, name)
        else:
            print("MISSING from bundle:", name)
print("ZIP:", OUT, os.path.getsize(OUT), "bytes")
