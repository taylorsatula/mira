# Manual Installation

The preferred installation path is still `deploy/deploy.sh`. Use this document when the installer rejects the platform or when you need to reproduce its steps manually.

## Supported Automation

The installer supports:

- macOS with Homebrew
- Debian/Ubuntu-family Linux with `apt`
- Fedora/RHEL/CentOS/Rocky/Alma-family Linux with `dnf`

Other platforms need equivalent services and packages installed manually.

## Required Services

Install and start:

- PostgreSQL 17 with the `pgvector` extension
- Valkey
- HashiCorp Vault
- Python 3.12

The default local ports are:

- MIRA HTTP: `1993`
- Vault: `8200`
- Valkey: `6379`
- PostgreSQL: `5432`

## Python Environment

From the repository root:

```bash
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

For the local embedding model (the default), install PyTorch's CPU wheel and
sentence-transformers. Skip this for a remote embedding endpoint:

```bash
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install sentence-transformers
```

For web rendering support, install the Playwright package and browser. Neither
is in `requirements.txt` — both are optional, and `web_tool` reports
`playwright_unavailable` / `chromium_unavailable` without them:

```bash
venv/bin/pip install playwright
venv/bin/playwright install chromium
```

DOCX/XLSX uploads extract through stdlib `zipfile` + `ElementTree` by default.
Install these only if you want the library extractors instead:

```bash
venv/bin/pip install python-docx openpyxl
```

## Database

Create the `mira_service` database and load the current schema. The schema
sizes its vector columns from the embedding model, so resolve that first with
`deploy/lib/embedding_config.sh` (local model shown; for a remote
OpenAI-compatible endpoint pass `remote <endpoint_url> <model> <token>` and
store the token in Vault as `secret/mira/api_keys` `embeddings_key`):

```bash
createdb mira_service
source deploy/lib/embedding_config.sh
resolve_embedding_schema_args venv/bin/python "$PWD" local "" "" ""
psql -d postgres -v ON_ERROR_STOP=1 -c "DO \$roles\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_admin') THEN CREATE ROLE mira_admin LOGIN PASSWORD 'changethisifdeployingpwd' BYPASSRLS; END IF; IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN CREATE ROLE mira_dbuser LOGIN PASSWORD 'changethisifdeployingpwd'; END IF; END \$roles\$;"
psql -U postgres -h localhost -d mira_service -v ON_ERROR_STOP=1 "${EMBEDDING_SCHEMA_ARGS[@]}" -f deploy/mira_service_schema.sql
```

The `DO` block provisions the `mira_admin` and `mira_dbuser` roles the schema
grants to (mirroring Step 13 of `deploy/postgresql.sh`): it is idempotent, gives
both roles a LOGIN password (TCP connections use scram-sha-256, so a role with
no password can never authenticate), and `mira_admin` gets `BYPASSRLS`.
`ON_ERROR_STOP` makes `psql` exit nonzero on the first failing statement —
without it, a failed `GRANT` or `CREATE POLICY` is reported as success.

Vault stores service credentials and provider keys. The deploy scripts in `deploy/vault.sh` and `deploy/postgresql.sh` are the source of truth for the exact key names used by the automated path.

## Injection Screen (optional)

MIRA screens external content (fetched pages, email, uploaded files) through a
System One decision model before it enters any model context. On a manual
install the screen is enabled by app-config default, so either give it a
reachable model or turn it off — otherwise the first screened content fails
closed.

Enable it with a reachable System One endpoint (hosted gateway or self-hosted
Kev):

```bash
export MIRA_INJECTION_SCREEN_ENABLED=1
export MIRA_SYSTEMONE_PROVIDER=remote        # or local for a self-hosted, unkeyed endpoint
export MIRA_SYSTEMONE_ENDPOINT=https://gw.lunaroute.com/v1/systemone
export MIRA_SYSTEMONE_MODEL=djev
# remote only: store the bearer token in Vault (no env var for secrets)
vault kv put secret/mira/api_keys systemone_key="your-token"
```

Or disable it (external content is still wrapped, never passed raw):

```bash
export MIRA_INJECTION_SCREEN_ENABLED=0
```

The bare-metal installer collects these through its interview /
`deploy-config.example.yml` (`injection_screen`, `systemone_provider`,
`systemone_endpoint`, `systemone_model`, `systemone_api_key`) and writes them
to `/opt/mira/systemone.env`; the Docker image ships with the screen off
because the container cannot provision the Vault key — pass the env vars and
seed Vault manually to opt in.

## Running

Once services, credentials, schema, and Python dependencies are in place:

```bash
venv/bin/python main.py
```

There is no in-place upgrade path: 2.0 installs the greenfield schema into an empty database. To salvage data from an older install, take a `pg_dump` first and restore it manually.
