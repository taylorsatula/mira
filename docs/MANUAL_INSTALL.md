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
psql -U postgres -h localhost -d mira_service "${EMBEDDING_SCHEMA_ARGS[@]}" -f deploy/mira_service_schema.sql
```

Vault stores service credentials and provider keys. The deploy scripts in `deploy/vault.sh` and `deploy/postgresql.sh` are the source of truth for the exact key names used by the automated path.

## Running

Once services, credentials, schema, and Python dependencies are in place:

```bash
venv/bin/python main.py
```

There is no in-place upgrade path: 2.0 installs the greenfield schema into an empty database. To salvage data from an older install, take a `pg_dump` first and restore it manually.
