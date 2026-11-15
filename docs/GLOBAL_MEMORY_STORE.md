# Global Memory Store

The global memory store is a small standalone service that holds memories shared across every MIRA instance that points at it. Each instance keeps a local copy of the store's contents in its own `global_memories` table, refreshes that copy at boot, and reads it during normal memory retrieval. Turn-time retrieval never contacts the store; the network is touched at boot and at the moment a new global memory is written.

A memory becomes global when the extraction model marks it as one: a stable, plainly factual attribute of a named person, place, or shared thing (a birthday, an anniversary, a person's role). Everything about how a speaker thinks, feels, or works stays personal to that speaker's instance.

## What you need

- A host for the store. A Mac Mini or a Raspberry Pi is enough. The store is a single Python process with its own Postgres database; it does not need MIRA, Vault, or Valkey.
- PostgreSQL 14 or newer with the `vector` extension (pgvector) available.
- Python 3.11 or newer.
- An embeddings provider. Two options:
  - **Remote** (recommended for small hosts): any OpenAI-compatible `/v1/embeddings` endpoint. The store makes one HTTP call per write and per boot probe; it installs no model runtime.
  - **Local**: `sentence-transformers` with `MongoDB/mdbr-leaf-ir-asym`, the same model the instance uses for its local provider. The store loads the model in-process; expect roughly a gigabyte of RAM for the model plus PyTorch. Use this only when the store must not call out to a remote endpoint.

The store's embedding configuration must match the instance's. The instance checks this at every boot and refuses to use a mismatched store (see "What the instance reports at boot" below). If the instance uses the remote `emb-qwen3` endpoint, configure the store with the same endpoint, model, and dimensions. If the instance uses the local provider, configure the store with `provider = "local"` and the same model.

## Install

```sh
git clone <this repo> mira && cd mira/global_memory_store
python3 -m venv venv
./venv/bin/pip install .
# or, for the local embedding provider:
./venv/bin/pip install '.[local-embeddings]'
```

Create a database and role for the store, and enable the extension as a superuser:

```sh
sudo -u postgres psql <<'SQL'
CREATE ROLE global_memory_store LOGIN PASSWORD 'pick-a-password';
CREATE DATABASE global_memory_store OWNER global_memory_store;
\c global_memory_store
CREATE EXTENSION vector;
SQL
```

## Configuration

One TOML file. Every field is required; the store refuses to start and names the missing field rather than guess.

```toml
bind_host = "0.0.0.0"
bind_port = 8471
database_url_file = "/etc/global-memory-store/database_url"
token_file = "/etc/global-memory-store/token"

[embedding]
provider = "remote"                      # or "local"
model = "emb-qwen3"                      # the model your instances use
endpoint_url = "https://gw.example.com/v1/embeddings"   # remote only; "" for local
api_key_file = "/etc/global-memory-store/embeddings_key" # remote only; "" for local
dimensions = 1024                        # must match the instance's embedding_config
```

`database_url_file` holds the Postgres connection string (`postgresql://global_memory_store:password@host/global_memory_store`). `token_file` holds the bearer token instances present. Both files should be mode 0600. The store reads no environment variables and no other files.

Generate a token with `openssl rand -hex 32`. The same token goes in the instance's Vault as `global_memory_store` (see below).

## Initialize and run

```sh
./venv/bin/python -m global_memory_store init --config /etc/global-memory-store/store.toml
./venv/bin/python -m global_memory_store serve --config /etc/global-memory-store/store.toml
```

`init` creates the two tables and records which embedding model and dimension count the database was built for. It is idempotent. `serve` verifies the tables exist and the recorded identity matches the config, then serves; it runs no DDL and refuses to start on a mismatch.

A sample systemd unit:

```ini
[Unit]
Description=Global memory store
After=postgresql.service
Requires=postgresql.service

[Service]
ExecStart=/opt/global-memory-store/venv/bin/python -m global_memory_store serve --config /etc/global-memory-store/store.toml
Restart=on-failure
User=global-memory-store

[Install]
WantedBy=multi-user.target
```

## Pointing an instance at the store

Two things on the instance side:

1. The store's base URL in `/opt/mira/systemone.env`: `MIRA_GLOBAL_MEMORY_STORE_URL=http://your-store-host:8471` (written by the installer when `MIRA_GLOBAL_MEMORY_STORE_URL` is set; otherwise add the line by hand and restart).
2. The bearer token in Vault at `secret/mira/api_keys`, field `global_memory_store`. The installer writes it when `MIRA_GLOBAL_MEMORY_STORE_TOKEN` is set; otherwise: `vault kv patch secret/mira/api_keys global_memory_store=<token>`.

Restart the instance. At boot it checks the store, pulls the full memory set into its local `global_memories` copy, and logs one of:

- `Global memory store active: N memories loaded into the local copy`: everything works.
- `Global memory store unavailable: ...`: the store is unreachable, the token is missing, or the embeddings do not match. The instance serves normally; shared memories are served from the last successful pull, and nothing new is shared until the next restart.
- Nothing at all: the store is not configured (`MIRA_GLOBAL_MEMORY_STORE_URL` unset). The instance behaves exactly as it did before the store existed.

The pre-server startup check reports the same thing as `global_memory_store` in the startup report. It is advisory: a failure there never blocks boot.

## Re-running the installer

The installer writes the store's token into Vault only when `MIRA_GLOBAL_MEMORY_STORE_TOKEN` is set in its environment, and it rewrites the whole `secret/mira/api_keys` secret each run. A re-run without the variable drops the `global_memory_store` field, and the instance boots with the store disabled (one advisory line in the startup report). Set the variable on every run, or restore the field afterwards with `vault kv patch secret/mira/api_keys global_memory_store=<token>`. The same is true of the other optional fields the installer writes (`systemone_key`, `kagi_api_key`, the embedding key).

## Deleting a memory

The store has no admin UI. Delete by id over HTTP:

```sh
curl -X DELETE -H "Authorization: Bearer $TOKEN" http://your-store-host:8471/v1/memories/<memory-id>
```

Instances pick up the deletion at their next restart. To find an id, list memories: `curl -H "Authorization: Bearer $TOKEN" http://your-store-host:8471/v1/memories`.

## Limits

- Another instance's new memories are visible only after this instance restarts.
- Two statements of one fact with different values (two different dates for one person's birthday) are close enough in vector space to count as duplicates. The first one written stays. Correction is by DELETE.
- Personal memories that existed before the store was configured are not copied into it.
- There is no per-user opt-out. Any user whose segments are extracted contributes.
- `contributor_user_id` is asserted by the instance under a shared token. It supports cleanup by an operator and proves nothing.
- Two people with the same name are not told apart.
- A user's MIRA cannot edit, link, merge, or archive a global memory.
- The boot pull holds the whole set in memory, and the store's duplicate check is a sequential scan. Revisit either when a measured boot time or write latency says so.
- The store's encoder is a second implementation of the instance's document-side encoder. Agreement is checked at each instance boot and nowhere else. A change to the instance's embedding model or normalization has to be made in the store as well.
- Global memory text reaches other users' prompts unscreened. Screening on write and wrapping on read are planned but not yet implemented.
