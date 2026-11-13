# global-memory-store

Standalone shared memory store for MIRA instances. One Postgres database, one
HTTP API, no dependency on any MIRA checkout. See `docs/GLOBAL_MEMORY_STORE.md`
in the MIRA repository for the operator guide.

## Install

```sh
python3 -m venv venv
./venv/bin/pip install .
# or, for the local (sentence-transformers) embedding provider:
./venv/bin/pip install '.[local-embeddings]'
```

## Use

```sh
./venv/bin/python -m global_memory_store init --config /path/to/store.toml
./venv/bin/python -m global_memory_store serve --config /path/to/store.toml
```

`init` creates the tables and records the embedding identity. `serve` verifies
both and refuses to start on a mismatch. Both read one TOML file; every field
is required and named in the error when missing.
