# global_memory_store/ — the standalone shared memory store

A self-contained Python package: one Postgres database, one HTTP API, no
import of anything in the MIRA tree. Instances pull the whole memory set at
boot into their local `global_memories` copy and write new global memories
here at extraction time. Turn-time retrieval never touches this process.

## Isolation rule

No file in this package imports from the MIRA tree. The audit, run from the
repo root:

```
rg '^\s*(from|import) (agents|auth|clients|cns|config|lt_memory|tools|ui|utils|working_memory)\b' global_memory_store/
```

It must print nothing. The package installs on a host with no MIRA checkout
(a Mac Mini, a Raspberry Pi); an import from the tree would make that
impossible and would couple the store's release to the instance's.

## Credential exception

The store reads its bearer token, its Postgres connection string, and its
embeddings API key from files named in its TOML config. It has no Vault.
This is a stated exception to the root credential rule, limited to this
package; the files are the operator's to protect (mode 0600 is the norm).

## Encoder coupling

`global_memory_store/embedding.py` is a second implementation of the
document-side encode in `clients/embeddings_provider.py` (the `_encode`
methods behind `encode_deep`). The two must produce the same vector for the
same text. The instance verifies agreement at every boot
(`lt_memory/global_store.py`); a change to either side — model, normalization,
input handling — must land in both, or every instance reports the store as
mismatched and runs without it.

## Files

- `src/global_memory_store/config.py` — the TOML config: every field required, no defaults, no environment variables. Open it before adding a field.
- `src/global_memory_store/embedding.py` — the document-side encoder, remote (OpenAI-compatible `/v1/embeddings`) and local (sentence-transformers `encode_document`).
- `src/global_memory_store/storage.py` — Postgres access: schema init, the identity check, the duplicate-checked insert, the paged list, delete.
- `src/global_memory_store/api.py` — the four routes and the bearer-token dependency.
- `src/global_memory_store/__main__.py` — `init` and `serve` subcommands.
- `pyproject.toml` — the package's own dependencies; `local-embeddings` extra carries sentence-transformers so a remote-provider host installs no model runtime. The src layout keeps the project root from shadowing the installed package under `python -m`.

## Wiring

- Instance side: `lt_memory/global_store.py` (boot check and pull), `clients/global_memory_store_client.py` (HTTP client), `main.py` lifespan (one step after the lt_memory factory). Counterpart: `lt_memory/AGENTS.md`.
- The store never calls the instance. All traffic is instance → store.
- `init` creates the tables and writes the `embedding_identity` row; `serve` verifies both and runs no DDL. A config whose model or dimensions differ from the identity row refuses to serve.
- The duplicate check is an exact nearest-neighbor query under an advisory lock; there is deliberately no vector index on `memories.embedding`.
