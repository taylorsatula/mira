# docs/ — operator-facing documentation

## Rules

- A path printed in a doc here must exist in the repo and describe the current workflow; keep deployment docs aligned with the `deploy/` scripts they cite. Verified examples: `deploy/deploy.sh`, `deploy/vault.sh`, `deploy/postgresql.sh`, `deploy/mira_service_schema.sql` (cited by MANUAL_INSTALL.md) and the `3090`/`3092` endpoint contract (cited by OFFLINE_MODELS.md, owned by `deploy/config.sh:CONFIG_LLAMA_MAIN_MODEL`/`CONFIG_LLAMA_SMALL_MODEL` and `deploy/postgresql.sh:LLAMA_MAIN_URL`/`LLAMA_SMALL_URL`).
- `MANUAL_INSTALL.md` is stale at its last line: it says to run `deploy/deploy.sh --migrate`, but no script in `deploy/` accepts a `--migrate` flag (grep-verified against all `deploy/` shell scripts and `deploy/lib/`). Do not build on that command; if an upgrade path is needed, derive it from `deploy/deploy_database.sh` or fix the doc.
- These files are read when the automated installer cannot continue or when an operator must prepare external resources; keep them as operational checklists, not design essays.
- `docs/` documents the host-metal install and offline model flow only. Docker bootstrap is owned by `deploy/docker/scripts/AGENTS.md`; that map's claim that `MANUAL_INSTALL.md` and `OFFLINE_MODELS.md` reference no `deploy/docker/scripts/` files holds (grep-verified).

## Files

- `MANUAL_INSTALL.md` — Manual setup checklist for platforms the installer rejects: required services (PostgreSQL 17 + pgvector, Valkey, Vault, Python 3.12), default ports (1993/8200/6379/5432), venv + spaCy + Playwright setup, schema load, run command. Mostly current; the `--migrate` tail is dead (see Rules).
- `OFFLINE_MODELS.md` — Offline/local LLM preparation: endpoints (`localhost:3090` main, `localhost:3092` small), model storage (`/opt/mira/models/`), llama-server log paths, startup checklist. Current — endpoint/port/log claims match `deploy/config.sh`, `deploy/postgresql.sh`, `deploy/finalize.sh`.
- `SEGMENT_SYSTEM.md` — Segment lifecycle overview: active segment, live context compaction vs segment collapse (`SegmentCollapseHandler`, sentinel messages, `messages.segment_embedding` 768-dim mdbr-leaf-ir-asym). Current; matches `cns/services/segment_collapse_handler.py` and `deploy/mira_service_schema.sql`. Conceptual overview, not an operator checklist.
