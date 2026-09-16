# tests/tmp/ — display exhibit for the disposable-probe pattern, not a test home

## Rules

- Disposable checks and probes are written directly in `/tmp` or as inline Python in the conversation, as a matter of courtesy — scratch work does not belong in the repository tree.
- Mocks are never used, without exception: no mock objects, no stubs, no fakes, no simulated infrastructure. Verification is live or it doesn't count.
- Any test file added to this folder will be **autodeleted instantly** on sight. This folder is not a test suite, is not run by anything, and will never contain more than the exhibit below.
- The folder exists only to display one exemplary disposable probe, so a session can see the shape: live infrastructure, real credentials plumbing, no mocks, executed once, then kept as an example rather than as a check.
- Verification in MIRA is live (see the NO MOCKS section of the root `AGENTS.md`). A one-off probe that earns permanence leaves this folder for one of two homes: a production path-probe registered alongside the POST gate when it needs live infrastructure, or `tests/protected/` when it runs offline and the user has authorized it. Never a file here — admission to `tests/protected/` is owned by `tests/protected/AGENTS.md`.

## Files

- `jsonb_roundtrip_probe.py` — The one display exhibit: an exemplary disposable probe that round-tripped real JSONB memory columns (`inbound_links`/`outbound_links`/`entity_links`/`annotations`) through `lt_memory/db_access.py:update_memory` against live Postgres with production Vault credentials. Executed once, never run again. Verified current against the source: `update_memory` still takes the `Dict[str, Any]` updates shape and still wraps JSONB fields via `_JSONB_MEMORY_FIELDS` (`Jsonb(value) or cannot adapt type 'dict'`), and the probe's other call surfaces (`get_memory`, `get_memories_paginated` returning a `MemoryPageResult` TypedDict) match. Do not rewrite, run, or extend it; see the `lt_memory/AGENTS.md` note on the `Jsonb()` wrapping contract.