# MIRA v2026.11.15-2.3.2

A security patch release. It bumps `pypdf` from 6.21.2 to 6.21.4, fixing
seven high-severity denial-of-service advisories, and carries everything in
the amended v2026.11.14-2.3.1 (the boot-gate dependency wait). No database
steps.

## pypdf 6.21.2 → 6.21.4

All seven open Dependabot alerts against this repository were the same
package. They are all denial-of-service class — unbounded memory use and
non-terminating loops when parsing cross-reference streams, page trees,
encrypted files, page labels, and dictionary cloning (CVE-2026-119203
through CVE-2026-119209). MIRA feeds user-uploaded PDFs to `pypdf` in the
chat request path (`utils/document_processing.py`), so a crafted PDF could
hang or exhaust the process; the 32 MB upload cap does not bound these
attacks. 6.21.4 is the patch release carrying all seven fixes. The usage
surface (`PdfReader` over `BytesIO`, `page.extract_text()`) is unchanged.

Because `requirements.txt` changed, `mira update` rebuilds the server
virtualenv on update. No configuration changes, no database steps, no
consent beyond the normal update prompt.

## Also in this release

The boot-gate dependency wait from the amended v2026.11.14-2.3.1: the
pre-server gate now waits for Postgres, Valkey and Vault to answer before
running its verification rounds (`MIRA_DEPENDENCY_WAIT_SECONDS`, default
180 seconds, `0` disables), so a slow-starting database no longer consumes
the round budget and parks a healthy-but-slow boot. If the wait expires,
the gate parks (or exits, per `MIRA_POST_GATE_FAILURE_ACTION`) without
running the rounds, so the billed provider probes never run against a boot
whose infrastructure never came up.

## For upgraders

`mira update`, as always, from v2026.10.16-2.1 or later. The server venv is
rebuilt because `requirements.txt` changed; there are no database steps.

## Verification

- Gate rows a–c verified at build time against the tagged asset:
  `build_release_artifact v2026.11.15-2.3.2` passed `scripts.migrate lint`
  (3 versioned and 1 repeatable step follow the rules), and
  `verify_release_identity` confirmed tag = v + VERSION, the asset's VERSION
  matches the repo, and the asset's RELEASE_COMMIT equals the tagged commit
  (79b4e38).
- Row c (convergence) ran on the build host against a local PostgreSQL 17
  superuser; `scripts.migrate generate` rewrote `deploy/migrations/BASELINE`
  and `HEAD_FINGERPRINT.json` byte-identically.
- Gate rows d (skip-chain) and e (restore round trip): not run. This
  release ships no database steps and no changes under `deploy/migrations/`
  (verified: no diff against v2026.11.14-2.3.1). The rows carry their
  v2026.10.21-2.1.2 results unchanged.
- `pypdf==6.21.4` verified by installing into a clean virtualenv and
  running the exact call pattern `document_processing` uses
  (`PdfReader(BytesIO(...))`, `page.extract_text()`), plus malformed-input
  checks (truncated and garbage PDFs raise promptly instead of hanging).
