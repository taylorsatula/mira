# MIRA v2026.10.21-2.1.2

This is a point release. It contains no new features and no database
steps — only a full refresh of the third-party dependency pins in
`requirements.txt`, cut as a precautionary response to the disclosure
wave of October 21, 2026.

**Why now.** Today's advisories were led by the Starlette "BadHost"
authentication-bypass (CVE-2026-48710, in the CISA KEV catalog), which
sits directly under FastAPI in MIRA's stack, alongside a path-traversal
RCE in `uv` and unauthenticated RCEs in PHP. None of MIRA's pinned
versions is affected by any advisory in the OSV.dev databases — every
pin was checked against OSV for its exact version and came back clean,
including starlette 1.9.0 (BadHost was fixed in 1.0.1). The release
exists so a `mira update` moves every install to the newest, verified-
clean dependency set in one hop, rather than leaving installs to drift
on whatever each deployment originally resolved.

**What changed under the hood.** All 50 requirements are now exact pins
to each package's newest release (previously most were floating with a
few floor constraints). The three contracts that justified floors were
re-verified against the new majors before pinning: the Anthropic
dialect's `output_config.effort` and `Timeout` usage holds on
anthropic 2.1.0, the MCP check-in endpoint's
`MCPServer.streamable_http_app(stateless_http=True)` interface holds
on mcp 2.3.0, and httpx 0.28.1 still ships against httpcore 1.x (the
IP-pinning transport requirement). The full set resolves cleanly and
every pin supports Python 3.12, the deployment floor.

**For upgraders.** `mira update`, as always. No database steps run;
nothing to consent to. The updater rebuilds the venv from the new pins.
Installs using the local embedding provider
(`embedding_provider: local`) re-pull torch and sentence-transformers
unchanged — those stay outside `requirements.txt`.

## Verification

- Gate rows a–c verified at build time against the tagged asset:
  `scripts.migrate lint` passed, and `verify_release_identity` confirmed
  tag = v + VERSION, the asset's VERSION matches the repo, and the asset's
  RELEASE_COMMIT equals the tagged commit (9d24c90).
- Gate rows d (skip-chain) and e (restore round trip): not re-run at cut
  time — this release ships no database steps and changes no migration or
  updater-flow code (`git diff v2026.10.20-2.1.1 v2026.10.21-2.1.2 --
  deploy/migrations/ utils/schema_migrations.py deploy/update.sh` is
  empty), so the two rows carry their v2026.10.20-2.1.1 results unchanged.
- The full pinned requirement set installed from scratch into a clean venv
  on the build host at cut time — the pins are installable, not merely
  resolvable.

- Known residuals: unchanged from v2026.10.20-2.1.1.
