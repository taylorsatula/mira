# deploy/vm/ — VM spin-up, dev-build deploy, and instance-sarcophagus restore toolkit

One command (`oneshot.sh`) yields a running VM with a fresh dev-build MIRA and a
chosen sealed sarcophagus restored onto it, verified against `SNAPSHOT-FACTS.txt`.
Modes: local libvirt (default), `--host user@host` (orchestrated over ssh; virsh
runs on the host), `--ip <addr>` (plain-ssh onto any existing VM). The
forker-facing guide with the full command quickbook is `README.md`.

## Rules

- Every driver owns its argument loop and assigns flags one at a time through `set_common` in `lib.sh`: a shared parser cannot consume the caller's positional args, because a function receives a copy of `$@` and its shifts never propagate (a `parse_common_flags` variant infinite-looped on `--domain`).
- Two ssh stdin disciplines in `lib.sh`, never mixed: `vmssh` runs `ssh -n` so command execution cannot eat a caller's read-loop stdin, while `vmstream` and every pipe-into-ssh transfer must omit `-n` or the producer SIGPIPEs. `talktomira.sh` applies the same rule to its remote heredoc.
- In `--host` mode every remote-shell argument goes through `printf %q` (`lib.sh:VIRSH`, and `oneshot.sh:hostfs` on the same ssh channel): ssh adds a second shell hop that otherwise eats inner quotes — unescaped, virsh received `{execute:guest-ping}` instead of valid JSON.
- Password bootstrap (`lib.sh:bootstrap_ip`) uses a single-stdin `sudo -S` form: dash has no herestrings, and a heredoc on the same command as the password pipe overrides the pipe, so sudo would read the sudoers line as the password.
- Transport mode is finalized only after argument parsing (`finish_flags` in `lib.sh`): `--ip` cannot be known at source time, so `IS_LIBVIRT` must not be derived there.
- The deploy is greenfield-only: `oneshot.sh` gates on the valkey `user_lock:*` scan, stops mira, terminates Postgres backends, and drops `mira_service` before re-deploying — an active connection silently blocks `DROP DATABASE` behind any `2>/dev/null`.
- `""`-key deploy configs park the mira POST gate by design; a healthy endpoint comes from `inject.sh` (real Vault/routes arrive with the sarcophagus). Exception: a config carrying real keys is seeded into Vault at deploy time (`deploy/postgresql.sh` Step 14: openai chat mode maps `chat_api_key`→`provider_key`, `subcortical_api_key`→`subcortical_key`), so a no-inject deploy can come up healthy with live routes — validated 2026-09-18 by the deploy-only flow (fresh template spawn + dev deploy, driver preserved on the reference host as bin/deploy-only.sh).
- Sarcophagi are sealed and append-only: `extract.sh` refuses an existing output directory, and any post-seal edit must regenerate `MANIFEST.sha256` and say so.
- `inject-restore-vm.sh` restores the whole dump (drop + recreate + `pg_restore`), so the restored schema is the sarcophagus's, not the deployed build's. A sarcophagus extracted before the `embedding_config` table existed restores without it, and the app then fails at boot on `load_embedding_config` (`clients/AGENTS.md`); its vectors are `mdbr-leaf-ir-asym`/768. Re-extract from an instance running current code before relying on such a sarcophagus.
- Restores remap the VM user: `inject-restore-vm.sh` rewrites `User=`/`Group=` in the app-owned captured units (`mira.service`, `vault.service`; `valkey.service` keeps its distro `User=valkey`) and relocates `MIRA_credentials.txt`/`.vault-token` to the actual user; the captured `authorized_keys` is deliberately not restored (the operator's own access is bootstrapped).
- `talktomira.sh` mints and abandons an `api_tokens` row per invocation (name `talktomira-cli-<ts>`); prune them periodically.
- All three modes are exercised end-to-end (2026-09-16/17); the `--ip` mode was validated on an aarch64 workstation VM — dev build deployed in 182 s, v3 sarcophagus restored, 9/9 facts PASS, chat-continuity confirmed from restored memories — and `extract.sh` against the same VM cross-arch (the remap rule above followed `--vm-user`).

## Files

- `lib.sh` — shared config, transports, and helpers; owns the transport and flag rules above. `VIRSH`, `vmssh`, `vmstream`, `vmcp_from` (defined but unused in this tree — the pull paths scp directly; do not build on it), `vmcp_to`, `vmexec`, `vmip`, `wait_guest_agent`, `bootstrap_ssh`, `bootstrap_ip`, `turn_lock_gate`, `resolve_sarc`, `set_common`, `finish_flags`.
- `oneshot.sh` — the whole flow: sarcophagus resolve → spawn/reuse VM → bootstrap ssh → push source + config → deploy → mira-service poll → exec `inject.sh`. The sarcophagus arg is mandatory (deploy-only is not a flag); for a fresh-template deploy-only run, replicate phases 1–4 without inject — reference host's bin/deploy-only.sh is the validated driver (README quickstart D).
- `extract.sh` — live instance → sealed sarcophagus; writes `SNAPSHOT-FACTS.txt` and `MANIFEST.sha256`, verifies VM↔caller byte parity and caller-side sqlite integrity. Materializes on the caller's `--snap-dir`; the two-homes sync + post-seal README/manifest steps are the README sarcophagus contract.
- `extract-stage-vm.sh` — in-VM staging half of `extract.sh` (pg_dump, sqlite `.backup()` snapshots, quiescent tars).
- `inject.sh` — restore driver: quiesce gate, payload push, run `inject-restore-vm.sh`, verify facts.
- `inject-restore-vm.sh` — in-VM restore half: app overlay, Postgres drop/recreate/`pg_restore --no-owner`, user data, Vault, credential remap, unit `User=` rewrite, health poll.
- `vm-exec.sh` — CLI for `lib.sh:vmexec` (guest-agent root exec; libvirt modes only).
- `make-base-template.sh` — EXPERIMENTAL: builds a default-state base template qcow2 from an Ubuntu cloud image via cloud-init seed.
- `base-vm.xml` — domain skeleton with `__DOMAIN__`/`__DISK__` placeholders; carries the qemu-guest-agent channel `oneshot.sh` bootstraps through.
- `deploy-config-dev-vm.yml` — non-interactive deploy answers; API keys ship as `""` placeholders that inject replaces with the real Vault. A copy with real keys set is the deploy-only exception: it seeds Vault directly and yields a healthy no-inject deploy — generate such a config from a sarcophagus Vault, never commit one, keep it 0600 and delete it after the deploy.
- `talktomira.sh` — one-liner chat with a deployed instance from any machine: resolves the VM IP on the libvirt host, mints a Bearer token, posts to `/v0/api/chat` with `--max-time 900`.
- `README.md` — forkers' guide: modes, prerequisites, sarcophagus contract, MIRA API quickbook, and the gotcha list.

## Wiring

- `oneshot.sh` resolves the sarcophagus via `lib.sh:resolve_sarc` (manifest + facts verified; `--host` mode syncs the host's `SNAP_DIR` copy to the caller first), pushes the source tree through `vmstream` plus the config through `vmcp_to`, runs the repo's `deploy/deploy.sh --config … --local` inside the VM under nohup with `/tmp/deploy.log` + `/tmp/deploy.exit` polling, then execs `inject.sh` with the resolved mode flags.
- `inject.sh` → `inject-restore-vm.sh` and `extract.sh` → `extract-stage-vm.sh` pair per side of the ssh boundary; both drivers route through `lib.sh` transports — except `extract.sh`'s pull path, which re-implements `vmcp_from` inline as a raw `scp` (ProxyJump handled, same behavior) — so all three modes share one code path.
- `talktomira.sh` is standalone: ssh to the libvirt host → `virsh domifaddr` → session/CSRF/token mint → `/v0/api/chat`, with the message base64-transported through both ssh and shell quoting.

## Reference-host recipes

**Deploy + sarcophagus restore** (on the host, or from a workstation via `--host`; `$SNAPSHOTS` = the host's snapshot-toolkit directory, `$LIBVIRT_HOST` = `user@host`):

```bash
ssh "$LIBVIRT_HOST" \
  "$SNAPSHOTS/bin/oneshot.sh" \
  "$SNAPSHOTS/mlfactory_v4_mira"   # or another sarcophagus
```

Spawns a fresh VM from the host's default-state frozen base template, deploys a dev build from the host's worktree snapshot of this repo (refresh recipe below), injects the sarcophagus (Postgres, user data incl. domaindocs, Vault with real keys, units), and verifies health + row counts against the sarcophagus's `SNAPSHOT-FACTS.txt`. `--fresh` rebuilds a running VM (old disk preserved); omit it to reuse a running one.

**Deploy-only on a fresh VM (no instance state):** `oneshot.sh` mandates a sarcophagus by design, so run its phases 1–4 without inject — the validated driver is bin/deploy-only.sh on the reference host (2026-09-18: template spawn + dev deploy, healthy in 152 s; the no-inject real-key contract is the Rules bullet above). To redeploy onto an already-running VM, run `deploy/deploy.sh --config <yml> --local` by hand inside it (greenfield rule above; `postgresql.sh` renames a populated `mira_service` aside — `deploy/AGENTS.md`).

**Refresh the host's source snapshot after changing this worktree** — the host deploys from its copy, not from here. The remote side MUST clear the worktree before extracting: a tar overlay never deletes files removed from the source, so a stale overlay silently ships retired modules (observed 2026-09-30: a quarantined tool left in `tools/implementations/` parked MIRA's boot gate with an ImportError). The clear preserves the host-local state the tarball excludes and cannot restore (`.git`, `venv`, `.env`, `data`, `logs`, `scratch`, `.claude`, `.DS_Store`) and removes everything else at the top level:

```bash
cd ~/Programming/GitHub/mira-OSS && tar --exclude=.git --exclude=data --exclude=logs \
  --exclude=scratch --exclude=__pycache__ --exclude='*.pyc' --exclude=.env \
  --exclude=venv --exclude=.claude -czf - . | \
  ssh "$LIBVIRT_HOST" 'mkdir -p "$MIRA_HOST_WORKTREE" \
    && find "$MIRA_HOST_WORKTREE" -mindepth 1 -maxdepth 1 \
    ! -name .git ! -name venv ! -name .env ! -name data ! -name logs \
    ! -name scratch ! -name .claude ! -name .DS_Store -exec rm -rf {} + \
    && tar -C "$MIRA_HOST_WORKTREE" -xzf -'
```

**Working with a deployed instance** (minting API tokens, chat endpoint, DB probing, turn-in-flight rules, memory/schema maps): read the snapshot toolkit's `AGENTS.md` on the host ($SNAPSHOTS/AGENTS.md) — the full quickbook of validated commands. Sarcophagi lineage (v1/v2/v3), extraction tooling, and restore contracts are documented there too.
