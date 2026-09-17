# deploy/vm/ — VM spin-up, dev-build deploy, and instance-sarcophagus restore toolkit

One command (`oneshot.sh`) yields a running VM with a fresh dev-build MIRA and a
chosen sealed sarcophagus restored onto it, verified against `SNAPSHOT-FACTS.txt`.
Modes: local libvirt (default), `--host user@host` (orchestrated over ssh; virsh
runs on the host), `--ip <addr>` (plain-ssh onto any existing VM). The
forker-facing guide with the full command quickbook is `README.md`.

## Rules

- Every driver owns its argument loop and assigns flags one at a time through `set_common` in `lib.sh`: a shared parser cannot consume the caller's positional args, because a function receives a copy of `$@` and its shifts never propagate (a `parse_common_flags` variant infinite-looped on `--domain`).
- Two ssh stdin disciplines in `lib.sh`, never mixed: `vmssh` runs `ssh -n` so command execution cannot eat a caller's read-loop stdin, while `vmstream` and every pipe-into-ssh transfer must omit `-n` or the producer SIGPIPEs. `talktomira.sh` applies the same rule to its remote heredoc.
- In `--host` mode every `VIRSH` argument goes through `printf %q` (`lib.sh:VIRSH`): ssh adds a second shell hop that otherwise eats inner quotes — unescaped, virsh received `{execute:guest-ping}` instead of valid JSON.
- Password bootstrap (`lib.sh:bootstrap_ip`) uses a single-stdin `sudo -S` form: dash has no herestrings, and a heredoc on the same command as the password pipe overrides the pipe, so sudo would read the sudoers line as the password.
- Transport mode is finalized only after argument parsing (`finish_flags` in `lib.sh`): `--ip` cannot be known at source time, so `IS_LIBVIRT` must not be derived there.
- The deploy is greenfield-only: `oneshot.sh` gates on the valkey `user_lock:*` scan, stops mira, terminates Postgres backends, and drops `mira_service` before re-deploying — an active connection silently blocks `DROP DATABASE` behind any `2>/dev/null`.
- Placeholder API keys park the mira POST gate by design; a healthy endpoint comes from `inject.sh` (real Vault/routes arrive with the sarcophagus), never from a fresh deploy.
- Sarcophagi are sealed and append-only: `extract.sh` refuses an existing output directory, and any post-seal edit must regenerate `MANIFEST.sha256` and say so.
- Restores remap the VM user: `inject-restore-vm.sh` rewrites `User=`/`Group=` in the captured units and relocates `MIRA_credentials.txt`/`.vault-token` to the actual user; the captured `authorized_keys` is deliberately not restored (the operator's own access is bootstrapped).
- `talktomira.sh` mints and abandons an `api_tokens` row per invocation (name `talktomira-cli-<ts>`); prune them periodically.

## Files

- `lib.sh` — shared config, transports, and helpers; owns every cross-file rule above. `VIRSH`, `vmssh`, `vmstream`, `vmcp_from`, `vmcp_to`, `vmexec`, `vmip`, `wait_guest_agent`, `bootstrap_ssh`, `bootstrap_ip`, `turn_lock_gate`, `resolve_sarc`, `set_common`, `finish_flags`.
- `oneshot.sh` — the whole flow: sarcophagus resolve → spawn/reuse VM → bootstrap ssh → push source + config → deploy → mira-service poll → exec `inject.sh`.
- `extract.sh` — live instance → sealed sarcophagus; writes `SNAPSHOT-FACTS.txt` and `MANIFEST.sha256`, verifies VM↔caller byte parity and caller-side sqlite integrity.
- `extract-stage-vm.sh` — in-VM staging half of `extract.sh` (pg_dump, sqlite `.backup()` snapshots, quiescent tars).
- `inject.sh` — restore driver: quiesce gate, payload push, run `inject-restore-vm.sh`, verify facts.
- `inject-restore-vm.sh` — in-VM restore half: app overlay, Postgres drop/recreate/`pg_restore --no-owner`, user data, Vault, credential remap, unit `User=` rewrite, health poll.
- `vm-exec.sh` — CLI for `lib.sh:vmexec` (guest-agent root exec; libvirt modes only).
- `make-base-template.sh` — EXPERIMENTAL: builds a default-state base template qcow2 from an Ubuntu cloud image via cloud-init seed.
- `base-vm.xml` — domain skeleton with `__DOMAIN__`/`__DISK__` placeholders; carries the qemu-guest-agent channel `oneshot.sh` bootstraps through.
- `deploy-config-dev-vm.yml` — non-interactive deploy answers; API keys ship as `""` placeholders that inject replaces with the real Vault.
- `talktomira.sh` — one-liner chat with a deployed instance from any machine: resolves the VM IP on the libvirt host, mints a Bearer token, posts to `/v0/api/chat` with `--max-time 900`.
- `README.md` — forkers' guide: modes, prerequisites, sarcophagus contract, MIRA API quickbook, and the gotcha list.

## Wiring

- `oneshot.sh` resolves the sarcophagus via `lib.sh:resolve_sarc` (manifest + facts verified; `--host` mode syncs the host's `SNAP_DIR` copy to the caller first), pushes the source tree through `vmstream` plus the config through `vmcp_to`, runs the repo's `deploy/deploy.sh --config … --local` inside the VM under nohup with `/tmp/deploy.log` + `/tmp/deploy.exit` polling, then execs `inject.sh` with the resolved mode flags.
- `inject.sh` → `inject-restore-vm.sh` and `extract.sh` → `extract-stage-vm.sh` pair per side of the ssh boundary; both drivers use only `lib.sh` transports so all three modes share one code path.
- `talktomira.sh` is standalone: ssh to the libvirt host → `virsh domifaddr` → session/CSRF/token mint → `/v0/api/chat`, with the message base64-transported through both ssh and shell quoting.
