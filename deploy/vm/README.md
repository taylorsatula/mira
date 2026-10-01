# deploy/vm — rapid MIRA VM snapshotting + spinup toolkit

One command produces a **running VM with a fresh dev-build MIRA and a chosen
instance sarcophagus restored onto it, verified**. Sarcophagi are sealed,
checksummed, append-only snapshots of a live MIRA instance — extract one from
any deployed VM, restore it onto any fresh one.

```bash
./oneshot.sh <sarcophagus>          # the whole point
./extract.sh <new-name>            # seal a live instance into a new sarcophagus
./inject.sh  <sarcophagus>         # restore half (also phase 6 of oneshot)
./vm-exec.sh 'command'              # run as root in the VM (guest-agent)
./make-base-template.sh             # EXPERIMENTAL: build a base template first
```

## Three modes (auto-selected by flags)

| Mode | Flags | Use when |
|---|---|---|
| **libvirt local** | *(default)* | Same machine runs libvirt. `oneshot.sh` spawns/reuses a domain from the frozen base template (`--vmimg-dir`, default `$HOME/virtual_machine`). |
| **libvirt remote** | `--host user@host` | Orchestrate from anywhere over ssh — e.g. from a Mac against a Linux box. All `virsh` commands execute **on the host** (macOS has no virsh; do not use qemu+ssh URIs). VM ssh/scp jump through the host. Sarcophagi sync host→caller first. |
| **plain ssh** | `--ip ADDR [--vm-pass PW]` | The VM exists but libvirt doesn't manage it (UTM/Lima/cloud/other). No spawn phase. `--vm-pass` performs the one-time bootstrap (install caller's pubkey + passwordless sudo — headless deploys cannot answer sudo prompts) via sshpass or expect. |

Common flags (all scripts, env-equivalents in `lib.sh`): `--domain NAME`
(default `ubuntu_vm`), `--vm-user U` (default `ubuntu`), `--snap-dir DIR`
(sarcophagi root; default `$HOME/mira-snapshots`), `--vmimg-dir DIR`,
`--template FILE`, `--xml FILE`, `--source DIR` (mira-OSS checkout to deploy —
defaults to the repo containing this toolkit, **including uncommitted changes**),
`--config FILE` (deploy answers yml; default `deploy-config-dev-vm.yml`).

## Quickstarts

```bash
# A. New VM from template + dev build + instance restored + verified
./oneshot.sh mlfactory_v4_mira                  # local libvirt
./oneshot.sh --host <user>@<libvirt-host> mlfactory_v4_mira   # orchestrated remotely
./oneshot.sh --ip <vm-ip> --vm-user mira_service --vm-pass '…' mlfactory_v4_mira
./oneshot.sh mlfactory_v4_mira --fresh          # rebuild a running VM (gated on
                                                 # no in-flight MIRA turn; old disk
                                                 # kept as <disk>.pre-oneshot-<ts>)

# B. Save a live instance before destroying its VM — always do this first
./extract.sh my-instance-$(date +%F)
#    In --host mode the sarcophagus materializes on the CALLER (--snap-dir);
#    sync it to the host lineage dir and verify MANIFEST.sha256 on both ends.
#    Two mandatory post-extract steps: replace the placeholder README-RESTORE.md
#    header with a real orientation (delta vs the previous sarcophagus), then
#    regenerate MANIFEST.sha256 (the contract for post-seal edits).

# C. Redeploy the dev build onto an already-running VM (no instance state) —
#    from the repo root of the checkout on the VM:
#    ./deploy/deploy.sh --config deploy-config.yml --local --loud
#    (that is exactly oneshot's phase 4; oneshot always pairs it with a restore)

# D. Fresh VM from the template + dev build only — no sarcophagus, no inject
#    (validated 2026-09-18): oneshot.sh mandates a sarcophagus arg by design,
#    so drive its phases 1–4 minus inject. The validated driver is the
#    reference host's bin/deploy-only.sh: turn-lock gate → graceful shutdown
#    with the old disk preserved as <disk>.pre-oneshot-<ts> → template →
#    fresh disk → bootstrap → push → deploy → service + health poll.
#    Keys decide health: with "" placeholder keys the POST gate parks by
#    design (inject is what installs real Vault/routes); with real
#    chat_api_key + subcortical_api_key in the config, deploy/postgresql.sh
#    seeds them into Vault and the instance comes up healthy with live model
#    routes and no inject. Generate the key-bearing config from a
#    sarcophagus Vault (never a literal in a committed file), keep it 0600,
#    delete it after the deploy.
```

**Prerequisites:** for libvirt modes — a frozen **base template** qcow2
(default-state Ubuntu ≥24.04: ssh + a sudo user + qemu-guest-agent; MIRA is
deployed fresh every time — try `make-base-template.sh`, EXPERIMENTAL) and a
domain XML skeleton (`base-vm.xml` ships with the toolkit). For remote mode —
ssh access to the host with libvirt permissions. For plain-ssh — a reachable
Ubuntu VM with ≥4 GB RAM, ≥15 GB disk. The caller needs a public key
(`$SSH_PUB`, else `~/.ssh/id_ed25519.pub`, else id_rsa, else ssh-agent).

## The sarcophagus contract

Sealed dir, append-only: `postgres/mira_service.dump` (pg_dump -Fc), `data-users.tar.gz`
(consistent sqlite `.backup()` snapshots — never tar a live WAL), `app-code.tar.gz`
(live source incl. llm_*.jsonl), `vault.tar.gz` (complete /opt/vault: init keys,
AppRole, the real API keys — Vault is restored wholesale, so no secret typing ever),
`home-ubuntu.tar.gz` (filename is contractual even when the VM user differs;
contains instance credentials), `systemd/` (units + valkey.conf), `system-info/`,
`SNAPSHOT-FACTS.txt`, `MANIFEST.sha256`. inject refuses manifest mismatch,
missing facts, or an in-flight MIRA turn. Never edit a sealed one without
regenerating its manifest and saying so.

Two homes for every sarcophagus: `extract.sh` writes the caller's `--snap-dir`
(default `~/mira-snapshots`); in `--host` mode that is the orchestrating machine,
so sync the sealed dir to the host lineage root and verify `MANIFEST.sha256` on
both ends before destroying the VM it came from. The lineage table (v1/v2/v3/…
facts and what changed) lives in the host lineage dir's `AGENTS.md` — add a row
per extraction, after replacing the placeholder `README-RESTORE.md` header with a
real orientation and regenerating the manifest.

## Talking to the deployed instance (validated commands — don't re-derive)

```bash
# inside the VM: http://127.0.0.1:1993 — from the caller: http://<vm-ip>:1993
JAR=$(mktemp)
curl -s -c $JAR http://127.0.0.1:1993/v0/auth/local/session            # session cookie
CSRF=$(curl -s -b $JAR -c $JAR -X POST http://127.0.0.1:1993/v0/auth/csrf \
       | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"]["csrf_token"])')
TOKEN=$(curl -s -b $JAR -H "X-CSRF-Token: $CSRF" -H 'Content-Type: application/json' \
        -d '{"name":"agent"}' -X POST http://127.0.0.1:1993/v0/auth/api-tokens \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"]["token"])')
curl -s --max-time 900 -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"message":"hello"}' -X POST http://127.0.0.1:1993/v0/api/chat
# reply: data.response, data.metadata.{tools_used,surfaced_memories,processing_time_ms}
```

Cookie-auth POSTs need `X-CSRF-Token`; Bearer does not. Probe health first:
`GET /v0/api/health`.

## Gotchas — hard-won, all encoded in the code; do not "simplify" them away

1. `ssh -n` for command execution (else ssh eats a caller's read-loop stdin);
   plain ssh for pipe/stream transfers (else the producer SIGPIPEs).
2. `cmd && echo ok` does **not** abort under `set -e` — make verification
   failures fatal with `|| { …exit 1; }`.
3. virsh `domstate` on a missing domain can return empty without failing the
   `||` fallback — normalize state explicitly.
4. The deploy is **greenfield-only** (installs schema into an empty
   `mira_service`) — drop the DB before re-deploys (oneshot does).
5. `""`-key deploy configs **park the POST gate by design** — health comes from
   inject (real Vault/routes from the sarcophagus). Exception: a config carrying real
   keys is seeded into Vault at deploy time (`deploy/postgresql.sh` Step 14 maps
   `chat_api_key`→`provider_key` and `subcortical_api_key`→`subcortical_key` in
   openai chat mode), so a no-inject deploy can be fully healthy with live routes — that
   is how quickstart D runs. `model_configs.api_key_name` names a field inside
   `secret/mira/api_keys`: it just has to exist there (a restored instance may use a
   different field name, e.g. `lunaroute_key`).
6. `/etc/valkey` is root-only — staging as a normal user silently drops
   valkey.conf behind `2>/dev/null`; fail loudly instead (a silent gap broke
   the first production inject).
7. VM DHCP addresses change with every spawned MAC — resolve via
   domifaddr/guest-agent; never hardcode.
8. Never stop/flush/restart mira with a turn in flight (`user_lock:*` in
   valkey); long turns block their HTTP call — `--max-time 900+` or poll.
9. App INFO logs don't reach the journal (WARNING+ only) — verify via DB/API.
10. sqlite under `/opt/mira/app/data` must be touched as the VM user (root
    hits `readonly database`); snapshot with the sqlite3 `.backup()` API.
11. An apostrophe inside `${var:?msg}` breaks bash parsing.
12. Headless deploys cannot answer sudo prompts — bootstrap passwordless sudo
    (`--vm-pass` does it; cloud images ship NOPASSWD by default).
13. `deploy.sh --local` installs from the *invocation cwd* — run it from the
    repo root; it excludes `.git, venv, __pycache__, *.pyc, .env, .claude,
    .DS_Store, data, logs, scratch` (parity with the GitHub tarball).
14. `deploy/python.sh` no longer string-patches the schema. Hosted installs
    apply the seeded (lunaroute-default) `model_configs` rows, then
    `deploy/postgresql.sh` rewrites `primary` from the chat config and the four
    aux routes from the subcortical config with UPDATEs after application (the
    same mechanism as OFFLINE_SQL) — a chat_model that is empty in anthropic
    mode, or empty chat_model/chat_endpoint in openai mode, aborts the install
    (guarded in `deploy/lib/config_file.sh` pre-sudo and again at Step 13).

## Reference deployment

The canonical deployment lives on the operator's libvirt host (<libvirt-host>, Ubuntu 26.04):
domain `ubuntu_vm`, base template `/home/admin/virtual_machine/ubuntu_vm-template.qcow2`
(default-state, purged + verified), sarcophagi v1–v4 under
`/home/admin/mira_instance_snapshots/` (see that dir's AGENTS.md for the full
instance lineage and operational history), plus `bin/deploy-only.sh` — the
validated fresh-template deploy-only driver (quickstart D). `make-base-template.sh` is
EXPERIMENTAL — not yet exercised end-to-end; the reference template was built
by hand. The same-named toolkit under `/home/admin/mira_instance_snapshots/bin/`
is the tuned operational original; this directory is its generalized descendant.
