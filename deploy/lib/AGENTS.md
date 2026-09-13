# deploy/lib/ — shared bash helper functions sourced by every deploy script

All three files are libraries: they are `source`d, never executed directly. There is no `__init__`-style entry point — `deploy.sh` sources them in fixed order (`output.sh` → `services.sh` → `vault.sh`, lines 56-58), and `deploy/docker/scripts/` re-sources them from `/opt/mira/app/deploy/lib/` inside the container.

## Rules

- Source order matters: `services.sh` calls `print_info`/`run_with_status` from `output.sh`, and `deploy/lib/vault.sh` calls functions from both. Sourcing `vault.sh` or `services.sh` before `output.sh` leaves functions undefined at call time. The `deploy/lib/vault.sh` header states "Requires: lib/output.sh and lib/services.sh sourced first".
- `LOUD_MODE` must be set before any lib function runs. `run_quiet`, `run_with_status`, `show_progress`, `install_python_package`, and `vault_extract_credential` all branch on it; it is initialized only by `deploy.sh` (`LOUD_MODE=false` default, `--loud` sets true). Docker scripts must set or inherit it themselves.
- `check_exists package` uses the relative path `venv/bin/pip3`, so it only works when the caller's CWD is the MIRA app root (the post-clone deploy context). Running it from elsewhere silently reports "not installed".
- Vault failure semantics in `deploy/lib/vault.sh` are encode-by-exit-code: `vault status` exit 0 = unsealed, 2 = sealed, other = error. `vault_is_sealed` treats errors as sealed (safe default). Do not "fix" the inverted-looking mapping.
- `vault_put_if_not_exists` preserves existing secrets — it skips the write if the path already exists, and calls `exit 1` (not `return 1`) when the write fails. Callers run it at top level; it is not safe inside a subshell where `exit` would not abort the script.
- `vault_extract_credential` prints the credential on stdout and debug output on stderr, so command substitution like `ROOT_TOKEN=$(vault_extract_credential "Initial Root Token")` (as in `finalize.sh`) captures only the value.

## Files

- `output.sh` — ANSI-colored print helpers (`print_header`, `print_step`, `print_success`, `print_warning`, `print_error`, `print_info`), command wrappers (`run_quiet`, `run_with_status`), and the background-job spinner `show_progress PID MSG`. Also defines the color/element variables (`RESET`, `CHECKMARK`, `ARROW`, `WARNING`, `ERROR`) that other scripts interpolate directly in `echo -e`. Consumed by: `deploy/*.sh` (all), `deploy/docker/scripts/container-setup.sh`, `deploy/docker/scripts/init-mira.sh`.
- `services.sh` — idempotent system helpers: `check_exists TYPE TARGET` (types: file, dir, command, package, db, db_user, service_systemctl, service_brew), `start_service`/`stop_service` (systemctl, brew, background/pid_file/port variants), `write_file_if_changed`, `install_python_package`. `db`/`db_user` checks require the `OS` variable (linux uses `sudo -u postgres psql`, else plain `psql`). Consumers: all `deploy/*.sh` except `preflight.sh` (output only), plus both docker scripts.
- `vault.sh` — HashiCorp Vault lifecycle: `vault_is_initialized`, `vault_is_sealed`, `vault_extract_credential`, `vault_unseal`, `vault_authenticate`, `vault_approle_exists`, `vault_initialize`, `vault_put_if_not_exists`. Hardcodes `/opt/vault/` paths (`init-keys.txt`, `role-id.txt`, `secret-id.txt`), the `mira-policy` AppRole policy (KV2 at `secret/`, TTL 1h/max 4h), and single-key-shares init (`-key-shares=1 -key-threshold=1`). `vault_initialize` is idempotent: on an initialized Vault it repairs unseal/auth/engine/AppRole state rather than re-initializing. Consumers: `deploy.sh`, `deploy/vault.sh`, `deploy/postgresql.sh`, `deploy/finalize.sh` (`vault_extract_credential`), `deploy/docker/scripts/init-mira.sh` (`vault_initialize`).

## Wiring

- `deploy.sh` sources all three in order `output.sh` → `services.sh` → `vault.sh`, then orchestrates the script chain (`config.sh`, `preflight.sh`, `dependencies.sh`, `python.sh`, `vault.sh`, `postgresql.sh`, `finalize.sh`), each of which re-declares its own lib requirements in its header. The lib files themselves are stateless beyond the sourced-in variables.
- In the container, `deploy/docker/scripts/init-mira.sh` and `container-setup.sh` source the same lib files from `/opt/mira/app/deploy/lib/`, so any lib change ships into the image — there is no separate container lib copy to keep in sync.
- `install_python_package` (services.sh) depends on `show_progress` (output.sh) for its quiet-mode spinner; `vault_unseal`/`vault_authenticate` depend on `vault_extract_credential` and `run_with_status`. These are intra-directory edges only; the cross-script orchestration flow is owned by `deploy.sh`.

## Vault state machine

`vault_initialize` implements a two-branch state repair with an ordering constraint worth knowing before editing:

- Initialized path: unseal → authenticate → ensure KV2 engine (`secret/`) → ensure AppRole `mira` with `mira-policy` → ensure `/opt/vault/role-id.txt` and `/opt/vault/secret-id.txt` exist (each written only if missing).
- Fresh path: `vault operator init -key-shares=1 -key-threshold=1 > /opt/vault/init-keys.txt` (chmod 600) → then the same sequence, but writing role-id/secret-id unconditionally.

The unseal key and root token are read by `grep ... | awk '{print $NF}'` from `init-keys.txt`, which is why that file is the single source of Vault credentials for the whole deploy (`finalize.sh` re-extracts them later). If the init output format or file path changes, `vault_extract_credential`, `vault_unseal`, `vault_authenticate`, and `finalize.sh` lines 138-139 all break together.
