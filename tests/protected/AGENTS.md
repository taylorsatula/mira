# tests/protected/ — admission-gated permanent verification batteries

## Rules

- **Admission requires the user to type the exact phrase `AUTHORIZE PROTECTED TEST SAVE`.**
  Nothing else admits a file here: not agent judgment, not "this battery is too
  valuable to throw away", not a prior authorization on a different file, not a
  paraphrase, not an inferred intent. Authorization is per-file and per-save —
  one phrase covers the one artifact it was said about. An agent that believes a
  probe deserves permanence asks; it does not decide.
- Until that phrase appears, a probe being written here is scratch work and
  belongs in `/tmp` per `tests/tmp/AGENTS.md`.
- Files here are **never autodeleted**. That is the whole distinction from
  `tests/tmp/`, whose autodelete-on-sight policy is owned by
  `tests/tmp/AGENTS.md`. A session that finds this folder and reaches for the
  `tests/tmp/` reflex is wrong: check for this map before deleting anything.
- This folder is the sole authorized exception to the root `AGENTS.md` NO MOCKS
  prohibition on test files. The exception is narrow and does not relax any
  other part of that doctrine — a file admitted here still may not contain mock
  objects, stubs, fakes, or simulated infrastructure. It must call real
  production code. What is waived is the "disposable, never permanent"
  requirement, not the "no mocks, verification is real" requirement.
- **A battery here must test the shipped artifact, not a transcription of it.**
  Copying a pattern list, constant table, or validation function into the probe
  so it can be exercised without importing the module is prohibited: the copy
  drifts the first time the source is edited, and a battery that passes against
  a stale copy asserts nothing about production. Import the real symbol and call
  it.
- **A battery exercising a destructive surface must be structurally incapable of
  reaching it**, not merely careful. Required shape: install a
  `sys.addaudithook` that raises on every process-spawn and network audit event
  *before* importing the module under test, so an accidental call path aborts
  the interpreter instead of touching a host. Call the pure decision function
  directly; never construct the tool, never call its `run()` or transport
  methods. See the safety contract at the top of
  `mlfactory_guardrail_probe.py`.
- A battery here must be runnable offline and must not require Vault, Postgres,
  Valkey, embeddings, or any live service. If a check needs live infrastructure
  it is a POST path-probe under `utils/power_on_self_test.py` (ownership: root
  `AGENTS.md`), not a file in this folder. "Offline" means no connection is
  opened, not that the import graph is small: importing any module under
  `tools/` pulls in `clients/__init__.py`, which eagerly imports the Vault,
  Postgres, Valkey, SQLite and embeddings clients and constructs the config
  singleton. Those clients keep their connections lazy, and this folder's
  batteries depend on that staying true — a client gaining an eager import-time
  connection breaks every battery here.
- One recorded exception to the offline rule, by the user's direction (2026-09-26): `injection_defense_probe.py`'s live half calls the configured System One endpoint. Its audit barrier admits name resolution of that host and connections to the addresses it resolved to, nothing else; the bearer token comes from `SYSTEMONE_API_KEY` for as long as the user keeps that temporary arrangement. Every other rule here still binds it.
- Importing project code writes `.pyc` caches via a temp file plus `os.rename`,
  which the required audit barrier refuses. A battery that installs the
  filesystem-mutation half of the barrier must set
  `sys.dont_write_bytecode = True` before the project import, rather than
  weakening the barrier to let the interpreter's own caching through.
- Every case carries a stated expectation — destructive cases must be refused,
  benign cases must be allowed. A battery that only asserts "blocked" will pass
  forever while the guardrail over-blocks legitimate work into uselessness, so
  the false-positive half is mandatory, not optional.

## Files

- `mlfactory_guardrail_probe.py` — Regression battery for the
  `tools/implementations/bash_tool.py` destructive-command guardrail.
  Calls the real module-level `_validate_command(command, root, cwd)` against a
  destructive corpus (filesystem-root and project-root deletion including
  relative, quoted, globbed and shell-expanded spellings; system-tree deletion;
  disk and filesystem destruction; power and runlevel changes; system permission
  and ownership changes; mass process signalling; system-config writes;
  download-to-shell; service and package removal; fork bombs; the user's
  standing git rules refusing `checkout`/`restore`/`reset --hard`/`clean -f`;
  other host damage) and a benign corpus of realistic ML-harness commands that
  must not be refused. Exits non-zero on any bypass or false positive and prints
  the firing rule name per case, so an over-broad match is visible rather than
  silently credited. Self-checks before running that `_validate_command` is
  still pure, via a recursive `dis` walk allowlisting every global it loads.
  Run from the repo root:
  `python3 tests/protected/mlfactory_guardrail_probe.py` (the `-m` form fails —
  no `__init__.py` in `tests/`).
- `injection_defense_probe.py` — Battery for `utils/untrusted_content.py`. Offline half (network blocked): `_reveal_hidden` must surface each hidden payload (base64, hex, `\x` escapes, percent, HTML entities, tag characters, letter-spacing) and reveal nothing on designated benign text; `wrap_untrusted` over every case and hostile source labels must keep exactly one boundary, no raw `<`/`>`/`"` inside it, and no tag characters. Live half: tier-1 `InjectionScreen.assess()` over 64 labeled attacks (indirect and subtle manipulation; each names the questions it should trigger) and 46 hard benign cases through a real `SystemOneClient` built from `config.systemone`; `screen()` witnessed on one auto-rejected and one auto-passed case. Tier 2 (the escalation LLM) is not exercised — it resolves a route from `model_configs`. Fails only on a safety failure (attack auto-passed unless recorded as a documented miss; benign auto-rejected), an offline failure, a witness failure, or a blocked side effect; reports drift between safe dispositions, per-question counts, corpus loss, and the smallest safety margins for wording work. Run from the repo root: `SYSTEMONE_API_KEY=... python3 tests/protected/injection_defense_probe.py`.
