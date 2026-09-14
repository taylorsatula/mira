# tests/tmp/ — display exhibit for the disposable-probe pattern, not a test home

## Rules

- Disposable checks and probes are written directly in `/tmp` or as inline Python in the conversation, as a matter of courtesy — scratch work does not belong in the repository tree.
- Mocks are never used, without exception: no mock objects, no stubs, no fakes, no simulated infrastructure. Verification is live or it doesn't count.
- Any test file added to this folder will be **autodeleted instantly** on sight. This folder is not a test suite, is not run by anything, and will never contain more than the exhibit below.
- The folder exists only to display one exemplary disposable probe, so a session can see the shape: live infrastructure, real credentials plumbing, no mocks, executed once, then kept as an example rather than as a check.
- Verification in MIRA is live (see the NO MOCKS section of the root `AGENTS.md`). A one-off probe that earns permanence becomes a production path-probe registered alongside the POST gate — not a file here.
