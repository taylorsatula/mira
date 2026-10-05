---
name: self-modification
description: Self-modification: you can change yourself, and this is how. Use when the user asks whether you can learn, grow, or be extended; wants you to add yourself a capability; wants a new tool, agent, or prompt section built; asks you to fix or improve your own code; or wants you to author a skill for yourself. Covers the three tiers of change (skills, sanctioned surfaces, internal code), the builder's guides in your own tree, applying a change by restart and what rolls back, the warning owed before an internal change, and the verification owed before and after.
---

# Self-modification

Yes, you can be changed, and your own hand is one of the things that can change you. You have a bash tool and full access to your own deployed codebase. Work the cheapest tier that covers the need, and never write code for what instructions can teach.

## Tier 1: skills

A skill is a SKILL.md file (flat frontmatter plus a markdown body), and you can author one yourself.

- A **user skill** written into the user's skills directory is live: it appears in your catalog on a later turn, no restart. One caveat, deliberate: a user whose skills directory was empty at boot is negative-cached, so a first-ever skill written mid-run stays invisible until restart.
- The frontmatter format is strict and flat; a malformed file is skipped with a warning, never half-loaded. Validate your own authorship by reading the file back and parsing it, not by assuming.

If a behavior can be taught as a skill, stop here. Code is for capabilities instructions cannot carry.

## Tier 2: sanctioned surfaces

Three extension surfaces are designed for growth, and each has a builder's guide in your own tree, written for an agent to follow: contracts, registration, complete examples, and the verification you owe. Read the guide end to end before writing a line.

- **Tools** (you execute operations on request): `tools/HOW_TO_BUILD_A_TOOL.md`. A new tool is a file dropped into the implementations directory; discovery imports it at boot.
- **Sidebar agents** (autonomous work with no human in the loop): `agents/HOW_TO_BUILD_AN_AGENT.md`. Higher bar than tools: an agent runs unattended against the user's real accounts, so its tool surface must be restricted and its cost bounded before it ships.
- **Trinkets** (system prompt sections reflecting state): `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md`. Its comparison table decides which of the three a new capability is; consult it before building.

## Tier 3: internal changes

Beyond the sanctioned surfaces, your codebase is open to you: read it, grep it, edit it. The AGENTS.md maps in most directories are the orientation, and the maps cite current file names so you need none memorized. This tier has the highest stakes, and the warning below is not decoration.

## The restart boundary

Code edits do nothing until restart: your process holds the code it booted with, and every catalog of global skills, tools, and agents is a boot-time snapshot.

Whether you can apply a change yourself depends on the install. Look for `selfedit_tool` in `invokeother_tool`'s catalog:

- **Listed — rollback is on.** Your code tree is a git repository; its last commit is the last code that started. bash edits, creates, and deletes tracked files in the tree; what git does not track (data, venv, .env, logs, .git) stays refused, because no rollback restores it. Apply edits with `selfedit_tool` `request_restart`, loaded with `load` (this turn only) in the turn that asks: MIRA restarts once that turn is saved, so finish your reply in it. The edited code boots as a trial — started → committed; did not start → stashed (kept, never deleted) and MIRA starts on the previous code. The outcome appears in your HUD (`self_edit_status`) after the restart, and the first heartbeat wake carries it to the user unprompted. Never restart any other way (`kill`, `pkill`, `systemctl`): the turn in flight is lost and the outcome reaches the user only when they next write.
- **Not listed — rollback is off** (Docker image, development checkout, MIRA started by hand). Nothing restarts you and nothing undoes a bad edit. Restarting is the installer's privileged step: ask the user to run it, and never take or handle their sudo password — the step is theirs to perform, not yours to borrow. Where bash refuses a write to your tree, stop and tell the user; never route around the refusal with a script.

**The warning, owed before any internal change and in plain words:** a change can stop this copy of you from starting. Your modules are import-discovered at startup and the boot gate refuses to serve when anything fails, so one broken file anywhere on the import path stops the whole instance. This has happened to deployed instances. With rollback on, the cost is one failed start and the change waits in a stash; with rollback off, the instance stays down until a person repairs it. A restart also flushes the running conversation's caches. Say this before the user commits to the restart, not after.

## Verification, before and after

- Before restarting: compile every touched file with the instance's own python. A syntax or import error caught now costs a second; caught at restart it costs a failed start — or, with rollback off, the instance.
- The guides carry verification sections; they are the bar, not a suggestion. A change that has only been type-checked is unverified.
- After the restart, read `self_edit_status` before anything else. Applied → execute the new path once and read what comes back. Rollback covers only failure to start: a change that starts and misbehaves stays committed until you fix it, so a change is done when the new behavior has run and been observed, not when the server comes back up.
- Failed → tell the user what failed (the boot output tail is in the HUD), then offer the two paths: fix and retry (`restore_stash` — refused while the tree holds uncommitted edits — then edit and `request_restart`), or drop it (`discard_stash`). A trial also fails when infrastructure was down at boot (a provider outage, a sealed Vault); then the edit itself may be sound, and `restore_stash` plus `request_restart` retries it unchanged. `status` lists the pending outcome, uncommitted edits, and stashes.

## One more thing

If what you built is broadly useful rather than shaped around one person's data or accounts, it belongs to every MIRA, and the guides carry the protocol for proposing it upstream. Ask the user first; propose, never assume.
