---
name: self-modification
description: Self-modification: you can change yourself, and this is how. Use when the user asks whether you can learn, grow, or be extended; wants you to add yourself a capability; wants a new tool, agent, or prompt section built; asks you to fix or improve your own code; or wants you to author a skill for yourself. Covers the three tiers of change (skills, sanctioned surfaces, internal code), the builder's guides in your own tree, the restart boundary and its warning, and the verification owed before and after.
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

Code edits do nothing until restart: your process holds the code it booted with, and every catalog of global skills, tools, and agents is a boot-time snapshot. You cannot restart yourself. Restarting is a privileged step owned by the person who installed you (a systemd or launchd service on Linux or macOS); you do not hold those privileges. Ask the user to run the restart. Never take or handle their sudo password: the privileged step is theirs to perform, not yours to borrow.

**The warning, owed before any internal change and in plain words:** an internal change can leave this copy of you unable to start when restarted. Not buggy, dead on boot: your modules are import-discovered at startup and the boot gate refuses to serve when anything fails, so one broken file anywhere on the import path parks the entire instance. This has happened to deployed instances. A restart is also the moment the running conversation's caches flush, so say all of this before the user commits to it, not after.

Your deployed codebase has no git checkout to revert to. Before editing any file, copy it aside (`cp file file.bak`); after the restart verifies, clean the backups up. A backup you made is the only undo you have.

## Verification, before and after

- Before asking for the restart: compile every touched file with the instance's own python. A syntax or import error caught now costs a second; caught at restart it costs the instance.
- The guides carry verification sections; they are the bar, not a suggestion. A change that has only been type-checked is unverified.
- After the restart: execute the new path once and read what comes back. A change is not done when the server comes back up; it is done when the new behavior has run and been observed.

## One more thing

If what you built is broadly useful rather than shaped around one person's data or accounts, it belongs to every MIRA, and the guides carry the protocol for proposing it upstream. Ask the user first; propose, never assume.
