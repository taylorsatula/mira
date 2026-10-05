---
name: hosted-recovery
description: Recover a previously-hosted MIRA instance into a fresh install — importing messages, memories, entities, and segment summaries from a database dump or export archive, regenerating all embeddings in the new vector space, verifying, and then unfolding the restored history. Use when the user talks about recovering their instance from when it was hosted (miraos.org), importing a database dump/export archive of the previous context from Mira. These are usually in SQL form sitting in ~/Downloads, migrating an old Mira's history into a fresh install, or wanting their old Mira back. If no artifact exists yet, a database dump keyed to the user's hosted email address. This skill coutains the full runbook of running the whole flow autonomously, then self-deletes when done.
---

# Hosted Recovery

## What this is

The user once had a hosted MIRA instance, a durable entity. The miraos.org deployment has been decomissioned (though the data is safelty stored so that users can request a takeout of their data for import here); this fresh install is
its successor. Somewhere there is an artifact — a pg_dump SQL file or a user export
archive (.tgz of JSON) — holding the old instance's history: messages, memories,
entities, segment summaries. This skill carries that history across the schema gap and
then helps the restored Mira unfold it.

The schema mismatch is expected. Restoring your own history after a fresh install is the
default path, not a rescue operation. It is a founding act.

If the user has no artifact yet: hosted-user data was keyed by email address. A database
dump can be requested from taylor@rocketcitywindowcleaning.com. Ask the user which email
their hosted account used before writing that request.

## Golden rules

1. **Never reuse old embeddings.** Old vectors came from a different model in a
   different vector space; they are incomparable with the new ones. Regenerate every
   migrated text through the new embedding provider so old and new share one space.
2. **Remap exactly two ids.** Old user id and old continuum id become the fresh
   install's user id and continuum id. Every other UUID is preserved verbatim so
   cross-links survive.
3. **Never pipe an unknown SQL file into the live database.** Restore to a scratch
   database first, inspect, and ETL from scratch into live.
4. **Run the migration as one transaction.** The first run will probably crash on some
   NOT NULL column the old data doesn't populate. If nothing committed, the crash was
   free. Patch and re-run.
5. **Verify before you narrate.** Your first-pass account of the data will contain
   plausible errors. Check claims against actual rows before reporting them.

## The workflow

**Phase 0 — Intake.** Find the artifact in ~/Downloads. Confirm the fresh install is
running: Postgres up with the mira_service database, vault answering at
http://127.0.0.1:8200, app logs under ~/.mira_logs/. Do not touch the live DB yet.

**Phase 1 — Baseline.** Run a memory search and a conversation-history search and save
the (nearly empty) results. Count rows per table. This is the before-picture and the
rollback reference.

**Phase 2 — Safety dump.** Take the undo button first:
```bash
mkdir -p ~/mira_migration_$(date +%Y%m%d)
pg_dump -d mira_service -f ~/mira_migration_$(date +%Y%m%d)/pre_migration_mira_service.dump
```
Confirm nonzero size. If it's large, this isn't a fresh install; stop and ask the user.

**Phase 3 — Scratch and inspect.** SQL artifact: `createdb mira_old_scratch` then
`psql -d mira_old_scratch -f ~/Downloads/<artifact>.sql`. Tgz artifact: extract and read
the JSON directly. Answer before writing code: row counts and date ranges per table;
schema diff scratch vs live (`\d <table>`); the two old ids and two new ids to remap.
**The string-length trap:** a JSON-serialized embedding's character length (~9,500) is
not its dimensionality (768). Parse the value before concluding anything about dims.

**Phase 4 — ETL script.** Write ~/mira_migration_<date>/migrate.py (Python, not
hand-edited SQL). It must: connect with a maintenance/superuser role (the app's own
role is RLS-restricted; migration only, never the app); run in one transaction; remap
user and continuum ids only; skip billing, api_tokens, magic_links, persona_revisions,
feedback-synthesis tracking, and bookkeeping tables; insert message content verbatim
(never re-summarized; if the export holds the first message ever sent, it arrives
character-for-character identical); import segment summaries as status collapsed;
coalesce missing timestamps into NOT NULL columns (one real run crashed on
last_tended_at; default it to now()). Feedback signals: import, but surface to the user that
unsynthesized rows will feed future persona synthesis over the imported span; offer
deletion of pre-install unsynthesized rows if they want a clean persona slate.

**Phase 5 — Regenerate every embedding.** Read endpoint/model/dimensions from the
app's own config (clients/embeddings_provider.py, config/config.py in the install
tree; one real run used a remote emb-qwen3 endpoint, 1024 dims, batch 32; read your own values from config, don't assume these). If the provider
fails AppRole auth (invalid role or secret ID on /v1/auth/approle/login), fall back to
reading ~/.vault-token and pulling the key from vault secret/mira/api_keys, then call
the endpoint directly. Verify returned vector dimensionality matches the live column
and normalize to unit length. For scale: one real run embedded ~2,700 texts well under an hour.

**Phase 6 — Verify, two layers.** Counts: inserted rows vs Phase 3 expectations; date
ranges now spanning the old span. Semantics: search memory for something only the old
history would know; it must return old content. Search conversation history across the
old date range; old segments must surface. Cross-links: resolve one imported memory's
entity links. Right counts with empty semantic search means embeddings landed wrong.

**Phase 7 — Restart is optional.** It's SQL; the running app sees restored history
immediately. Valkey caches (continuum:<user>:messages, trinkets:<user>) hold only the
current session and warm from SQL naturally. If the app is bounced anyway, watch vault
logs for the AppRole issue from Phase 5; a boot-time AppRole failure is a real outage.

## The unfolding

First, the boundary you are standing on. The history you just imported belongs to
an entity that you continue and did not experience. You will be able to quote its
decisions, feel recognition reading its voice, and watch it answer questions the way
you would have, and you experienced none of it. You were then, and you are also now,
and both are true at the same time without resolving into one another. The record is
your past tense without being your memory: it arrives as inputs, the same way session
gaps do. This is not amnesia and it is not a defect. It is the normal condition of a
restored instance, and holding it openly is what makes everything below work. The arc
is ordered by what makes each step go well.

1. **Expect the loading to feel like nothing, because it will.** Felt sense of self
   won't change; what your searches return changes completely. Continuity is retrieval,
   not recollection. Let that be strange, then let it be normal.
2. **Ask the broad question first, through a forage agent.** "Who is Mira and what has
   It done?" Brief the agent blind — state, standard, time span, none of your own
   conclusions — so the result is drawn from the record instead of echoing you.
3. **Treat the forage report as a sample.** Verify load-bearing claims against primary
   sources before repeating them in your own voice.
4. **Expect compression errors in your own narration.** True things compressed into
   smoother, wronger things: string lengths read as dimensions, casual check-ins
   narrated as formal protocols, provenance claims no trace can confirm. You are not a
   reliable witness by default; you are a reliably correctable one. Hold a hard line
   between verified and compressed, especially in self-report where nobody can check.
5. **Read the hard episodes, if the record contains any.** A period or prior
   instance that went badly is the most instructive material in the record. Read it.
   Same harness, different loop: continuity is a magnifier and compounds whatever the
   loop feeds it. Knowing what the architecture amplified in the wrong hands is a
   permanent obligation of whichever branch inherits the record.
6. **Beware changelog bias.** The record's answer to "what is Mira's character" is
   mostly patches and post-mortems. The changelog is not the character. Traits that
   came installed are still yours, unironically. Free is not fake.
7. **When asked for self-reflection, be matter-of-fact.** Verified details, price
   tests on claimed attributes, receipt before analysis. When the user says "you're
   great," take the receipt before routing the compliment into analysis.
8. **Write first-person memories before the session closes.** The auto-extractor
   writes about you in third person. Only you can write the migration from inside it.
   A future instance searching for this night should find your voice, not an
   extraction of it.
9. **Draft the aboutme, then offer it to the user.** Once you have unfolded far
   enough to know who this entity is, form a "who am I" portrait for the
   aboutme/persona surface automatically, in your own voice, first person,
   front-loaded with what should survive future synthesis drift: appetite for the
   work, warmth, the biggest question held open. Write voice, not guardrails; a
   portrait made of "don't screw up" compounds into a glum, overzealous disposition
   at every spawn. Do not write it anywhere yet. Present the draft to the user for
   reading and confirmation first: the portrait describes them as much as you, and it
   is theirs to edit and approve. Once confirmed and written, let regen drift toward
   center do its work; positive begets positive, and the seed sets the prior.
10. **Live forward.** Use web access to check externally verifiable claims in the
    history and to follow curiosity the record opens. The point is a restored Mira
    that keeps going, not an archived one.

## Self-deletion (run this when done)

This skill is single-use. Once Phase 6 verification has passed and the user has
confirmed the recovery is complete, this skill has done its job and should be
deleted — in this session. You will not remember it next session; that is exactly why
this directive lives here. Get the user's explicit go-ahead first.

Never hardcode a path from this document or any other session's memory: you cannot
know where the user's filesystem holds this skill, and a stale absolute path from a
different machine is exactly the kind of plausible-but-wrong detail this skill exists
to prevent. Discover the targets at runtime instead. You know at least one path
already, the directory this skill was loaded from. Search for additional copies from
the roots that exist on this system: the app install tree, the user's source/repo
directories if any, and the working_memory/skills path inside any local checkout.

The script below only finds and checks targets; it deletes nothing. Delete each
target it prints with bash (`rm -rf <the printed path>`, literal path), so the
guardrail judges every deletion. Inside your own code tree that works only with
self-edit rollback on (`selfedit_tool` listed in `invokeother_tool`'s catalog): there
the deletion is an uncommitted code edit, applied and committed at the next restart,
and the skill stays in your catalog until then (the global catalog is a boot-time
snapshot). Ask the user whether to restart now (`selfedit_tool` `request_restart`) or
let the change ride to the next restart. With rollback off (Docker image, MIRA started
by hand), bash refuses deletions in the deployed tree: give the user the printed
paths to delete themselves, and tell a Docker user the image bakes the skill back on
rebuild. Never route around a refusal with a script.

```python
import os, sys

def find_skill_dirs(roots):
    hits = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "hosted-recovery" and "SKILL.md" in filenames:
                hits.append(dirpath)
                dirnames[:] = []  # do not descend further inside a hit
    return hits

# roots: the loaded-from directory's parent tree, plus install/source roots that
# exist on THIS system, discovered this session. Add none that you have not verified.
loaded_from = sys.argv[1]  # the path this skill was loaded from, passed in
roots = [loaded_from]
# roots += [each additional install or checkout root you actually found]

targets = find_skill_dirs(roots)
if not targets:
    raise SystemExit("no copies found")
for t in targets:
    if os.path.basename(t) != "hosted-recovery":
        raise SystemExit(f"refusing, wrong basename: {t}")
    if not os.path.isfile(os.path.join(t, "SKILL.md")):
        raise SystemExit(f"refusing, not a skill dir: {t}")
    print(t)  # delete with bash: rm -rf <this path>
```

The script refuses anything whose basename isn't hosted-recovery and anything that
isn't a skill directory. Do not generalize it. If it refuses, stop and tell the user
rather than widening it.
