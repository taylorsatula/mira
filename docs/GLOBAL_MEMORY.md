# Global memories

MIRA keeps two kinds of long-term memory.

- **Local memories** belong to one account on one install: coding preferences, a day at the beach, the state of a project, opinions, habits. Almost everything MIRA remembers is local.
- **Global memories** are fixed facts about a person that stay true whatever the conversation is about: someone's birthday, who the people in the user's life are, where they live, an allergy. Every MIRA instance that serves the same person should know them.

When MIRA turns a finished conversation into memories, it marks each one local or global. It marks a memory global only when it is clearly a fixed fact about a person, and picks local when unsure: a local memory still reaches its user, while a wrong global memory reaches every instance that shares the store. A memory with an expiry date is always local.

On a multi-user install every memory MIRA extracts stays local, and remote mode is refused at startup. Every account on one install reads the same global memories, so a global tag, or a remote store full of one person's facts, would show that person's birthday to all of them.

## Where global memories are kept

**Local mode (the default).** Global memories live in the `global_memories` table of this install's own database. There is nothing to set up.

**Remote mode.** A separate Postgres database holds the global memories, and any number of MIRA instances share it. Each instance copies the whole remote set into its own `global_memories` table when it starts, and writes each new global memory to the remote as soon as it is extracted, then to its own copy. Another instance sees that memory at its next restart.

The remote database stores text, not search vectors. Each instance computes vectors for new or changed text with its own embedding model at startup, so instances with different embedding models can share one store. A large set of new remote rows makes that startup slower.

## Before you turn on remote mode

In remote mode the remote database is the only authority. At every startup MIRA deletes any local global memory that the remote does not hold: rows you added by hand, and global memories MIRA extracted while it ran in local mode. If your local `global_memories` table has rows you want to keep, copy them to the remote first:

```
psql "postgresql://mira_admin@localhost:5432/mira_service" -c "\copy (SELECT id, text, importance_score, happens_at, created_at, is_archived, archived_at FROM global_memories) TO 'global_memories.csv' CSV"
psql "postgresql://OWNER@REMOTEHOST:5432/DBNAME" -c "\copy global_memories (id, text, importance_score, happens_at, created_at, is_archived, archived_at) FROM 'global_memories.csv' CSV"
```

Run the second command after step 2 below. The startup log reports how many local rows each pull removed. The same applies if you ever remove the Vault field and add it back later: global memories extracted in between exist only locally.

## Turning on remote mode

1. Create a database on a Postgres server that every instance can reach. The pgvector extension is not needed there.
2. Create the table by running the shipped file against that database as its owner:
   `psql "postgresql://OWNER@REMOTEHOST:5432/DBNAME" -f deploy/global_memory_remote_schema.sql`
3. Create a login role for MIRA and give it `SELECT` and `INSERT` on `global_memories`. It needs nothing else.
4. Store the address in Vault on each instance, then restart MIRA:
   `vault kv patch secret/mira/services global_memory_url='postgresql://ROLE:PASSWORD@REMOTEHOST:5432/DBNAME'`
   Percent-encode any reserved character in the password (`@` becomes `%40`, `/` becomes `%2F`). When the database is across a network you do not control, add TLS settings to the address, for example `?sslmode=verify-full&sslrootcert=/path/to/ca.pem`; MIRA passes the address's settings to the connection, except the connection timeout, keepalive, `tcp_user_timeout`, and `options` settings, which it sets itself.

That Vault field is the whole switch: present means remote mode, absent means local mode. Vault keeps it across updates and container recreates. The startup log says which mode MIRA chose. If the address is not a `postgresql://` address with a host, or the install is multi-user, MIRA refuses to start and says which. The startup self-test also checks that the remote table exists and that the role has `SELECT` and `INSERT`; that check only warns, so an outage there never blocks startup.

To go back to local mode, remove the field and restart. `vault kv patch` only adds or changes fields, so rewrite the secret instead: list its fields with `vault kv get secret/mira/services`, then `vault kv put secret/mira/services` with every one of them except `global_memory_url`. `vault kv put` replaces the whole secret, so a field left off that command is deleted too, and MIRA will not start without its required ones. The local copy of the global memories stays as it was.

## Day-to-day behavior

**Removing a global memory.** Delete its row on the remote, or set `is_archived = TRUE` there. Each instance drops it at its next startup.

**Importance.** Global memories added by extraction are stored at importance 0.5, and rows you insert by hand default to 1.0. MIRA never changes a global memory's importance on its own, so raise or lower it on the remote.

**Entity lookup.** Global memories are not linked to people, places, or product names, so they come up through text and meaning search only, not through a lookup by name.

**The remote is down at startup.** MIRA starts anyway, logs an error, and serves the global memories from its last successful startup. The same happens if the connection drops partway through. At connection time Postgres reports a down server, a wrong password, and a wrong database name the same way, so if new global memories never appear, read that log line. If MIRA connects but the table is missing or the role lacks permission, startup stops with an error.

**The remote is down during extraction.** The finished conversation's memories wait. MIRA retries that conversation on its regular extraction sweep until the remote answers, nothing is lost, and the retries do not count toward the limit that makes MIRA give up on a conversation.

**Duplicates and corrections.** MIRA checks a new global memory only against its own copy as of its last startup. Two instances told the same fact can each add it, and a correction ("her birthday is actually June 3") is added as a new global memory beside the old one rather than replacing it. MIRA's memory tools never merge, edit, or archive global memories, so clean these up with SQL on the remote; every instance picks up the result at its next startup.
