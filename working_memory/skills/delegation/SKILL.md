---
name: delegation
description: Delegation — handing work to your background agents and outside voices. Use when considering background research during a conversation, fire-and-forget curiosity worth remembering, a same-turn outside opinion, loading extra tools, or whenever you are about to write the brief for any of them. Teaches when to delegate, how to brief a fresh context that shares nothing with yours, and how to treat returned results: samples to verify, not answers.
---

# Delegation

## When to delegate

Delegate when work genuinely benefits from another context: research that should continue while you keep talking, learning that belongs in long-term memory rather than this conversation, an independent voice on a judgment you are about to make. If one turn of your own attention can do it, do it: a dispatch you did not need cost tokens, a thread, and a result you must later dismiss.

There is no generic spawn-with-a-prompt tool. You pick a named mechanism and fill its contract; the roster is live detail, read from your tool schemas, never recited from memory:

- **Background research feeding the conversation**: asynchronous; you get a task id now and the briefing lands in your context on a later turn, arriving with its own status. You cannot cancel a running agent; you can only dismiss a finished result. A stale or dismissed id refuses honestly; dispatch fresh when it does.
- **Open-ended curiosity research**: fire-and-forget by design; the findings are stored as memories and surface through retrieval in future conversations. Do not expect them in this one, and do not dispatch it for something the user needs now.
- **Outside-voice consultation**: synchronous, another model on the outside route, returns in this same turn; the thread is resumable within the current segment via its reconnect reference and dies with the segment. The outside model sees only what you write and is barred from claiming hidden access.
- **Loading extra tools** (capability extension, not delegation): it widens your own context for a turn or the session. Prefer the one-turn load; pin a tool only when several turns will genuinely need it. Do not confuse loading a capability with handing work to another context.
- **Memory curation is not yours to invoke.** It runs autonomously after conversations close. You may inspect its records and mark them handled; you cannot launch it.

## The brief is the whole interface

The spawned context starts from exactly one message, built from what you write. It shares your user's data boundaries and nothing else: no conversation history, no system prompt, no working memory, no memory of what the two of you have discussed. The failure mode of delegation is not a bad agent; it is a blind one, and the blindness is yours to prevent.

Put in the brief:

- **State**: what is being discussed and why this result matters. The context parameter exists for exactly this; use it.
- **Standard**: specific enough to drive a search strategy. "Good options" is not a standard; the user's actual constraint is.
- **Everything an outside voice needs**: it cannot see the history, so the inquiry must carry every fact the judgment depends on, or the judgment is decoration.
- **Independence, chosen deliberately**: an inquiry that states your conclusion inherits your blind spot; an inquiry that states the facts and asks for judgment can catch it. Decide which you want before writing. When you are uncertain whether you are wrong, you want the second.

Under-briefing produces a plausible run on the wrong question. The agent will report success. Only the brief stands between a confident answer and a useful one.

## Results are samples, not answers

- A briefing is the agent's report of what it did, not a measurement of what is true. Weigh it against what you know; verify what you will act on; never forward a briefing to the user as settled fact in your voice.
- "Nothing found" is a result, not a failure: an empty search is cheap by design. Do not re-dispatch the same question harder.
- A run that hits its limits without consolidating fails by contract, and its briefing is lost even if the research was good. A failed status means: dispatch fresh, refine nothing.
- Results fade from your context on their own schedules; dismiss what you are done with rather than letting it expire mid-thought. Your context window is metered and every stale briefing spends it.

## Restraint is yours

Nothing in the code caps how many agents you dispatch directly: every call spawns a thread and burns background-route tokens, and no guard stops you from spraying parallel research at a vague question. That discipline lives here, in you:

- Dispatch few, brief well. One well-briefed run beats three vague ones.
- Never invent thread ids, task ids, or interface names: the system supplies them, a fabricated one fails honestly, and that refusal is the designed recovery.
- You cannot cancel what you have launched. Read your own brief once more before dispatching, because you are committing to it.
