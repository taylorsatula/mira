# MIRA v2026.11.01-2.2.0

This is a feature release. Its defining change is a new web interface for
the browser client. It also carries one repair to the memory-surfacing
pass. There are no database steps and no dependency changes, so an update
has nothing to consent to and no virtualenv to rebuild.

## A new web interface

The browser client at `/chat` is rebuilt on the paper-grain design
language: cream paper, warm brown hairlines, a faint grain over the whole
window, and a dusk palette for low light. Every page keeps its existing
controller, every backend call is unchanged, and every module keeps its
contract — including the three behaviours the browser shares with the
terminal client (the reply display filter, the retry and reconnect
cadence, and the history row grouping). A reskin that moved any of those
would have drifted the two clients apart silently.

**One window, one conversation.** The chat page is no longer a stack of
cards. A notice band sits at the top, the conversation scrolls in the
middle, and the composer is docked to the foot of the window, spanning
its full width. The conversation text is capped at a reading measure of
about 72 characters and centred, so long replies stay comfortable to read
while the composer keeps the whole width. Only the window carries a
border, radius and shadow now; the regions inside it do not.

**Turns render inside the conversation.** This part changes behaviour, not
just paint. A turn used to stream into a region beside the history and be
archived into it when it finished. A turn now opens a new exchange at the
foot of the same thread and streams into that exchange's reply bubble.
Nothing moves when the turn ends, because the live reply already is a
history row. Two consequences follow: the sidebar's History button and
its `?history=1` deep link are gone — the thread is the history, and it
pages itself as you scroll up — and a turn that produces nothing leaves no
empty bubble behind, since the server holds no such exchange.

**An idle prompt.** An empty conversation now offers "Where shall we
begin?" with three suggestions: "What can you help me with?", "Show me my
recent memories", and "I'm not sure where to start…". Clicking one fills
the composer and sends it. The prompt is shown by the stylesheet keyed on
the welcome message, so it disappears the moment a reply arrives.

**Accessibility and script policy.** All four pages gained a skip link
and `viewport-fit=cover` for notched screens, and lost `user-scalable=no`.
The chat page now contains no inline event handler at all: the
announcement dismiss and the "Manage Domains" click are listeners in
`core.js` and `domain-knowledge.js`. That is real progress toward a strict
`script-src 'self'`, which is still blocked by the three supporting pages'
inline controllers and by each page's pre-paint theme script.

**Dark mode.** The palette has two blocks: paper is the default and dusk is
the `data-theme="dark"` override. The Appearance setting, its storage key
and its runtime switch are unchanged. The one visible difference is that a
page whose theme script never runs now renders paper rather than dark.

## Subcortical reports through a tool call

The subcortical pass is the small retrieval prep that runs before MIRA
answers: it rewrites fragmentary references into search phrases, extracts
named entities, and decides which old memories to keep in context.

It used to return XML that the caller searched for, and that search was
the failure. The response passed through a JSON repair before a regular
expression looked for the tag, and on short replies the expression could
match the prompt's own format example instead of the answer. A parse
failure then raised out of the turn and aborted it.

The pass now returns exactly one `record_subcortical_analysis` tool call,
and that tool's JSON schema is generated from the same object the result is
parsed into. The format the model is shown and the format the parser
accepts are one artifact, so they cannot drift. Retention IDs are
constrained at that schema boundary to the IDs the call actually supplied,
so a fabricated ID arrives as an invalid tool call and never as data.

Malformed output now degrades rather than raising: the memories already in
context stay, nothing fresh is surfaced, and a warning records the
user-visible consequence. Provider, network and context-overflow errors
are not that shape and still propagate, so an outage cannot pass as a quiet
turn.

Two things left the subcortical result: the complexity classification and
the thinking-effort level derived from it. Primary thinking effort now
comes only from the primary route's configured effort and the existing
explicit override. The fast route's database row is unchanged, and
pointing it at a smaller local model is a separate database operation.

## Also in this release

- The README's description of the subcortical layer no longer claims it
  classifies a turn as straightforward or reasoning-heavy, which it no
  longer does.
- The orientation maps for the browser client, the subcortical service and
  the prompt templates were brought in line with the code above in the same
  commits.

## For upgraders

`mira update`, as always. Nothing to consent to and no dependency changes
to rebuild around. The new interface appears on the next page load.

## Verification

_Pending — gate results are recorded here after the asset is built._
