# Segment System

MIRA keeps one continuous conversation while storing durable boundaries as segment sentinels. A segment is the unit that can collapse into a first-person summary, feed long-term memory extraction, and keep older conversation history searchable without leaving every message live in the provider request.

## Active Segment

New messages append to the current active segment. The active segment remains expanded while the conversation is ongoing.

A segment can be paused — by the idle-timeout service or an explicit pause — and later resumed by the next message or an explicit resume. A paused segment is still the current segment: at most one live sentinel (active or paused) exists per continuum, enforced by a unique partial index, and a save that lands while the segment sits paused reuses that sentinel rather than minting a second one. Pausing and resuming are refused while a turn is running.

If the active history becomes too large before natural collapse, live context compaction replaces older provider-request messages with a rolling continuation brief. The original PostgreSQL message history remains intact.

## Segment Collapse

When a segment has been idle long enough, `SegmentCollapseHandler` generates:

- First-person memory trace
- Two-sentence precis
- Display title
- Complexity score

The collapsed segment is represented by a sentinel message. The sentinel carries the summary fields and the `segment_embedding` used for segment-level retrieval.

## Embeddings

`messages.segment_embedding` stores a document embedding from the install's embedding model (vector length fixed at install by `embedding_config`) for collapsed segment sentinels. The embedding supports semantic search over collapsed conversation history without loading full message bodies into context.

## Boundaries

Segment collapse is durable history management. Live context compaction is request shaping for an active conversation. They are separate mechanisms and should not be merged.
