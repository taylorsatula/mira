"""Segment-lifecycle-bound polling services.

`segment_poller` holds the base class that owns per-user thread lifecycle;
`inbox_poller` is its one concrete subclass. Nothing is re-exported and nothing
self-registers: a poller runs only because `CNSIntegrationFactory` constructs an
instance alongside the trinket it publishes to, so a subclass defined here but
never constructed there is dead code that fails silently at boot.
"""
