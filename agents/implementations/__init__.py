"""
Package marker for the concrete sidebar agents; it re-exports nothing.

There is no agent registry in this package and nothing here imports an agent
class. Each type is addressed by module path -- `agents.specs.SUBAGENT_SPECS`
holds `"module:ClassName"` strings and imports lazily at spawn -- so an import
added here would pull every agent's tool module (and the heavy clients behind
them) into any consumer that touches the package at all.
"""
