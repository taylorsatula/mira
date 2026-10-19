"""Package marker for the sidebar-agent runtime.

Nothing is re-exported here. `agents.base` pulls config and the LLM client,
and every agent implementation pulls its tool modules, so importing this
package has to stay free of that chain — consumers import the submodule they
need (`agents.base`, `agents.sidebar`, `agents.spawn`, `agents.specs`).
"""
