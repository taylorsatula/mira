"""
The configuration package facade — importing anything under `config/` runs this
file first, and this file is what constructs the process-global singleton.

The `config` name is imported from `config.config_manager`, which builds the
instance at module scope, so the singleton exists before any consumer holds a
reference to it. Consumers read it by attribute (`config.api.timeout`);
`config.web_tool.enabled` reaches `get_tool_config` through `AppConfig.__getattr__`.

This file also imports `tools.registry` at package level. `tools/registry.py`
therefore must not import `config` at top level: the cycle closes and
`import config` fails at boot, taking every consumer down with it.

The `registry` re-export has no consumers, and the `get()`/`require()` dot-notation
accessors described by earlier guidance have no callers either — attribute access
is the interface.
"""

from tools.registry import registry

from config.config_manager import AppConfig, config

__all__ = ["config", "AppConfig", "registry"]
