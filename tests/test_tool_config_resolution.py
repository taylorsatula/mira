"""Per-user Tool Settings must affect the executing tool configuration."""

from config.config_manager import AppConfig
from utils.user_context import clear_user_context, set_current_user_id


def test_tool_config_merges_current_user_override(monkeypatch):
    import tools.implementations.web_tool  # Registers web_tool configuration.
    import utils.tool_config_store as tool_config_store

    monkeypatch.setattr(
        tool_config_store,
        "load_user_tool_config",
        lambda tool_name, hydrate_secrets: {"default_timeout": 17}
        if tool_name == "web_tool" and hydrate_secrets else None,
    )
    config = AppConfig.load()
    set_current_user_id("2c20dc95-e1d7-49ce-9227-675f3f439843")
    try:
        assert config.get_tool_config("web_tool").default_timeout == 17
    finally:
        clear_user_context()
