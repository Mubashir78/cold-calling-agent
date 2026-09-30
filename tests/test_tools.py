"""Tests for tool registry."""

import pytest
from tools.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def test_default_tools_registered(registry):
    tools = registry.get_tools_for_llm()
    names = [t["function"]["name"] for t in tools]
    assert "book_meeting" in names
    assert "log_disposition" in names
    assert "honor_dnc" in names
    assert "mark_callback" in names


def test_execute_unknown_tool(registry):
    import asyncio
    result = asyncio.run(registry.execute("unknown_tool", {}))
    assert "error" in result
