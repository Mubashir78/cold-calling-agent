"""Guards on the tool layer.

The number dialed and the validity of a booking time are things the model
cannot know reliably. These tests pin the behaviour that stops it guessing.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tools.registry import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry(
        crm_webhook="http://127.0.0.1:9/api/crm",
        calendar_webhook="http://127.0.0.1:9/api/calendar",
        dnc_webhook="http://127.0.0.1:9/api/dnc",
    )
    reg.start_call()
    reg.bind_call("+14155550123", "Priya Raman", "Cobalt Systems")
    return reg


def test_no_tool_takes_a_phone_number():
    """The model must not be able to supply a number; it does not know it."""
    for tool in registry_tools():
        assert "phone_number" not in tool["function"]["parameters"].get(
            "properties", {}
        ), f"{tool['function']['name']} must not accept a phone number"


def test_no_tool_asks_for_identity_the_system_already_knows():
    """Regression: book_meeting required prospect_name, and gpt-oss models
    declined to call the tool rather than invent a name. Identity is injected
    by bind_call, so the model is only asked for what it can actually know."""
    for tool in registry_tools():
        props = tool["function"]["parameters"].get("properties", {})
        assert "prospect_name" not in props, (
            f"{tool['function']['name']} must not ask for a name"
        )
        assert "company" not in props, (
            f"{tool['function']['name']} must not ask for a company"
        )


def test_booking_only_requires_a_time():
    """The model should need to supply exactly one thing: when."""
    for tool in registry_tools():
        if tool["function"]["name"] != "book_meeting":
            continue
        params = tool["function"]["parameters"]
        assert params["required"] == ["time_slot"], params["required"]
        assert set(params["properties"]) == {"time_slot"}


def registry_tools():
    reg = ToolRegistry()
    return reg.get_tools_for_llm()


def test_past_datetime_is_rejected():
    past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with pytest.raises(ValueError, match="in the past"):
        ToolRegistry._validate_future_datetime(past, "time_slot")


def test_malformed_datetime_is_rejected():
    with pytest.raises(ValueError, match="ISO 8601"):
        ToolRegistry._validate_future_datetime("next tuesday", "time_slot")


def test_future_datetime_is_accepted():
    future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    assert ToolRegistry._validate_future_datetime(future, "time_slot") == future


async def test_dnc_refuses_when_no_call_bound():
    """A bare 'unknown' in the DNC list suppresses nobody."""
    reg = ToolRegistry(dnc_webhook="http://127.0.0.1:9/api/dnc")
    reg.start_call()
    result = await reg.execute("honor_dnc", {})
    assert result["success"] is False
    assert "placeholder" in result["error"]


async def test_duplicate_booking_is_refused(registry):
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    args = {"time_slot": future}

    # Force the first booking to "succeed" so the second is a real duplicate.
    registry._booked_this_call = True
    result = await registry.execute("book_meeting", args)
    assert result["success"] is False
    assert "already booked" in result["error"].lower()


async def test_booking_injects_the_real_prospect_identity(registry):
    """The system knows who it dialed; the model must not have to guess."""
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return {"status": "ok"}

    registry._tools["book_meeting"].handler = capture
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()

    result = await registry.execute("book_meeting", {"time_slot": future})

    assert result["success"] is True, result
    assert seen["phone_number"] == "+14155550123"
    assert seen["prospect_name"] == "Priya Raman"
    assert seen["company"] == "Cobalt Systems"
    assert seen["duration_minutes"] == 30


class _StubResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubClient:
    """Captures the JSON body actually sent to the calendar."""

    def __init__(self):
        self.posts = []

    async def post(self, url, json=None):
        self.posts.append({"url": url, "json": json})
        return _StubResponse({"status": "ok", "event_id": "evt_1"})

    async def aclose(self):
        return None


@pytest.mark.parametrize("supplied", [
    {"time_slot": "PLACEHOLDER"},
    # The model sometimes volunteers identity anyway; it must not break us.
    {"time_slot": "PLACEHOLDER", "prospect_name": "Invented Name"},
])
async def test_real_booking_handler_sends_a_complete_payload(supplied):
    """Regression: the handler raised NameError on 'prospect_name'.

    The earlier test stubbed the handler itself, so the real body never ran.
    This exercises it end to end with only the HTTP client replaced.
    """
    reg = ToolRegistry()
    reg._client = _StubClient()
    reg.start_call()
    reg.bind_call("+14155550101", "Dana Whitfield", "Northwind Logistics")

    args = dict(supplied)
    args["time_slot"] = (
        datetime.now(timezone.utc) + timedelta(days=3)
    ).isoformat()

    result = await reg.execute("book_meeting", args)

    assert result["success"] is True, f"booking failed: {result}"
    assert reg._client.posts, "nothing was posted to the calendar"
    body = reg._client.posts[0]["json"]
    assert body["prospect_name"] == "Dana Whitfield"
    assert body["phone_number"] == "+14155550101"
    assert body["company"] == "Northwind Logistics"
    assert body["time_slot"]


async def test_duplicate_dnc_is_refused(registry):
    registry._dnc_this_call = True
    result = await registry.execute("honor_dnc", {})
    assert result["success"] is False
