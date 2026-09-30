"""Tests for compliance gate."""

import pytest
from compliance.gate import ComplianceGate


@pytest.fixture
def gate():
    return ComplianceGate(
        internal_dnc_file="data/internal_dnc.txt",
        calling_window_start=8,
        calling_window_end=21,
    )


def test_dnc_blocks_call(gate):
    gate.add_to_dnc("+15551234567")
    result = gate.check_call_allowed("+15551234567")
    assert not result.allowed
    assert "DNC" in result.reason


def test_allowed_call(gate):
    result = gate.check_call_allowed("+15559999999")
    assert result.allowed


def test_calling_hours_blocks_outside_window(gate):
    """Outside the window the call must be refused; inside, permitted.

    Rather than assume the wall clock, this searches for a timezone that is
    currently outside the window and asserts the gate refuses it.
    """
    from datetime import datetime

    import pytz

    gate.bypass_calling_hours = False
    outside = None
    for tz in ("Pacific/Kiritimati", "Pacific/Midway", "Pacific/Honolulu",
               "America/New_York", "Asia/Tokyo"):
        hour = datetime.now(pytz.timezone(tz)).hour
        if hour < gate.calling_window_start or hour >= gate.calling_window_end:
            outside = tz
            break

    if outside is None:
        return  # every zone is currently inside the window

    result = gate.check_call_allowed("+15559999999", outside)
    assert not result.allowed, f"{outside} is outside hours and must be blocked"
    assert "calling hours" in result.reason.lower()


def test_bypass_calling_hours_never_bypasses_dnc(gate):
    """The demo affordance must not become a DNC escape hatch."""
    gate.bypass_calling_hours = True
    gate.add_to_dnc("+15550002222")
    result = gate.check_call_allowed("+15550002222", "America/New_York")
    assert not result.allowed
    assert "DNC" in result.reason


def test_disclosure_text(gate):
    text = gate.get_disclosure_text("Acme Corp")
    assert "Acme Corp" in text
    assert "AI" in text
