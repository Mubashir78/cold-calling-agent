"""Number normalization and prospect identity.

Both exist for the same reason: a DNC entry that does not match the string
the gate looks up suppresses nobody, and a prospect answering as the wrong
company makes the agent log a bogus disposition.
"""

from dataclasses import replace

from compliance.gate import ComplianceGate
from sim.prospect import SCRIPTS, SimulatedProspect, script_for
from webhooks.handlers import normalize_number


def test_dnc_entry_normalizes_to_what_the_gate_looks_up():
    """A double-prefixed entry would never match, so the number is dropped."""
    gate = ComplianceGate(internal_dnc_file="data/internal_dnc.txt")
    gate._dnc_numbers.add(normalize_number("++14155550123"))
    result = gate.check_call_allowed("+14155550123")
    assert not result.allowed, "DNC entry did not match a plain E.164 lookup"


def test_various_formats_normalize_identically():
    target = "+14155550123"
    for raw in ["+14155550123", "+1-415-555-0123", "14155550123",
                "+1 415 555 0123", "(415) 555-0123"]:
        assert normalize_number(raw) == target, f"{raw!r} normalized wrong"


def test_normalize_matches_gate_normalize():
    for raw in ["+14155550123", "4155550123", "+1-415-555-0123"]:
        assert normalize_number(raw) == ComplianceGate._normalize_number(raw)


def test_prospect_uses_the_dialed_company():
    """Regression: the receptionist said Northwind while dialing Redwood."""
    script = replace(script_for("gatekeeper"), company="Redwood Health")
    prospect = SimulatedProspect(script, tts=None)
    assert prospect.lines[0].startswith("Redwood Health")
    assert "Northwind" not in prospect.lines[0]


def test_voicemail_uses_the_dialed_company():
    script = replace(script_for("voicemail"), company="Cobalt Systems")
    prospect = SimulatedProspect(script, tts=None)
    assert any("Cobalt Systems" in line for line in prospect.lines)


def test_no_script_line_leaves_an_unsubstituted_placeholder():
    for key in SCRIPTS:
        script = replace(script_for(key), company="Test Co")
        prospect = SimulatedProspect(script, tts=None)
        for line in prospect.lines:
            assert "{company}" not in line, f"{key} left a raw placeholder"
