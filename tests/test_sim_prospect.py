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


def test_prospect_takes_the_line_without_observed_silence():
    """Regression: a forced 30s wait before every prospect line.

    The agent's streaming turn enqueued sentences directly and never took the
    turn lock, so the holder stayed None and mutual exclusion rested entirely
    on the observed-silence counter. But a waiter is by definition producing
    no audio, and silence is only credited from audio — so the prospect was
    asked to clear a gate only it could open, and could never clear it. It sat
    out the full lock timeout and was forced on all seven turns of a real call.

    The silence gate exists to stop the *agent* cutting in on a prospect. For
    the prospect the holder is the authority: the agent does not release until
    its audio has finished playing.
    """
    import asyncio
    import time

    from turnlock import Speaker, TurnLock

    async def go():
        lock = TurnLock(silence_required=0.35, timeout=5.0)
        prospect = SimulatedProspect(
            script_for("interested"), tts=None, turn_gap=0.0, lock=lock
        )
        # A streamed agent turn: synthesis resets observed silence, and no
        # acquire/release ever happened, so the line reads as free.
        lock.note_speech()
        assert lock.is_free()

        start = time.monotonic()
        await lock.acquire(Speaker.PROSPECT, require_silence=False)
        elapsed = time.monotonic() - start
        lock.release(Speaker.PROSPECT)
        return elapsed, lock.overlaps, prospect

    elapsed, overlaps, prospect = asyncio.run(go())
    assert elapsed < 0.2, f"prospect waited {elapsed:.2f}s for a free line"
    assert not any("timeout" in o for o in overlaps), overlaps
