"""Prospect pacing: the agent must get a turn.

The prospect must not guess how long the agent's reply takes. It waits for
`TurnCoordinator` to report the line free, because guessing failed in both
directions — it cut the agent off when the guess ran long, and it sat out
fixed dead air when the reply was short.

These tests exercise the wait logic through the real coordinator rather than a
stand-in, because the bugs here were all about the seam between the two.
"""

import asyncio
import time

from sim.prospect import (
    AGENT_TURN_START_WAIT,
    MIN_TURN_GAP,
    SCRIPTS,
    SimulatedProspect,
    script_for,
)
from turns import TurnCoordinator


def _prospect(key="interested", **kw) -> SimulatedProspect:
    return SimulatedProspect(script_for(key), tts=None, turn_gap=0.0,
                             turns=TurnCoordinator(), **kw)


def test_gap_is_a_pause_not_the_agents_speaking_time():
    """A pause is short; a guessed turn length is what broke this.

    Human turn-taking leaves well under a second of dead air. The old 3.5-6.5s
    values were turn lengths, so using them as pauses still talked over slow
    replies.
    """
    for key in SCRIPTS:
        p = _prospect(key)
        gaps = [p._gap_for(i) for i in range(30)]
        assert all(MIN_TURN_GAP <= g <= 2.5 for g in gaps), (
            f"{key} pause too long: {max(gaps):.2f}s"
        )


def test_prospect_waits_for_the_turn_to_finish():
    """The agent holds the line; the prospect stays quiet."""
    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(script_for("interested"), tts=None,
                              turn_gap=0.0, turns=turns)
        turns.agent_claims()
        released = asyncio.Event()

        async def release_later():
            await asyncio.sleep(0.4)
            turns.agent_releases()
            released.set()

        asyncio.create_task(release_later())
        start = time.monotonic()

        # Drain the prospect's wait the way the pipeline does.
        async for _ in p._wait_for_turn(start + AGENT_TURN_START_WAIT):
            pass

        elapsed = time.monotonic() - start
        return elapsed, released.is_set()

    elapsed, did_release = asyncio.run(go())
    assert did_release, "prospect stopped waiting before the turn ended"
    assert elapsed > 0.35, f"returned in {elapsed:.2f}s, did not wait"


def test_wait_emits_silence_while_waiting():
    """The STT only runs when fed audio, and endpointing needs trailing silence.

    Blocking silently would glue the agent's reply onto the prospect's last
    sentence and transcribe the two as one utterance.
    """
    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(script_for("interested"), tts=None,
                              turn_gap=0.0, turns=turns)
        turns.agent_claims()

        chunks = []

        async def release_later():
            await asyncio.sleep(0.1)
            turns.agent_releases()

        asyncio.create_task(release_later())
        async for chunk in p._wait_for_turn(0.0):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(go())
    assert chunks, "prospect emitted nothing while waiting"
    assert all(c == bytes(len(c)) for c in chunks), "wait should emit silence"


def test_gives_up_if_the_agent_never_speaks():
    """A silent agent must not hang the call."""
    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(script_for("interested"), tts=None,
                              turn_gap=0.0, turns=turns)
        p._agent_turn_start_wait = 0.2   # the prospect's own budget
        start = time.monotonic()
        async for _ in p._wait_for_turn(start + 0.2):
            pass
        return time.monotonic() - start

    elapsed = asyncio.run(go())
    assert elapsed < 1.0, f"waited {elapsed:.2f}s for a turn that never began"


def test_gives_up_after_a_turn_that_never_finishes():
    """An agent that claims the line and never releases must not trap us."""
    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(script_for("interested"), tts=None,
                              turn_gap=0.0, turns=turns)
        turns.agent_claims()
        start = time.monotonic()
        async for _ in p._wait_for_turn(start + 10.0):
            if time.monotonic() - start > 1.0:
                break
        return time.monotonic() - start

    elapsed = asyncio.run(go())
    assert elapsed < 1.5, f"waited {elapsed:.2f}s on a stuck agent"


def test_prospect_takes_the_line_without_observed_silence():
    """Regression: a forced 30s wait before every line.

    The silence gate used to be the only thing arbitrating the line, and a
    waiter cannot produce the audio that would satisfy it. Waiting on the
    coordinator instead removed the trap: the agent does not release until its
    audio has finished playing, so the prospect never needs to guess.
    """
    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(script_for("interested"), tts=None,
                              turn_gap=0.0, turns=turns)
        # Nobody holds the line.
        assert not turns.agent_holding
        loop = asyncio.get_event_loop()
        start = loop.time()
        ok = await turns.wait_for_release(timeout=5.0)
        return ok, loop.time() - start, p

    ok, elapsed, _ = asyncio.run(go())
    assert ok is True, "wait failed on a free line"
    assert elapsed < 0.2, f"waited {elapsed:.2f}s for a line nobody held"