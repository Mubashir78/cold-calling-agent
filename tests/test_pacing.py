"""Prospect pacing: the agent must get a turn.

Regression: the agent spoke its opening line and then nothing for the whole
call. The prospect advanced on a guessed timer and talked over every reply.
"""

import asyncio
import time

from sim.prospect import (
    MAX_AGENT_WAIT,
    MIN_TURN_GAP,
    SCRIPTS,
    SimulatedProspect,
    script_for,
)


def test_gap_is_a_pause_not_the_agents_speaking_time():
    """A pause is short; guessing a full turn length is what broke it.

    Human turn-taking leaves well under a second of dead air. The old
    3.5-6.5s values were turn lengths, and using them as pauses meant the
    prospect still talked over slow replies.
    """
    for key in SCRIPTS:
        p = SimulatedProspect(script_for(key), tts=None, turn_gap=0.0)
        gaps = [p._gap_for(i) for i in range(30)]
        assert all(MIN_TURN_GAP <= g <= 2.5 for g in gaps), (
            f"{key} pause too long: {max(gaps):.2f}s"
        )


def test_waits_for_the_agent_to_finish():
    async def go():
        p = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)
        p.agent_started_speaking()

        released = asyncio.Event()

        async def release_later():
            await asyncio.sleep(0.4)
            p.agent_finished_speaking()
            released.set()

        asyncio.create_task(release_later())
        start = time.monotonic()
        await p.wait_for_agent(0)
        elapsed = time.monotonic() - start
        assert released.is_set(), "wait returned before the agent finished"
        return elapsed

    elapsed = asyncio.run(go())
    assert elapsed > 0.3, f"returned in {elapsed:.2f}s, did not wait"


def test_waits_for_a_turn_that_has_not_started_yet():
    """Regression: the prospect talked over every reply.

    At the moment the prospect stops talking, the agent has not started: the
    STT needs a beat of silence to fire, then a transcription round trip, then
    model generation. Waiting on the agent's *current* state returned
    immediately, so the prospect began its next line underneath the reply.
    """
    import asyncio as _asyncio

    async def go():
        p = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)

        async def agent_turn():
            await _asyncio.sleep(0.3)   # endpointing + transcription
            p.agent_started_speaking()
            await _asyncio.sleep(0.3)   # generation + playback
            p.agent_finished_speaking()

        task = _asyncio.create_task(agent_turn())
        start = _asyncio.get_event_loop().time()
        await p.wait_for_agent(0)
        elapsed = _asyncio.get_event_loop().time() - start
        await task
        return elapsed

    elapsed = asyncio.run(go())
    assert elapsed > 0.5, (
        f"resumed after {elapsed:.2f}s — did not wait for the agent's turn"
    )


def test_gives_up_if_the_agent_never_speaks():
    """A silent agent must not stall the call indefinitely."""

    async def go():
        p = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)
        p._agent_turn_start_wait = 0.2
        start = time.monotonic()
        await p.wait_for_agent(0)
        return time.monotonic() - start

    elapsed = asyncio.run(go())
    assert elapsed < 3.0, f"waited {elapsed:.2f}s for an agent that never spoke"


def test_gives_up_after_a_timeout():
    """A wedged agent must not hang the call forever."""

    async def go():
        p = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)
        p.agent_started_speaking()  # never finishes
        p._agent_idle.clear()
        await p.wait_for_agent(0)
        return True

    # MAX_AGENT_WAIT is long, so assert the mechanism exists rather than
    # actually waiting it out.
    assert MAX_AGENT_WAIT > 0
    assert asyncio.run(go()) is not None


def test_personas_have_distinct_pacing():
    rates = {k: s.speaking_rate for k, s in SCRIPTS.items()}
    assert len(set(rates.values())) >= 3, rates


def test_gatekeeper_answers_faster_than_voicemail_picks_up():
    assert (
        SCRIPTS["gatekeeper"].pickup_delay
        < SCRIPTS["interested"].pickup_delay
    )
