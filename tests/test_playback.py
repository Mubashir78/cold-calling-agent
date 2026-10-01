"""Playback, and the turn-completion signal it owns.

The bug this exists to prevent: a turn ended when its audio was *queued*, not
when it finished *playing*. Piper synthesizes faster than real time, so a
reply is fully prepared seconds before it is audible. Releasing the line at
that point is precisely when two voices become audible at once.

Synthesize, play and finish are three different moments. Every bug in this
area came from conflating them.
"""

import asyncio

import numpy as np
import pytest

from playback import Playback, _unpack
from turns import TurnCoordinator

SAMPLE_RATE = 16000


def _speech(seconds: float) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.clip(0.3 * np.sin(2 * np.pi * 220 * t), -1, 1) * 32767
            ).astype(np.int16).tobytes()


def test_unpack_accepts_the_current_shape():
    assert _unpack(("agent", b"x", True)) == ("agent", b"x", True)


def test_unpack_accepts_older_shorter_shapes():
    """Existing callers keep working across the shape change."""
    assert _unpack(("agent", b"x")) == ("agent", b"x", False)
    assert _unpack(b"x") == ("agent", b"x", False)


def test_line_is_held_until_audio_finishes_playing():
    """The core guarantee: releasing at synthesis time causes overlap."""
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        playback.begin_turn()
        turns.agent_claims()
        # 0.4s of audio, marked as the last chunk of the turn.
        q.put_nowait(("agent", _speech(0.4), True))

        await asyncio.sleep(0.15)
        held_early = turns.agent_holding
        await asyncio.sleep(0.45)
        held_late = turns.agent_holding

        consumer.cancel()
        return held_early, held_late

    held_early, held_late = asyncio.run(go())
    assert held_early is True, "line released while audio was still playing"
    assert held_late is False, "line never released after playback finished"


def test_turn_completion_event_follows_playback():
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        playback.begin_turn()
        turns.agent_claims()
        q.put_nowait(("agent", _speech(0.3), True))

        await playback.await_turn_complete(timeout=5.0)
        consumer.cancel()
        return turns.agent_holding

    assert asyncio.run(go()) is False


def test_await_turn_complete_times_out_instead_of_hanging():
    """If nothing ever drains, the pipeline must not block forever."""
    async def go():
        playback = Playback(coordinator=TurnCoordinator(), turn_gap=0.0)
        playback.begin_turn()
        loop = asyncio.get_event_loop()
        start = loop.time()
        got = await playback.await_turn_complete(timeout=0.2)
        return got, loop.time() - start

    got, elapsed = asyncio.run(go())
    assert got is False
    assert 0.15 < elapsed < 1.0


def test_playback_plays_at_real_time_not_instantly():
    """Draining instantly would stamp a whole reply inside milliseconds and
    invent overlap no listener could hear."""
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        loop = asyncio.get_event_loop()
        start = loop.time()
        playback.begin_turn()
        turns.agent_claims()
        for _ in range(3):
            q.put_nowait(("agent", _speech(0.1), False))
        q.put_nowait(("agent", _speech(0.1), True))
        await playback.await_turn_complete(timeout=5.0)
        elapsed = loop.time() - start
        consumer.cancel()
        return elapsed

    elapsed = asyncio.run(go())
    assert 0.35 < elapsed < 1.2, f"drained {elapsed:.2f}s for 0.4s of audio"


def test_turn_gap_is_applied_before_the_line_is_released():
    """The prospect must not start on the agent's last consonant."""
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.25)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        playback.begin_turn()
        turns.agent_claims()
        q.put_nowait(("agent", _speech(0.1), True))

        await asyncio.sleep(0.2)
        still_held = turns.agent_holding
        await asyncio.sleep(0.3)
        consumer.cancel()
        return still_held, turns.agent_holding

    still_held, released = asyncio.run(go())
    assert still_held is True, "line released with no beat of silence"
    assert released is False

def test_end_of_turn_marker_releases_the_line():
    """Regression: nothing marked the end of a turn, so the line never freed.

    Every queued chunk was `("agent", pcm, False)`. Playback only releases on a
    turn_end marker, so the call would deadlock after its first reply: the
    agent would hold the line forever and the prospect would wait out its full
    allowance. Every test that fed Playback directly passed, because they
    supplied the marker themselves — only the pipeline's own enqueue was wrong.
    """
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        playback.begin_turn()
        turns.agent_claims()
        q.put_nowait(("agent", _speech(0.1), False))
        q.put_nowait(("turn_end", b"", True))   # the marker, no audio

        got = await playback.await_turn_complete(timeout=3.0)
        consumer.cancel()
        return got, turns.agent_holding

    got, holding = asyncio.run(go())
    assert got is True, "turn-end marker never completed the turn"
    assert holding is False, "line still held after the turn-end marker"


def test_marker_is_not_recorded_as_audio():
    """A zero-length marker must not land on either audio track."""
    from sim.recorder import CallRecording

    async def go():
        rec = CallRecording("A", "B", "+14155550101")
        rec.start()
        turns = TurnCoordinator()
        playback = Playback(recorder=rec, coordinator=turns, turn_gap=0.0)
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        playback.begin_turn()
        turns.agent_claims()
        q.put_nowait(("agent", _speech(0.1), False))
        q.put_nowait(("turn_end", b"", True))
        await playback.await_turn_complete(timeout=3.0)
        consumer.cancel()
        return rec

    rec = asyncio.run(go())
    assert all(c.samples.size > 0 for c in rec.prospect), (
        "marker was written to the prospect track"
    )
