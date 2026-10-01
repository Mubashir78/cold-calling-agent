"""Turn arbitration.

Regression: eleven seconds of overlap in one conversation, because three
mechanisms each answered "who has the line?" and disagreed. A turn lock
reported the line free (`holder=None silence=0.14s need=0.35s`) at the same
moment the turn flag reported it busy. The prospect spoke through a reply that
was still playing.

Overlap is the signature failure of a duplex agent. These tests pin the one
mechanism that is supposed to prevent it.
"""

import asyncio

import pytest

from turns import TurnClaim, TurnCoordinator, merge_intervals


def test_line_starts_free():
    c = TurnCoordinator()
    assert not c.agent_holding


def test_claim_and_release():
    c = TurnCoordinator()
    c.agent_claims()
    assert c.agent_holding
    c.agent_releases()
    assert not c.agent_holding


def test_wait_returns_immediately_when_nobody_holds():
    """A prospect must not sit out a wait budget for a turn that never began.

    Waiting on "is the agent speaking?" when it is not returns at once, and the
    prospect answers over a reply that has not started yet. That was the first
    overlap bug.
    """
    async def go():
        c = TurnCoordinator()
        loop = asyncio.get_event_loop()
        start = loop.time()
        await c.wait_for_release(timeout=30.0)
        return loop.time() - start

    assert asyncio.run(go()) < 0.1


def test_wait_blocks_until_the_turn_actually_finishes():
    async def go():
        c = TurnCoordinator()
        c.agent_claims()
        loop = asyncio.get_event_loop()
        start = loop.time()

        async def release_later():
            await asyncio.sleep(0.3)
            c.agent_releases()

        task = asyncio.create_task(release_later())
        got = await c.wait_for_release(timeout=5.0)
        elapsed = loop.time() - start
        await task
        return got, elapsed

    got, elapsed = asyncio.run(go())
    assert got is True
    assert elapsed > 0.25, f"returned in {elapsed:.2f}s, did not wait"


def test_wait_times_out_rather_than_hanging_the_call():
    """A silent agent must not be able to hold the line forever."""
    async def go():
        c = TurnCoordinator()
        c.agent_claims()
        loop = asyncio.get_event_loop()
        start = loop.time()
        got = await c.wait_for_release(timeout=0.2)
        return got, loop.time() - start

    got, elapsed = asyncio.run(go())
    assert got is False
    assert 0.15 < elapsed < 1.0
    assert any("timeout" in o for o in TurnCoordinator().overlaps) is False


def test_release_without_holding_is_harmless():
    """Defensive: a double release must not corrupt the next turn's state."""
    c = TurnCoordinator()
    c.agent_releases()          # nothing held
    assert not c.agent_holding
    c.agent_claims()
    assert c.agent_holding


def test_double_claim_is_recorded_not_silently_accepted():
    c = TurnCoordinator()
    c.agent_claims()
    c.agent_claims()
    assert "agent:double_claim" in c.overlaps


def test_turn_claim_waits_for_the_previous_turn():
    """Answering a queued segment must not barge the previous reply."""
    async def go():
        c = TurnCoordinator()
        c.agent_claims()
        acquired_while_first_turn_held = []

        async def use_claim():
            async with TurnClaim(c, timeout=5.0) as claim:
                # Inside the block the line must be ours, and it must have been
                # taken *after* the previous turn released, not over it.
                acquired_while_first_turn_held.append(
                    c.agent_holding and "agent:double_claim" not in c.overlaps)

        task = asyncio.create_task(use_claim())
        await asyncio.sleep(0.2)
        # Still blocked: the first turn has not released.
        still_waiting = not task.done()
        c.agent_releases()
        await task
        return still_waiting, acquired_while_first_turn_held

    still_waiting, acquired = asyncio.run(go())
    assert still_waiting is True, "queued turn barged the previous reply"
    assert acquired == [True], "queued turn never took the line"


def test_merge_intervals_joins_a_turn_split_by_a_plosive():
    """Two spans 30ms apart are one utterance for overlap purposes."""
    merged = merge_intervals([(0.0, 1.0), (1.03, 2.0), (5.0, 6.0)])
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_intervals_keeps_genuinely_separate_spans():
    merged = merge_intervals([(0.0, 1.0), (3.0, 4.0)])
    assert len(merged) == 2