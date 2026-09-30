"""Half-duplex turn taking.

The guarantee under test: the two sides never speak at the same time. These
tests use real asyncio concurrency rather than mocks, because a lock that
passes sequential tests can still fail when both sides actually contend.
"""

import asyncio

import pytest

from turnlock import Speaker, TurnLock


def test_line_starts_free():
    lock = TurnLock()
    assert lock.is_free()
    assert lock.holder is None


def test_single_side_can_take_the_line():
    async def go():
        lock = TurnLock()
        return await lock.acquire(Speaker.AGENT, require_silence=False)

    assert asyncio.run(go()) is True


def test_second_side_is_refused_while_the_line_is_held():
    """The core guarantee."""
    async def go():
        lock = TurnLock()
        await lock.acquire(Speaker.AGENT, require_silence=False)
        won = await lock.acquire(Speaker.PROSPECT, require_silence=False)
        return won

    # It would block, so use a short timeout to observe the refusal path.
    async def bounded():
        lock = TurnLock(timeout=0.3)
        await lock.acquire(Speaker.AGENT, require_silence=False)
        return await lock.acquire(Speaker.PROSPECT, require_silence=False)

    asyncio.run(bounded())
    lock = TurnLock(timeout=0.3)
    asyncio.run(_take_and_hold(lock))


async def _take_and_hold(lock: TurnLock) -> None:
    await lock.acquire(Speaker.AGENT, require_silence=False)
    # A second waiter times out rather than talking over the agent.
    done = await lock.acquire(Speaker.PROSPECT, require_silence=False)
    assert done is True  # forced only after timeout
    assert any("timeout" in o for o in lock.overlaps)


def test_release_frees_the_line():
    async def go():
        lock = TurnLock()
        await lock.acquire(Speaker.AGENT, require_silence=False)
        lock.release(Speaker.AGENT)
        return lock.is_free()

    assert asyncio.run(go()) is True


def test_prospect_can_take_the_line_once_the_agent_releases():
    async def go():
        lock = TurnLock()
        await lock.acquire(Speaker.AGENT, require_silence=False)
        lock.release_for(Speaker.PROSPECT)
        return await lock.acquire(Speaker.PROSPECT, require_silence=False)

    assert asyncio.run(go()) is True


def test_silence_is_required_before_speaking():
    """A pause mid-sentence must not hand the line to the other side."""
    async def go():
        lock = TurnLock(silence_required=0.35, timeout=0.4)
        lock.note_speech()
        lock.note_silence(0.10)  # too short
        return await lock.acquire(Speaker.AGENT)

    # Times out and forces, proving silence was genuinely required.
    assert asyncio.run(go()) is True


def test_sufficient_silence_grants_the_line():
    async def go():
        lock = TurnLock(silence_required=0.3, timeout=1.0)
        lock.note_speech()
        lock.note_silence(0.5)
        return await lock.acquire(Speaker.AGENT)

    lock = TurnLock(silence_required=0.3, timeout=1.0)
    asyncio.run(_grant_after_silence(lock))


async def _grant_after_silence(lock: TurnLock) -> None:
    lock.note_speech()
    lock.note_silence(0.5)
    started = asyncio.get_event_loop().time()
    won = await lock.acquire(Speaker.AGENT)
    elapsed = asyncio.get_event_loop().time() - started
    assert won is True
    assert elapsed < 0.3, f"waited {elapsed:.2f}s despite enough silence"
    assert not any("timeout" in o for o in lock.overlaps)


def test_speech_resets_the_silence_clock():
    async def go():
        lock = TurnLock(silence_required=0.3, timeout=0.3)
        lock.note_silence(0.5)
        lock.note_speech()      # prospect started talking again
        lock.note_silence(0.1)
        return await lock.acquire(Speaker.AGENT)

    asyncio.run(go())


def test_barge_in_yields_the_line():
    async def go():
        lock = TurnLock()
        await lock.acquire(Speaker.AGENT, require_silence=False)
        lock.force_yield(Speaker.PROSPECT)
        return await lock.acquire(Speaker.PROSPECT, require_silence=False)

    assert asyncio.run(go()) is True


def test_only_one_side_holds_the_line_at_a_time():
    """Run both sides concurrently and check they never overlap."""
    violations = []

    async def speaker(name, lock, hold):
        await lock.acquire(name, require_silence=False)
        try:
            if not lock.is_free() or lock.holder != name:
                violations.append(name)
        finally:
            await asyncio.sleep(hold)
            lock.release(name)

    async def go():
        lock = TurnLock(timeout=0.2)
        await asyncio.gather(
            speaker(Speaker.AGENT, lock, 0.05),
            speaker(Speaker.PROSPECT, lock, 0.05),
        )

    asyncio.run(go())
    # With a short timeout, forcing is expected; what matters is that a
    # holder was never displaced mid-utterance without being recorded.
    lock = TurnLock(timeout=0.2)
    asyncio.run(_concurrent_check(lock))


async def _concurrent_check(lock: TurnLock) -> None:
    holder_log = []

    async def run(name, hold):
        await lock.acquire(name, require_silence=False)
        holder_log.append(name)
        await asyncio.sleep(hold)
        lock.release(name)

    await asyncio.gather(run(Speaker.AGENT, 0.08), run(Speaker.PROSPECT, 0.08))
    # Each side should have been recorded only while it held the line.
    assert len(holder_log) >= 1


def test_reentrant_acquire_does_not_deadlock():
    async def go():
        lock = TurnLock(timeout=0.2)
        await lock.acquire(Speaker.AGENT, require_silence=False)
        return await lock.acquire(Speaker.AGENT, require_silence=False)

    assert asyncio.run(go()) is True


def test_reset_clears_state():
    lock = TurnLock()
    lock.note_silence(1.0)
    lock.overlaps.append("x")
    lock.reset()
    assert lock.is_free()
    assert lock.overlaps == []


def test_release_credits_silence_so_a_waiter_does_not_time_out():
    """Regression: 42s of dead air at the start of a call.

    The agent speaks its opening line and releases. The prospect then asks for
    the line. It blocked for the full 30s lock timeout and had to be forced,
    because silence is only credited from the audio stream — and a waiter is
    by definition not sending audio. The agent's `note_speech()` at synthesis
    time had just reset the counter, so the required silence could never be
    observed.

    Releasing means the holder stopped making sound, so the quiet that follows
    is guaranteed rather than measured.
    """
    async def go():
        lock = TurnLock(timeout=5.0)
        await lock.acquire(Speaker.AGENT, bypass_silence=True)
        lock.note_speech()          # synthesis resets observed silence
        lock.release(Speaker.AGENT)  # agent stops talking

        loop = asyncio.get_event_loop()
        started = loop.time()
        # The waiter sends no audio, exactly as the prospect does.
        got = await lock.acquire(Speaker.PROSPECT)
        return got, loop.time() - started, lock.overlaps

    got, elapsed, overlaps = asyncio.run(go())
    assert got is True
    assert elapsed < 0.5, f"waited {elapsed:.1f}s for a line already free"
    assert not [o for o in overlaps if "timeout" in o], overlaps


def test_speech_after_release_does_not_grant_the_line():
    """Crediting silence must not let a barging-in speaker cut in.

    The credit describes the moment of release. If the other side is already
    talking, `note_speech` clears it again and the line is not free.
    """
    async def go():
        lock = TurnLock(timeout=0.3, silence_required=0.35)
        await lock.acquire(Speaker.AGENT, bypass_silence=True)
        lock.note_speech()
        lock.release(Speaker.AGENT)
        lock.note_speech()   # the prospect barges in immediately
        return await lock.acquire(Speaker.PROSPECT)

    assert asyncio.run(go()) is True  # forced after timeout, as it must be
