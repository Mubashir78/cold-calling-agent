"""Turn arbitration: exactly one authority on who has the line.

This module replaces a turn lock, a boolean "agent is speaking" flag, and a
pair of queued/finished sample counters. Those were three mechanisms that each
answered the same question, and they disagreed: on a real call the lock
reported the line free (`holder=None`) while the turn flag reported it busy,
and the prospect spoke through a reply that was still playing. Eleven seconds
of overlap across one conversation.

Overlap is the signature failure of a duplex voice agent, and it is caused by
slow or missed interruption detection. Prevention is not three interlocking
mechanisms, it is one.

So there is one now. `TurnCoordinator` holds one fact — whether the agent has
the line — and both sides read that. The prospect waits on an event that is set
when the agent releases. The agent releases when its audio has finished
PLAYING, which is signalled by the playback consumer rather than inferred from
sample counts.

Two rules that the disagreements all broke:

A turn ends when the audio has finished playing, not when it has been
synthesized. Piper synthesizes faster than real time, so a reply is fully
"prepared" well before it is audible.

Releasing is something the *player* announces, not something the speaker
infers. The speaker cannot know when its audio stopped coming out of the
speaker; only the consumer knows that.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TurnCoordinator:
    """Who has the line, and when it is handed back.

    Deliberately half-duplex. On a phone call only one side talks at a time,
    and an explicit turn boundary is both easier to test and safer for
    scripted accuracy than learned overlap handling.
    """

    def __init__(self) -> None:
        # True from the moment the agent claims its turn — before the model is
        # even called, since the prospect must know a reply is coming — until
        # its last sentence has finished playing.
        self._agent_holding = False
        self._released = asyncio.Event()
        self._released.set()   # nobody holds the line to begin with
        self.overlaps: list[str] = []

    @property
    def agent_holding(self) -> bool:
        return self._agent_holding

    def agent_claims(self) -> None:
        """Take the line for a turn.

        Re-entrant: a turn that claims twice is a bug upstream, but it must
        not deadlock the call waiting for a release that pairs with the first.
        """
        if self._agent_holding:
            logger.warning("Agent claimed the line while already holding it")
            self.overlaps.append("agent:double_claim")
            return
        self._agent_holding = True
        self._released.clear()

    def agent_releases(self, reason: str = "turn_complete") -> None:
        """Hand the line back. Called by the playback consumer."""
        if not self._agent_holding:
            # Releasing without holding is harmless but worth seeing: it means
            # something upstream released twice.
            logger.debug("Release with no turn held (%s)", reason)
            return
        self._agent_holding = False
        self._released.set()

    async def wait_for_release(self, timeout: float = 25.0) -> bool:
        """Block until the agent's turn has finished playing.

        Returns False on timeout. A prospect will not hold a line forever, so a
        silent agent must not be able to hang the call — but it must also not
        be able to end the call by accident.
        """
        if not self._agent_holding:
            return True
        try:
            await asyncio.wait_for(self._released.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.info("Agent did not release the line within %.0fs", timeout)
            self.overlaps.append("prospect:timeout")
            return False

    def reset(self) -> None:
        self._agent_holding = False
        self._released.set()
        self.overlaps.clear()


class TurnClaim:
    """Context manager claiming the line for one turn.

    Used with `async with` only where the release is genuinely known to be
    immediate. The normal path releases from the playback consumer, because
    that is the only place that knows when audio finished coming out.
    """

    def __init__(self, coordinator: TurnCoordinator, timeout: float = 30.0):
        self._coordinator = coordinator
        self._timeout = timeout

    async def __aenter__(self) -> "TurnClaim":
        if self._coordinator.agent_holding:
            # Wait rather than barge. This is the path used when the agent is
            # answering a queued turn while the previous one still plays out.
            await self._coordinator.wait_for_release(self._timeout)
        self._coordinator.agent_claims()
        return self

    async def __aexit__(self, *exc) -> None:
        # Only release what this block actually took. If the caller is driving
        # playback completion itself, the playback consumer releases instead
        # and a release here would hand the line back mid-audio.
        self._coordinator.agent_releases("turn_claim_exit")


def merge_intervals(spans, gap: float = 0.05) -> list[tuple[float, float]]:
    """Merge overlapping [start, end] spans, optionally absorbing small gaps.

    Two spans 30ms apart are one utterance for the purpose of "did these
    overlap", not two: they are the same phrase split by a plosive.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]