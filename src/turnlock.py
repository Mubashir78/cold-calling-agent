"""Half-duplex turn taking.

On a phone call only one side talks at a time. Two independent hazards break
that, and both have bitten this project:

1. The agent starts speaking while the prospect is still talking. The agent
   is triggered by an STT segment, and endpointing can fire on a short pause
   in the middle of a sentence.
2. The prospect starts while the agent is mid-reply.

This module is the single arbiter. Both sides ask for permission; neither gets
it until the other has explicitly released.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# How long the speaker must be silent before the line is considered free.
# Long enough to cover a breath between words, short enough to feel natural.
DEFAULT_SILENCE_MS = 350

# If the other side never releases (crash, hangup), don't deadlock the call.
DEFAULT_LOCK_TIMEOUT = 30.0


class Speaker(str, Enum):
    AGENT = "agent"
    PROSPECT = "prospect"


class TurnLock:
    """Mutual exclusion between the two sides of a call.

    `acquire` waits until the line is free. `release` hands it over. The
    prospect releases while the agent is expected to speak; the agent
    releases when it has finished or when the prospect barges in.
    """

    def __init__(
        self,
        silence_required: float = DEFAULT_SILENCE_MS / 1000.0,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
    ):
        self.silence_required = silence_required
        self.timeout = timeout
        self._holder: Optional[Speaker] = None
        self._event = asyncio.Event()
        self._event.set()
        self._silence_seen_at: Optional[float] = None
        self._previous_holder: Optional[Speaker] = None
        self.overlaps: list[str] = []

    @property
    def holder(self) -> Optional[Speaker]:
        return self._holder

    def is_free(self) -> bool:
        return self._holder is None

    def note_silence(self, seconds: float) -> None:
        """Report that the line has been quiet for `seconds`."""
        if self._silence_seen_at is None:
            self._silence_seen_at = seconds
        else:
            self._silence_seen_at += seconds

    def note_speech(self) -> None:
        self._silence_seen_at = None

    def _silence_is_sufficient(self) -> bool:
        return (
            self._silence_seen_at is not None
            and self._silence_seen_at >= self.silence_required
        )

    def is_quiet(self) -> bool:
        """Is the line currently silent, regardless of duration?"""
        return self._silence_seen_at is not None

    async def acquire(
        self,
        speaker: Speaker,
        require_silence: bool = True,
        bypass_silence: bool = False,
    ) -> bool:
        """Take the line, waiting for the other side to finish.

        `bypass_silence` is for the opening line, which must be spoken before
        the prospect has said anything. Returns False only if the wait timed
        out, which should be logged rather than treated as success.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.timeout

        while True:
            if self._holder is None:
                if (
                    not require_silence
                    or bypass_silence
                    or self._silence_is_sufficient()
                ):
                    self._holder = speaker
                    self._event.clear()
                    self._note_takeover(speaker)
                    return True
            if self._holder == speaker:
                # Already speaking; re-entrant, so do not deadlock.
                return True
            if loop.time() >= deadline:
                logger.warning(
                    "%s could not take the line within %.0fs; forcing "
                    "(holder=%s silence=%.2fs need=%.2fs)",
                    speaker.value, self.timeout,
                    self._holder.value if self._holder else None,
                    self._silence_seen_at or 0.0, self.silence_required,
                )
                self.overlaps.append(f"{speaker.value}:timeout")
                self._holder = speaker
                self._event.clear()
                return True

            self._event.clear()
            with_timeout = max(0.05, deadline - loop.time())
            try:
                await asyncio.wait_for(self._event.wait(), with_timeout)
            except asyncio.TimeoutError:
                continue

    def _note_takeover(self, speaker: Speaker) -> None:
        previous = self._previous_holder
        if previous is not None and previous != speaker:
            self.overlaps.append(f"{speaker.value}:took_over:{previous.value}")
            logger.info("%s took the line from %s", speaker.value, previous.value)
        self._previous_holder = speaker

    def release(self, speaker: Speaker) -> None:
        """Give up the line.

        Releasing also *credits* silence. Silence is otherwise only observed
        from the audio stream, but a side waiting in `acquire` is by
        definition not producing audio — so it cannot be the thing that
        reports the quiet it is waiting for. Without this credit, a caller
        that had just been silent (the agent finishing a reply) left
        `_silence_seen_at` at None and the waiter blocked until the lock
        timeout, which on a real call is tens of seconds of dead air.
        """
        if self._holder == speaker:
            self._holder = None
            self._silence_seen_at = max(
                self._silence_seen_at or 0.0, self.silence_required
            )
            self._event.set()

    def release_for(self, other: Speaker) -> None:
        """Release and tell `other` it may take the line."""
        self._holder = None
        self._event.set()

    def force_yield(self, to_speaker: Speaker) -> None:
        """The current speaker stops because the other side barged in."""
        if self._holder is not None and self._holder != to_speaker:
            logger.info("Line yielded from %s to %s",
                        self._holder.value, to_speaker.value)
        self._holder = None
        self._event.set()

    def reset(self) -> None:
        self._holder = None
        self._previous_holder = None
        self._silence_seen_at = None
        self.overlaps.clear()
        self._event.set()
