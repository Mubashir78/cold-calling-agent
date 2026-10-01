"""Playback consumer.

The agent's queue carries chunks. Something has to play them at real time and
say when it is finished, because synthesis finishing is not playback finishing:
Piper generates audio faster than 1x, so a reply is fully prepared well before
it is audible.

That distinction was the source of an entire class of bug. Handing the line
back when the last sentence was *queued* releases it while the agent is still
speaking, and the prospect — which is now, correctly, waiting on the line —
answers over the tail of the reply.

So this module owns two facts:

- a chunk is *playing* from the moment it is taken off the queue
- it has *finished* only after its own duration has elapsed

The last chunk of a turn carries `turn_end=True`. When that chunk finishes, the
turn is over and the line is released. Nobody infers this from sample counts.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2   # 16-bit mono


class Playback:
    """Plays the agent's queued audio at real time and tracks turn completion.

    Turn completion is what releases the line. Synthesize/play/finish are three
    different moments, and conflating them is what made two voices audible at
    once.
    """

    def __init__(self, recorder=None, coordinator=None, turn_gap: float = 0.3):
        self.recorder = recorder
        self.coordinator = coordinator
        self.turn_gap = turn_gap
        self.bytes_played = 0
        # Set when the final chunk of a turn has finished playing. Awaited by
        # the agent so its turn does not end while audio is still audible.
        self._turn_complete = asyncio.Event()
        self._turn_complete.set()

    async def run(self, audio_out: asyncio.Queue) -> None:
        """Consume the queue until cancelled. Never returns."""
        total = 0
        try:
            while True:
                item = await audio_out.get()
                speaker, pcm, turn_end = _unpack(item)

                if turn_end and not pcm:
                    # End-of-turn marker: no audio, just the boundary. Hand the
                    # line back after a beat of silence so the prospect does
                    # not start on the agent's last consonant.
                    if self.turn_gap:
                        await asyncio.sleep(self.turn_gap)
                    if self.coordinator is not None:
                        self.coordinator.agent_releases()
                    self._turn_complete.set()
                    continue

                total += len(pcm)
                if self.recorder is not None:
                    # Stamp at the instant playback begins. Stamping at
                    # synthesis would compress the agent's track against the
                    # prospect's and invent overlap nobody could hear.
                    self.recorder.add_playback(speaker, pcm)

                await asyncio.sleep(len(pcm) / BYTES_PER_SECOND)

                if self.recorder is not None:
                    self.recorder.mark_playback_finished(speaker, len(pcm))

                if turn_end:
                    # Audio carried its own turn_end: the turn drained with it.
                    if self.turn_gap:
                        await asyncio.sleep(self.turn_gap)
                    if self.coordinator is not None:
                        self.coordinator.agent_releases()
                    self._turn_complete.set()
        except asyncio.CancelledError:
            logger.info("Agent said %d bytes of audio (%.1fs)",
                        total, total / BYTES_PER_SECOND)
            raise

    async def await_turn_complete(self, timeout: float = 30.0) -> bool:
        """Block until the current turn's audio has finished playing."""
        if not self._turn_complete.is_set():
            try:
                await asyncio.wait_for(self._turn_complete.wait(), timeout)
            except asyncio.TimeoutError:
                logger.warning("Agent playback did not drain within %.0fs", timeout)
                return False
        return True

    def begin_turn(self) -> None:
        """Mark that a turn is starting, so completion is awaited afresh."""
        self._turn_complete.clear()


def _unpack(item) -> tuple[str, bytes, bool]:
    """Accept the several shapes a queue item can take.

    `(speaker, pcm, turn_end)` is the current form. The shorter forms are kept
    working so existing callers and tests do not break on the shape change.
    """
    if isinstance(item, tuple):
        if len(item) == 3:
            return item[0], item[1], bool(item[2])
        if len(item) == 2:
            return item[0], item[1], False
    return "agent", item, False