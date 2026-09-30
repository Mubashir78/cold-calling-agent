"""Simulated prospect.

The PSTN is unreachable from a sandbox, so the far end is simulated. The
simulation is honest about where it stops: the prospect's *voice* is real
audio, synthesized by a second Piper voice, then put through the acoustic
conditions of a real phone call — narrowband codec, room tone, a breath
before anyone answers, a beep on voicemail. The agent therefore transcribes
degraded audio, which is the condition it will meet in production.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

from sim.environment import (
    add_noise_bed,
    band_limit,
    breath,
    to_pcm,
    voicemail_beep,
)
from tts.engine import PiperTTS
from turnlock import Speaker

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000
SILENCE = b"\x00" * CHUNK_BYTES

# A prospect does not resume talking the instant they stop. Human turn-taking
# is 200-300ms of dead air; anything past ~1.5s stops reading as conversation
# and starts reading as a query-response form. This is the *minimum* pause,
# independent of how long the agent's turn actually took.
MIN_TURN_GAP = 0.2

# Give up waiting for the agent after this long, so a silent agent cannot
# hang the call forever.
MAX_AGENT_WAIT = 25.0

# How long the prospect will wait for the agent to even *start* its turn.
# Covers endpointing plus a transcription round trip; the agent has not
# claimed the floor at the moment the prospect stops talking.
AGENT_TURN_START_WAIT = 12.0


@dataclass
class ProspectScript:
    """Behavioral profile for one simulated prospect."""

    name: str
    company: str
    persona: str
    lines: list[str] = field(default_factory=list)
    # Behavioral knobs, so the four scenarios do not all sound alike.
    speaking_rate: float = 1.0
    hangup_probability: float = 0.0
    voicemail: bool = False
    wrong_number: bool = False
    asks_to_be_removed: bool = False
    # Turn gaps, in seconds, sampled per turn.
    gap_range: tuple[float, float] = (4.0, 8.0)
    pickup_delay: float = 1.5


SCRIPTS: dict[str, ProspectScript] = {
    "interested": ProspectScript(
        name="Interested",
        company="Northwind Logistics",
        persona="busy operations manager who will take a short meeting",
        lines=[
            "Hello?",
            "Uh, hi. What's this about?",
            "Okay, well how much does it cost?",
            "That sounds reasonable. Yeah let's do it.",
            "Tuesday afternoon works for me, after two.",
            "Great, does that come with onboarding and training?",
            "Perfect. Send me the invite, I'll make it.",
        ],
        speaking_rate=1.05,
        gap_range=(0.25, 0.45),  # human inter-turn pause is 200-300ms
    ),
    "gatekeeper": ProspectScript(
        name="Gatekeeper",
        company="Redwood Health",
        persona="receptionist screening calls",
        lines=[
            "{company}, how may I direct your call?",
            "I'm not sure, who's asking?",
            "Well he's in meetings all afternoon.",
            "Can I take a message instead?",
            "No, that's okay. Just tell him an automated call, no callback.",
        ],
        speaking_rate=1.15,   # clipped and fast
        pickup_delay=0.6,     # receptionist picks up immediately
        gap_range=(0.2, 0.4),
    ),
    "dnc": ProspectScript(
        name="Do Not Call",
        company="Cobalt Systems",
        persona="irritated, has asked to be removed before",
        lines=[
            "Yeah, what?",
            "I already told you guys not to call. Remove me from your list.",
            "No, seriously. Take me off. Put me on the do not call list.",
            "Fine. Goodbye.",
        ],
        speaking_rate=1.1,
        hangup_probability=0.3,
        asks_to_be_removed=True,
        gap_range=(0.2, 0.4),
    ),
    "voicemail": ProspectScript(
        name="Voicemail",
        company="Ironwood Manufacturing",
        persona="voicemail box",
        lines=[
            "Hi you've reached {company}. Leave a message after the beep.",
        ],
        voicemail=True,
        speaking_rate=0.95,
        gap_range=(0.4, 0.8),
    ),
}


class SimulatedProspect:
    """Yields the prospect's audio as a live 16 kHz mono PCM stream.

    Paces itself so the agent gets a turn: after finishing a line it emits
    silence long enough for the agent's reply to be generated, synthesized
    and played out.
    """

    def __init__(
        self,
        script: ProspectScript,
        tts: PiperTTS,
        turn_gap: float = 0.0,
        pickup_delay: Optional[float] = None,
        phone_filter: bool = True,
        seed: int = 0,
        lock=None,
    ):
        # The script supplies the persona; the lead supplies the identity, so
        # a receptionist answers as the company actually called.
        self.script = script
        self.lines = [
            line.replace("{company}", script.company) for line in script.lines
        ]
        self.tts = tts
        self.turn_gap = turn_gap
        self.pickup_delay = (
            pickup_delay if pickup_delay is not None else script.pickup_delay
        )
        self.phone_filter = phone_filter
        self.rng = np.random.default_rng(seed)
        self._line_cache: dict[str, bytes] = {}

        # Handshake with the agent: the prospect waits for a reply rather
        # than assuming a fixed one.
        self.agent_speaking = False
        self.agent_turn_active = False
        self._agent_idle = asyncio.Event()
        self._agent_idle.set()
        self.lock = lock

    def agent_started_speaking(self) -> None:
        """Called when the agent claims the line, before it generates."""
        self.agent_speaking = True
        self.agent_turn_active = True
        self._agent_idle.clear()

    def agent_finished_speaking(self) -> None:
        """Called when the agent's turn is completely done."""
        self.agent_speaking = False
        self.agent_turn_active = False
        self._agent_idle.set()

    def heard(self) -> list[str]:
        return list(self.lines)

    async def prime(self) -> None:
        """Pre-synthesize every line so playback is not blocked on synthesis."""
        self._line_cache = {}
        for line in self.lines:
            pcm = await self.tts.synthesize_pcm(line)
            self._line_cache[line] = self._condition(pcm)

    def _condition(self, pcm: bytes) -> bytes:
        """Put synthesized speech through a phone line."""
        if not pcm:
            return pcm
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self.phone_filter:
            audio = band_limit(audio, SAMPLE_RATE, aggressiveness=0.6)
        audio = add_noise_bed(audio, self.rng)
        return to_pcm(audio)

    def _gap_for(self, index: int) -> float:
        """How long the prospect waits before speaking again.

        This is a *pause*, not the agent's speaking time. The agent's turn is
        awaited separately via `wait_for_agent`; guessing a duration here is
        what caused the agent to be talked over.
        """
        low, high = self.script.gap_range
        if self.turn_gap:
            return self.turn_gap
        return float(max(MIN_TURN_GAP, self.rng.uniform(low, high)))

    async def wait_for_agent(self, index: int) -> None:
        """Block until the agent has finished its reply.

        Used by tests and by any caller driving the prospect directly. The
        streaming path uses `_wait_for_agent_chunks`, which keeps feeding
        silence so the STT stays live.
        """
        await asyncio.sleep(self._gap_for(index))
        loop = asyncio.get_event_loop()
        start_wait = getattr(self, "_agent_turn_start_wait", AGENT_TURN_START_WAIT)
        # Phase 1 gets its own budget; using the combined deadline here made
        # the prospect sit out the full playback allowance waiting for a turn
        # that never began.
        start_deadline = loop.time() + start_wait
        end_deadline = start_deadline + MAX_AGENT_WAIT
        while not self.agent_turn_active and loop.time() < start_deadline:
            await asyncio.sleep(0.02)
        try:
            await asyncio.wait_for(
                self._agent_idle.wait(),
                timeout=max(0.1, end_deadline - loop.time()),
            )
        except asyncio.TimeoutError:
            logger.info("Agent did not finish its turn; prospect resumes anyway")

    async def _yield_pcm(self, pcm: bytes) -> AsyncIterator[bytes]:
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]
            await asyncio.sleep(CHUNK_MS / 1000)

    async def _yield_silence(self, seconds: float) -> AsyncIterator[bytes]:
        for _ in range(max(1, int(seconds * 1000 / CHUNK_MS))):
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

    async def _wait_for_agent_chunks(self, index: int) -> AsyncIterator[bytes]:
        """Emit silence until the agent has finished replying.

        Waiting only while the agent is *already* speaking was wrong. At the
        moment the prospect stops talking, the agent has not started: the
        STT still needs a beat of silence to fire, then a transcription
        round trip, then model generation. Waiting on the current state
        therefore returns immediately and the prospect talks straight over
        the reply.

        So: wait for the agent's turn to arrive, then for it to finish.
        """
        loop = asyncio.get_event_loop()
        # Phase 1 budget: enough for endpointing + transcription + generation.
        # Phase 2 budget: the turn itself, including playback.
        start_wait = getattr(
            self, "_agent_turn_start_wait", AGENT_TURN_START_WAIT
        )
        start_deadline = loop.time() + start_wait
        end_deadline = start_deadline + MAX_AGENT_WAIT

        # Phase 1: wait for the agent to claim the floor. Its turn starts
        # only after endpointing and transcription, neither of which has
        # happened yet.
        while not self.agent_turn_active and loop.time() < start_deadline:
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

        if not self.agent_turn_active:
            logger.info("Agent never took the floor; prospect resumes")

        # Phase 2: wait for that turn to finish playing.
        while self.agent_turn_active and loop.time() < end_deadline:
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

    async def _wait_silence(self, seconds: float) -> AsyncIterator[bytes]:
        """Emit silence in real time while waiting for the agent.

        Must keep yielding chunks. The STT only runs when fed audio, and
        endpointing needs the trailing silence to arrive as a stream — if this
        blocked silently, the agent's reply would be glued onto the
        prospect's last sentence and transcribed as one utterance.
        """
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

    async def stream(self) -> AsyncIterator[bytes]:
        """Ring, pickup, then the conversation."""
        async for chunk in self._yield_silence(self.pickup_delay):
            yield chunk
        if self.lock is not None:
            self.lock.note_silence(self.pickup_delay)

        # A breath before anyone answers, unless it went to voicemail.
        if not self.script.voicemail:
            async for chunk in self._yield_pcm(
                to_pcm(breath(self.rng) * 0.9)
            ):
                yield chunk

        if self.script.voicemail:
            async for chunk in self._yield_pcm(voicemail_beep()):
                yield chunk

        for index, line in enumerate(self.lines):
            if (
                self.script.hangup_probability
                and index > 0
                and self.rng.random() < self.script.hangup_probability
            ):
                logger.info("Prospect hung up early at line %d", index)
                break

            # Half-duplex: the prospect may only talk on a free line. Acquiring
            # blocks while the agent holds it — which now covers the agent's
            # whole turn, generation and playback alike.
            #
            # `require_silence=False` because the silence gate is the *agent's*
            # to clear: it stops the agent cutting in on a prospect who is
            # still speaking. For the prospect it is a trap. Waiting means
            # producing no audio, and silence is only credited from audio, so a
            # prospect that waited for silence could only ever reach the
            # threshold by not waiting — it sat out the full 30s lock timeout
            # and had to be forced on every single turn. The holder is the
            # authority here: the agent does not release until its audio has
            # finished playing.
            if self.lock is not None:
                await self.lock.acquire(Speaker.PROSPECT, require_silence=False)

            async for chunk in self._yield_pcm(self._line_cache.get(line, b"")):
                if self.lock is not None:
                    self.lock.note_speech()
                yield chunk

            if self.lock is not None:
                self.lock.release(Speaker.PROSPECT)

            if index < len(self.lines) - 1:
                # Keep the line fed with silence: first the natural pause,
                # then whatever the agent takes to reply. The STT needs
                # audio arriving continuously or it will not transcribe the
                # reply at all.
                async for chunk in self._wait_silence(self._gap_for(index)):
                    yield chunk
                if self.lock is not None:
                    self.lock.note_silence(self._gap_for(index))
                async for chunk in self._wait_for_agent_chunks(index):
                    yield chunk

        # Line goes dead.
        async for chunk in self._yield_silence(1.0):
            yield chunk


def script_for(key: str) -> ProspectScript:
    return SCRIPTS[key]
