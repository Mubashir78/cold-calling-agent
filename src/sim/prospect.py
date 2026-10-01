"""Simulated prospect.

The PSTN is unreachable from a sandbox, so the far end is simulated. The
simulation is honest about where it stops: the prospect's voice is real audio,
synthesized by a second Piper voice, then put through the acoustic conditions
of a real phone call — narrowband codec, room tone, a breath before anyone
answers, a beep on voicemail. The agent therefore transcribes degraded audio,
which is the condition it will meet in production.

On turn-taking: the prospect does not guess how long the agent's reply will
take. It waits for `TurnCoordinator` to report the line free. Guessing was the
original design and it failed in both directions — it cut the agent off when
the guess ran long, and it sat out fixed dead air when the reply was short.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

from sim.dialogue import Conversation, persona_for
from sim.environment import (
    add_noise_bed,
    band_limit,
    breath,
    to_pcm,
    voicemail_beep,
)
from tts.engine import PiperTTS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000
SILENCE = b"\x00" * CHUNK_BYTES

# A prospect does not resume the instant they stop. Human turn-taking leaves
# 200-300ms; past ~1.5s it stops reading as conversation and starts reading as
# a query-response form. This is the floor, before the agent's reply.
MIN_TURN_GAP = 0.2

# Give up waiting for a reply that never comes, so a silent agent cannot hang
# the call. Also bounds the phase where the agent is generating: at the moment
# the prospect stops, the agent has not started, because transcription and
# endpointing have not happened yet.
AGENT_TURN_START_WAIT = 12.0
MAX_AGENT_WAIT = 30.0


@dataclass
class ProspectScript:
    """Behavioral profile for one simulated prospect."""

    name: str
    company: str
    persona: str
    lines: list[str] = field(default_factory=list)
    speaking_rate: float = 1.0
    hangup_probability: float = 0.0
    voicemail: bool = False
    asks_to_be_removed: bool = False
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
        lines=["Hi you've reached {company}. Leave a message after the beep."],
        voicemail=True,
        speaking_rate=0.95,
        gap_range=(0.4, 0.8),
    ),
}


class SimulatedProspect:
    """Yields the prospect's audio as a live 16 kHz mono PCM stream."""

    def __init__(
        self,
        script: ProspectScript,
        tts: PiperTTS,
        turns=None,
        turn_gap: float = 0.0,
        pickup_delay: Optional[float] = None,
        phone_filter: bool = True,
        seed: int = 0,
        llm=None,
    ):
        # The script supplies the persona; the lead supplies the identity, so a
        # receptionist answers as the company actually called.
        self.script = script
        self.lines = [
            line.replace("{company}", script.company) for line in script.lines
        ]
        self.tts = tts
        self.turns = turns
        self.turn_gap = turn_gap
        self.pickup_delay = (
            pickup_delay if pickup_delay is not None else script.pickup_delay
        )
        self.phone_filter = phone_filter
        self.rng = np.random.default_rng(seed)
        self._line_cache: dict[str, bytes] = {}
        self._spoken: list[str] = []
        self._last_agent_turn: str = ""
        # Text of the reply that was generated during the agent's turn. Paired
        # with the cached audio under "__prepared__".
        self._prepared_text: Optional[str] = None

        # The prospect generates its own lines. Both sides of this call are then
        # language models talking to each other, which is the honest shape of
        # the problem — a scripted far end produces a scripted conversation, and
        # it answers a question that was never asked.
        #
        # Each persona carries a goal and a turn budget so the model is steered
        # rather than left to improvise: a free-running caller keeps talking past
        # the end of the call.
        self._persona = persona_for(_persona_key(script), script.company)
        self._conversation = Conversation(self._persona, llm=llm)
        # Only the first line is scripted. That is the "hello?" before the agent
        # has said anything; every line after it answers a real turn.
        self._opening_lines = self.lines[:1]

    def heard(self) -> list[str]:
        return list(self.lines)

    async def prime(self) -> None:
        """Pre-synthesize the opening lines so playback is not blocked.

        Only the opening line is scripted, so only it can be pre-synthesized.
        Every later line is generated in response to the agent, and is prepared
        during that agent's turn instead — see `prepare`.
        """
        self._line_cache = {}
        for line in self._opening_lines:
            pcm = await self.tts.synthesize_pcm(line)
            self._line_cache[line] = self._condition(pcm)

    def on_agent_turn(self, text: str) -> None:
        """Record what the agent said.

        Called as each agent turn completes. The reply is generated and
        synthesized by `warm_next_reply`, which the pipeline awaits while this
        turn is still playing out.
        """
        self._last_agent_turn = text or ""
        self._conversation.note_agent_said(self._last_agent_turn)

    def prepare(self) -> None:
        """Begin generating and synthesizing the answer to this agent turn.

        Called while the agent is still speaking. Without this the prospect
        cannot reply until it has both generated and synthesized, which is
        seconds of dead air after every agent turn — the single largest source
        of the gap in this call.

        Deliberately takes no synthesizer argument. It used to accept one and
        forward it to `Conversation.prepare`, and the pipeline passed its OWN
        TTS. That object synthesized the prospect's reply, so from the second
        utterance onward the caller spoke in the agent's voice — the transcript
        still said `prospect:`, so nothing looked wrong except the audio. The
        voice belongs to the speaker, so it is not a parameter.
        """
        self._conversation.prepare(self.tts)

    async def warm_next_reply(self) -> None:
        """Wait for the prepared reply, then cache audio and text together.

        Both are stashed. The audio is what plays; the text is what advances
        the conversation, so losing it makes the far end mute on every turn
        after the first.
        """
        prepared = await self._conversation.take_prepared(
            self.tts, condition=self._condition)
        if prepared:
            pcm, text = prepared
            self._line_cache["__prepared__"] = pcm
            self._prepared_text = text

    def _next_line(self, index: int) -> Optional[str]:
        """The text of this turn, or None when the conversation is over."""
        if self._conversation.should_stop:
            return None
        if index == 0 and self._opening_lines:
            return self._opening_lines[0]
        return self._prepared_text

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
        """The pause before speaking again.

        A pause, not the agent's speaking time. The agent's turn is waited on
        separately via `TurnCoordinator`; guessing a duration here is what
        caused the agent to be talked over.
        """
        low, high = self.script.gap_range
        if self.turn_gap:
            return self.turn_gap
        return float(max(MIN_TURN_GAP, self.rng.uniform(low, high)))

    async def _yield_pcm(self, pcm: bytes) -> AsyncIterator[bytes]:
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]
            await asyncio.sleep(CHUNK_MS / 1000)

    async def _yield_silence(self, seconds: float) -> AsyncIterator[bytes]:
        for _ in range(max(1, int(seconds * 1000 / CHUNK_MS))):
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

    async def _wait_while_speaking(self, deadline: float) -> AsyncIterator[bytes]:
        """Emit silence while the agent holds the line.

        Must keep yielding chunks. The STT only runs when fed audio, and
        endpointing needs trailing silence to arrive as a stream — blocking
        silently would glue the agent's reply onto the prospect's last sentence
        and transcribe the two as one utterance.
        """
        loop = asyncio.get_event_loop()
        while self.turns is not None and self.turns.agent_holding:
            if loop.time() > deadline:
                break
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

    async def stream(self) -> AsyncIterator[bytes]:
        """Ring, pickup, then the conversation."""
        async for chunk in self._yield_silence(self.pickup_delay):
            yield chunk

        # A breath before anyone answers, unless it went to voicemail.
        if not self.script.voicemail:
            async for chunk in self._yield_pcm(
                to_pcm(breath(self.rng) * 0.9)
            ):
                yield chunk

        if self.script.voicemail:
            async for chunk in self._yield_pcm(voicemail_beep()):
                yield chunk

        loop = asyncio.get_event_loop()
        said = 0
        while True:
            if (self.script.hangup_probability and said > 0
                    and self.rng.random() < self.script.hangup_probability):
                logger.info("Prospect hung up early after %d turns", said)
                break

            # The reply generated during the agent's turn. Taken before any
            # waiting, so the audio is already conditioned and ready the
            # instant the line frees.
            prepared = self._line_cache.pop("__prepared__", None)

            line = self._next_line(said)
            if line is None and prepared is None:
                break

            if prepared is not None:
                pcm = prepared
            else:
                if line is None:
                    break
                pcm = self._line_cache.get(line)
                if pcm is None:
                    raw = await self.tts.synthesize_pcm(line)
                    pcm = self._condition(raw)
                    self._line_cache[line] = pcm

            async for chunk in self._yield_pcm(pcm):
                yield chunk
            if line:
                self._conversation.note_prospect_said(line)
                self._spoken.append(line)
                self._prepared_text = None
            said += 1

            if self._conversation.should_stop:
                break

            # Wait for the agent's whole turn. At the moment the prospect stops
            # talking the agent has not started — endpointing and transcription
            # have not happened yet — so waiting on "is it speaking?" would
            # return immediately and the prospect would answer over the reply.
            deadline = loop.time() + AGENT_TURN_START_WAIT
            async for chunk in self._yield_silence(self._gap_for(said)):
                yield chunk
            async for chunk in self._wait_and_prepare(deadline):
                yield chunk

        # Line goes dead.
        async for chunk in self._yield_silence(1.0):
            yield chunk

    async def _wait_and_prepare(self, deadline: float) -> AsyncIterator[bytes]:
        """Wait for the agent's turn, and build our answer while we wait.

        Preparation happens here rather than at the top of the loop because the
        reply is generated *during* the agent's turn. Reading the prepared audio
        before that point always found nothing: the prospect arrived at the top
        of the loop, found no reply yet, waited, and then left with silence
        because the reply it had just caused to be built was never read.
        """
        async for chunk in self._wait_for_turn(deadline):
            yield chunk
        # The turn is over; its text is on the conversation already.
        if self._last_agent_turn:
            self.prepare()
            await self.warm_next_reply()


    async def _wait_for_turn(self, start_deadline: float) -> AsyncIterator[bytes]:
        """Wait for the agent's whole turn, emitting silence throughout.

        Two phases because they are different failures. If the agent never
        claims the line, waiting on it alone burns the entire playback budget
        on a turn that does not exist. So: wait for the turn to *arrive*, then
        wait for it to *finish*.
        """
        loop = asyncio.get_event_loop()
        if self.turns is None:
            return

        # Phase 1: wait for the turn to be claimed.
        while not self.turns.agent_holding and loop.time() < start_deadline:
            yield SILENCE
            await asyncio.sleep(CHUNK_MS / 1000)

        if not self.turns.agent_holding:
            logger.info("Agent never took the floor; prospect resumes")

        # Phase 2: wait for that turn to finish playing.
        async for chunk in self._wait_while_speaking(
            loop.time() + MAX_AGENT_WAIT
        ):
            yield chunk


def script_for(key: str) -> ProspectScript:
    return SCRIPTS[key]


# Which persona a script maps onto. Keyed by persona name so the script stays
# the behavioural description and the persona stays the voice.
_PERSONA_BY_NAME = {
    "interested": "interested",
    "gatekeeper": "gatekeeper",
    "do not call": "dnc",
    "dnc": "dnc",
}


def _persona_key(script: ProspectScript) -> str:
    return _PERSONA_BY_NAME.get((script.name or "").strip().lower(), "interested")