"""Voice pipeline: STT -> LLM -> TTS for a single call.

Turn arbitration lives in `turns.TurnCoordinator` and completion lives in
`playback.Playback`. This module owns neither; it only drives them. That
separation is the point: when one module decided both, the two facts
disagreed and the prospect spoke through a reply that was still playing.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from groq import RateLimitError

from llm.client import GroqLLM, MalformedToolCall, Message
from stt.engine import TranscriptSegment
from turns import TurnCoordinator

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Flush to TTS on sentence boundaries, not on tokens. Word-level flushing
# wrecks prosody, and a lone "Yes." is not a sentence worth a chunk.
MIN_FLUSH_CHARS = 15

# Zero-length queue item that declares "the turn ends here". The playback
# consumer releases the line when it reaches one. Using an explicit marker
# rather than marking the final sentence keeps the boundary independent of how
# the text happened to be split into sentences.
TURN_END = "turn_end"


@dataclass
class _Turn:
    """Mutable state for one agent turn while it streams."""

    text: str = ""
    spoke: bool = False
    audio_started: bool = False


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class Latency:
    """Per-turn timing, so the gap is measured rather than guessed.

    Mouth-to-ear latency is what a caller perceives: the span from the moment
    the turn is claimed to the first audio. It is the sum of endpointing,
    transcription, generation and synthesis.
    """

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.monotonic()

    def stop(self, name: str) -> None:
        start = self._marks.pop(name, None)
        if start is not None:
            self.stages[name] = time.monotonic() - start

    def summary(self) -> str:
        parts = [f"{k}={v * 1000:.0f}ms" for k, v in self.stages.items()]
        total = sum(self.stages.values())
        return " ".join(parts) + f" total={total * 1000:.0f}ms"


class VoicePipeline:
    """Runs one call. Not shared across calls: it holds per-call state."""

    def __init__(
        self,
        stt,
        llm: GroqLLM,
        tts,
        tools,
        system_prompt: str,
        max_call_duration: int = 600,
        recorder=None,
        coordinator: Optional[TurnCoordinator] = None,
        playback=None,
        turn_spacing: float = 1.5,
        retry_backoff: float = 1.0,
        max_history_turns: int = 6,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_call_duration = max_call_duration
        self.max_history_turns = max_history_turns
        self.recorder = recorder
        self.turns = coordinator or TurnCoordinator()
        self.playback = playback
        self._turn_spacing = turn_spacing
        # Scales the wait between a 429 and its retry. Tests set it to 0 so a
        # backoff schedule does not add tens of seconds of wall clock to a
        # suite that only checks which path was taken.
        self.retry_backoff = retry_backoff

        self._failed_tools = False
        self._latency = Latency()
        # Spoken when the model cannot be reached. Deliberately dull: the point
        # is to hold the line, not to improvise.
        self.hold_line = "Sorry, one moment while I check that for you."
        self._conversation: list[Message] = []
        self._transcript: list[ConversationTurn] = []
        self._start_time: Optional[float] = None
        self._call_ended = False
        self._audio_out: Optional[asyncio.Queue] = None
        self._successful_tools: set[str] = set()
        # Segments that arrived while the agent still held the line.
        self._queued: list[str] = []

    async def run_call(
        self,
        audio_in: AsyncIterator[bytes],
        audio_out: asyncio.Queue,
        opening: str = "",
        sample_rate: int = 16000,
    ) -> None:
        """Run a full call.

        The reply runs as a task rather than being awaited inline. `audio_in` is
        a pull-based generator advanced only by the STT, so awaiting inline
        froze the entire chain for the length of the agent's turn: the
        prospect's wait was frozen mid-wait and resumed against a turn that had
        already been spoken.
        """
        self._start_time = time.time()
        self._conversation = [Message(role="system", content=self.system_prompt)]
        self._transcript = []
        self._call_ended = False
        self._audio_out = audio_out
        self._successful_tools = set()
        self._queued = []

        try:
            if opening.strip():
                # The opening occupies the line exactly like any other turn.
                # The prospect starts after a pickup delay and must not talk
                # over an opening that is still playing.
                await self._run_turn(self._opening_stream(opening))

            task: Optional[asyncio.Task] = None
            async for segment in self.stt.transcribe_stream(
                _record_remote(audio_in, self.recorder),
                sample_rate=sample_rate,
            ):
                if time.time() - self._start_time > self.max_call_duration:
                    logger.info("Max call duration reached")
                    break
                if self._call_ended:
                    break

                logger.info("STT: %s", segment.text)
                self._note_prospect(segment.text)

                if task is not None and not task.done():
                    # Half-duplex. The agent still has the line; hold this text
                    # and answer once the turn releases rather than talk over it.
                    self._queued.append(segment.text)
                    continue

                task = asyncio.create_task(self._run_turn(self._reply_stream(
                    segment.text)))

            if task is not None:
                await task
            for text in self._queued:
                if self._call_ended:
                    break
                await self._run_turn(self._reply_stream(text))
        except Exception:
            logger.exception("Call pipeline error")
            raise

    async def _run_turn(self, source) -> str:
        """Claim the line, speak a turn, wait for it to finish playing."""
        if self.turns.agent_holding:
            # Answering a queued segment while a reply is still audible.
            await self.turns.wait_for_release(timeout=30.0)

        self.turns.agent_claims()
        if self.playback is not None:
            self.playback.begin_turn()
        self._latency = Latency()
        self._latency.mark("to_first_audio")

        try:
            state = _Turn()
            try:
                await self._speak_stream(source(), state)
            except RateLimitError as exc:
                await self._recover_rate_limit(source, exc, state)
            except MalformedToolCall as exc:
                # The model emitted a tool call it could not parse. Answer this
                # turn as plain text rather than lose the turn — and do NOT
                # disable tools for the rest of the call, which is what
                # silently cost a real booking.
                logger.warning("Malformed tool call (%s); plain text reply",
                               str(exc)[:60])
                plain = _Turn()
                await self._speak_stream(self._plain_stream(), plain)
                state.text += plain.text
                state.spoke = state.spoke or plain.spoke
            text = await self._finish_turn(state.text, spoken=state.spoke)

            # Mark the turn's end explicitly. Playback releases the line when it
            # reaches this marker, so the boundary is declared by the producer
            # and *observed* by the consumer — nobody has to infer which queued
            # chunk happened to be last. Without it the line is never released
            # and the call deadlocks after the first reply.
            if self._audio_out is not None:
                await self._audio_out.put((TURN_END, b"", True))

            # The turn is not over until its audio has finished PLAYING.
            if self.playback is not None:
                await self.playback.await_turn_complete()
            return text
        finally:
            # Safety net: if anything above raised, the line must still come
            # back or the rest of the call is a monologue.
            if self.turns.agent_holding:
                self.turns.agent_releases("abandoned")

    async def _speak_stream(self, deltas, state: _Turn) -> None:
        """Emit one sentence of audio as the model completes it.

        Waiting for the whole response before touching TTS is the largest
        avoidable latency in a cascaded pipeline: the caller hears nothing
        while the model finishes writing. Flushing on sentence boundaries
        overlaps generation with synthesis.
        """
        if self._turn_spacing:
            await asyncio.sleep(self._turn_spacing)

        buffer = ""
        async for delta in deltas:
            if delta is None:
                continue
            state.text += delta
            buffer += delta
            parts = SENTENCE_SPLIT.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    await self._flush_sentence(sentence, state)
                buffer = parts[-1]

        # Models routinely stop mid-sentence, so the tail always goes out.
        await self._flush_sentence(buffer, state, force=True)

    async def _flush_sentence(self, sentence: str, state: _Turn,
                              force: bool = False) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        # Don't spend a chunk on a fragment; hold it for the next flush.
        if not force and not state.spoke and len(sentence) < MIN_FLUSH_CHARS:
            return
        state.spoke = True

        pcm = await self.tts.synthesize_pcm(sentence)
        if not pcm or self._audio_out is None:
            return
        if self.recorder is not None:
            self.recorder.add_agent(pcm)
        await self._audio_out.put(("agent", pcm, False))
        if not state.audio_started:
            state.audio_started = True
            self._latency.stop("to_first_audio")

    async def _recover_rate_limit(self, source, exc, state: _Turn) -> None:
        """Ride out a 429 without losing the turn.

        A rate limit says nothing about whether the model can follow a tool
        schema, so tools must not be dropped for it: doing so silently removed
        the booking path for the rest of the call.
        """
        if GroqLLM.is_daily_exhausted(exc):
            logger.warning("Daily token limit reached; using hold line")
            await self._say_hold(state)
            return

        for attempt in range(1, 4):
            wait = GroqLLM._suggested_wait(exc, 2.0 * attempt)
            logger.info("Rate limited (%s); backing off %.1fs (try %d)",
                        GroqLLM.limit_kind(exc) or "unknown", wait, attempt)
            if self.retry_backoff:
                await asyncio.sleep(wait * self.retry_backoff)
            retry_state = _Turn()
            try:
                await self._speak_stream(source(), retry_state)
                state.text += retry_state.text
                return
            except RateLimitError as retry_exc:
                exc = retry_exc
            except MalformedToolCall:
                logger.warning("Malformed tool call after retry; plain text")
                break

        logger.warning("Rate limited after retries; using hold line")
        await self._say_hold(state)

    async def _say_hold(self, state: _Turn) -> None:
        """Speak the hold line as the whole turn.

        `state.text` is reset rather than assigned, because `_speak_stream`
        *appends* to it. Assigning first made the hold line speak twice and
        appear twice in the transcript.
        """
        state.text = ""
        state.spoke = False
        await self._speak_stream(_text_stream(self.hold_line), state)

    def _opening_stream(self, opening: str):
        """Factory for the opening utterance."""
        return lambda: _text_stream(opening)

    def _reply_stream(self, user_text: str):
        """Generate one turn, with tools unless this turn must go without them.

        Restarting the stream after a 429 means re-requesting with the same
        schemas, so this is a factory rather than a one-shot generator.
        """
        tools = None if self._failed_tools else self._tools_for_turn(user_text)

        def make():
            if tools is None:
                return self._plain_stream()
            return self._tool_stream(tools)

        return make

    def _tools_for_turn(self, user_text: str) -> Optional[list[dict]]:
        """Tool schemas for this turn.

        Always offered, with one exception: once a booking exists the
        book_meeting schema is withheld, so a duplicate is impossible.

        An earlier version attached schemas only when the prospect's words
        looked booking-related. That backfired: on "That sounds reasonable.
        Yeah, let's do it." the model called a tool it had never been offered
        and the SDK raised "Tool choice is none, but model called a tool",
        killing the call.
        """
        if self._call_ended:
            return None
        if "book_meeting" in self._successful_tools:
            return [t for t in self.tools.get_tools_for_llm()
                    if t["function"]["name"] != "book_meeting"]
        return self.tools.get_tools_for_llm()

    def _plain_stream(self):
        return self._text_stream(
            self.llm.generate(self._trimmed_conversation(), stream=True)
        )

    def _tool_stream(self, tools):
        return self._text_stream(
            self.llm.generate_with_tools(
                self._trimmed_conversation(), tools, self._executor
            )
        )

    def _executor(self, name: str, args: dict) -> dict:
        async def run() -> dict:
            result = await self.tools.execute(name, args)
            if result.get("success"):
                self._successful_tools.add(name)
                logger.info("tool %s ok", name)
            else:
                logger.warning("tool %s failed: %s", name, result.get("error"))
            return result
        return run()

    def _text_stream(self, source):
        """Normalize a response stream to text deltas, tracking tool calls."""
        async def gen():
            async for resp in source:
                if resp.tool_calls:
                    call = resp.tool_calls[0]
                    name = call["function"]["name"]
                    if name == "honor_dnc":
                        logger.info("Prospect asked to be removed")
                        self._call_ended = True
                    elif name == "log_disposition" and "end_call" in \
                            call["function"]["arguments"]:
                        self._call_ended = True
                if resp.text:
                    yield resp.text
        return gen()

    async def _finish_turn(self, response_text: str, spoken: bool = False) -> str:
        response_text = response_text.strip()
        if not response_text:
            return ""
        self._conversation.append(Message(role="assistant", content=response_text))
        self._transcript.append(
            ConversationTurn(role="assistant", content=response_text)
        )
        if self.recorder is not None:
            self.recorder.note("agent", response_text)
        if not spoken:
            await self._speak_literal(response_text)
        if self._latency.stages:
            logger.info("turn latency: %s", self._latency.summary())
        return response_text

    async def _speak_literal(self, text: str) -> None:
        """Speak text that was not streamed (the hold line, an empty reply)."""
        for sentence in [s for s in SENTENCE_SPLIT.split(text) if s.strip()]:
            pcm = await self.tts.synthesize_pcm(sentence.strip())
            if pcm and self._audio_out is not None:
                if self.recorder is not None:
                    self.recorder.add_agent(pcm)
                await self._audio_out.put(("agent", pcm, False))

    def _note_prospect(self, text: str) -> None:
        self._transcript.append(ConversationTurn(role="user", content=text))
        self._conversation.append(Message(role="user", content=text))
        if self.recorder is not None:
            self.recorder.note("prospect", text)

    def _trimmed_conversation(self) -> list[Message]:
        """System prompt plus the most recent turns.

        Every turn resends the history and the hosted tier meters input
        tokens per minute, so an unbounded transcript exhausts the budget
        mid-call. Phone calls stay short, and a rolling window is closer to
        what a person would remember anyway.
        """
        history = self._conversation[1:]   # keep the system prompt pinned
        if len(history) <= self.max_history_turns:
            return self._conversation
        return [self._conversation[0]] + history[-self.max_history_turns:]

    def get_transcript(self) -> list[ConversationTurn]:
        return self._transcript

    def get_disposition(self) -> str:
        """Outcome of the call.

        Tool results are ground truth: a booking either happened or it did not.
        Inferring "booked" from the conversation lets the agent claim a meeting
        no calendar ever received.
        """
        if "book_meeting" in self._successful_tools:
            return "booked"
        if "mark_callback" in self._successful_tools:
            return "callback"
        if "honor_dnc" in self._successful_tools:
            return "dnc"

        full_text = " ".join(t.content.lower() for t in self._transcript)
        if any(w in full_text for w in
               ["voicemail", "leave a message", "not available"]):
            return "voicemail"
        if any(w in full_text for w in
               ["wrong number", "not the right", "who is this"]):
            return "wrong_number"
        if any(w in full_text for w in ["not interested", "no thanks"]):
            return "not_interested"
        return "no_outcome"


async def _text_stream_static(text: str):
    yield text


def _text_stream(text: str):
    return _text_stream_static(text)


async def _record_remote(audio_in, recorder) -> AsyncIterator[bytes]:
    """Tee the far end's audio into the recording.

    Must be a single continuous pass-through: the STT owns one segmenter for
    the whole call, so re-wrapping chunks per segment would break endpointing.

    This is only recording. Turn arbitration is the coordinator's job, and
    doing it here as well is what let three mechanisms disagree about who had
    the line. Dropping this wrapper entirely is also wrong — without it the
    prospect track is empty, and an overlap report computed against an empty
    track reports zero overlap no matter how badly the call actually overlaps.
    """
    if recorder is None:
        async for chunk in audio_in:
            yield chunk
        return
    async for chunk in audio_in:
        recorder.add_prospect(chunk)
        yield chunk