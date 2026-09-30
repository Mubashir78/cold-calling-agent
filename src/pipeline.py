"""Voice pipeline — ties STT, LLM, and TTS together for a single call."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from stt.engine import StreamingSTT, TranscriptSegment
from groq import RateLimitError

from llm.client import GroqLLM, MalformedToolCall, Message
from tts.engine import PiperTTS
from tools.registry import ToolRegistry
from turnlock import Speaker

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Production guidance: flush to TTS on sentence boundaries, not tokens.
# Word-level flushing wrecks prosody, and a lone "Yes." is not worth a chunk.
MIN_FLUSH_CHARS = 15


@dataclass
class _Turn:
    """Mutable state for one agent turn while it streams."""

    text: str = ""
    spoke: bool = False
    audio_started: bool = False


class Latency:
    """Per-turn timing, so the gap is measured rather than guessed.

    Mouth-to-ear latency — end of prospect speech to the agent's first audio
    — is the number users actually perceive, and it is the sum of these
    stages. Production guidance budgets ~50ms endpointing, ~200ms STT,
    ~300ms LLM first token and ~150ms TTS, targeting under 500ms total.
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

    def total(self) -> float:
        return sum(self.stages.values())

    def summary(self) -> str:
        parts = [f"{k}={v * 1000:.0f}ms" for k, v in self.stages.items()]
        return " ".join(parts) + f" total={self.total() * 1000:.0f}ms"


async def _tee(audio_in, recorder, lock=None):
    """Pass audio through, recording it and tracking line occupancy.

    Must stay a single continuous stream: the STT owns one segmenter for the
    whole call, so re-wrapping each chunk would break endpointing.
    """
    async for chunk in audio_in:
        if recorder is not None:
            recorder.add_prospect(chunk)
        if lock is not None:
            if _is_speech(chunk):
                lock.note_speech()
            else:
                lock.note_silence(CHUNK_SECONDS)
        yield chunk


CHUNK_SECONDS = 0.02


def _is_speech(chunk) -> bool:
    """Cheap RMS check on a PCM chunk.

    Accepts bytes or float samples. Never use `if not chunk` on a numpy
    array: its truthiness check raises rather than returning False.
    """
    import numpy as np

    if chunk is None:
        return False
    if isinstance(chunk, np.ndarray):
        samples = chunk.astype(np.float32, copy=False)
    else:
        if len(chunk) == 0:
            return False
        samples = np.frombuffer(bytes(chunk), dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
    if samples.size == 0:
        return False
    return float(np.sqrt(np.mean(samples**2))) >= 0.012


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class VoicePipeline:
    """Manages a single call's voice pipeline."""

    def __init__(
        self,
        stt: StreamingSTT,
        llm: GroqLLM,
        tts: PiperTTS,
        tools: ToolRegistry,
        system_prompt: str,
        max_call_duration: int = 600,
        recorder=None,
        max_history_turns: int = 6,
        lock=None,
        turn_spacing: float = 1.5,
        retry_backoff: float = 1.0,
        turn_gap: float = 0.3,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_call_duration = max_call_duration
        self.max_history_turns = max_history_turns
        self.recorder = recorder
        self.lock = lock
        self._turn_spacing = turn_spacing
        # Scales the wait between a 429 and its retry. Tests set it to 0
        # so a backoff schedule does not add tens of seconds of wall
        # clock to a suite that only checks which path was taken.
        self.retry_backoff = retry_backoff
        # Silence left after the agent's audio finishes, before the line is
        # released. Two voices at once is far worse than a slightly long gap.
        self.turn_gap = turn_gap
        # Set when a tool turn fails, so the next turn replies as plain text
        # rather than repeating a call that already broke.
        self._failed_tools = False
        # True from the moment a turn is claimed (before the model is even
        # called) until the last sentence has been spoken. The prospect waits
        # on this, not on "is audio playing".
        self._agent_turn_active = False
        self._latency = Latency()
        # Spoken when the model cannot be reached at all. Deliberately dull:
        # the point is to hold the line, not to improvise.
        self.hold_line = "Sorry, one moment while I check that for you."
        # Called with True when a reply starts, False when it ends, so the
        # simulated prospect can wait for the agent instead of guessing.
        self.on_agent_speaking: Optional[Callable[[bool], None]] = None
        self._conversation: list[Message] = []
        self._transcript: list[ConversationTurn] = []
        self._start_time: Optional[float] = None
        self._call_ended = False
        self._audio_out: Optional[asyncio.Queue] = None
        self._successful_tools: set[str] = set()

    async def run_call(
        self,
        audio_in: AsyncIterator[bytes],
        audio_out: asyncio.Queue,
        opening: str = "",
        sample_rate: int = 16000,
    ) -> None:
        """Run a full call.

        If `opening` is set the agent speaks first, which is what actually
        happens on a real call: ring, pickup, agent opens.

        One pipeline instance handles exactly one call.
        """
        self._start_time = time.time()
        self._conversation = [Message(role="system", content=self.system_prompt)]
        self._transcript = []
        self._call_ended = False
        self._audio_out = audio_out
        self._successful_tools = set()

        try:
            if opening.strip():
                # The opening occupies the line exactly like any other turn.
                # Claiming the turn flag here matters: the prospect starts
                # after a short pickup delay, and without this it would talk
                # straight over an opening that is still playing.
                self._begin_agent_turn()
                try:
                    await self._speak(opening, is_opening=True)
                finally:
                    self._end_agent_turn()
                self._conversation.append(Message(role="assistant", content=opening))
                self._transcript.append(
                    ConversationTurn(role="assistant", content=opening)
                )
                self._record("agent", opening)

            turn_task: Optional[asyncio.Task] = None
            pending: list[str] = []

            async for segment in self.stt.transcribe_stream(
                _tee(audio_in, self.recorder, self.lock),
                sample_rate=sample_rate,
            ):
                if time.time() - self._start_time > self.max_call_duration:
                    logger.info("Max call duration reached")
                    break
                if self._call_ended:
                    break

                logger.info("STT: %s", segment.text)
                self._transcript.append(
                    ConversationTurn(role="user", content=segment.text)
                )
                self._record("prospect", segment.text)
                self._conversation.append(
                    Message(role="user", content=segment.text)
                )

                if turn_task is not None and not turn_task.done():
                    # Half-duplex: the agent holds the line until its audio has
                    # finished playing. Hold this text and answer once the
                    # turn releases, rather than speaking over it.
                    pending.append(segment.text)
                    continue

                # The reply runs as a task so this loop keeps pulling audio.
                # Awaiting it here froze the whole chain: `transcribe_stream`
                # is a pull-based generator and `audio_in` is only advanced by
                # it, so nothing moved the prospect while the agent spoke. The
                # prospect's wait for the agent's turn was frozen mid-wait and
                # resumed against a turn that had already finished, which cost
                # the full wait timeout of dead air on every single turn.
                turn_task = asyncio.create_task(self._respond(segment.text))

            if turn_task is not None:
                await turn_task
            for text in pending:
                if self._call_ended:
                    break
                await self._respond(text)
        except Exception:
            logger.exception("Call pipeline error")
            raise
        finally:
            self._close()

    def _tools_for_turn(self, user_text: str) -> Optional[list[dict]]:
        """Tool schemas for this turn.

        Always offered, with one exception: once a booking exists, the
        book_meeting schema is withheld so a duplicate is impossible.

        An earlier version tried to save tokens by only attaching schemas
        when the prospect's words looked booking-related. That backfired:
        on "That sounds reasonable. Yeah, let's do it." the model called a
        tool it had never been offered and the SDK raised "Tool choice is
        none, but model called a tool", killing the call. The ~200 tokens a
        turn are not worth losing the booking path.
        """
        if self._call_ended:
            return None
        if "book_meeting" in self._successful_tools:
            return [t for t in self.tools.get_tools_for_llm()
                    if t["function"]["name"] != "book_meeting"]
        return self.tools.get_tools_for_llm()

    async def _respond(self, user_text: str = "") -> str:
        """Generate one agent turn and speak it.

        The whole turn is held, not just the TTS phase. The agent spends
        1-2s generating before any audio exists; if the line is only claimed
        while synthesizing, the prospect sees an idle line during generation
        and starts talking, and the two overlap. Claiming the line first is
        what makes turn-taking sound natural.
        """
        self._begin_agent_turn()
        try:
            return await self._respond_inner(user_text)
        finally:
            # A streamed turn enqueues sentences directly rather than going
            # through _speak, so the playback wait has to happen here. Ending
            # the turn when the last sentence is *queued* releases the line
            # while it is still audible.
            if self.recorder is not None:
                await self.recorder.await_agent_playback()
            if self.turn_gap:
                await asyncio.sleep(self.turn_gap)
            self._end_agent_turn()

    def _begin_agent_turn(self) -> None:
        self._agent_turn_active = True
        if self.on_agent_speaking is not None:
            self.on_agent_speaking(True)

    def _end_agent_turn(self) -> None:
        self._agent_turn_active = False
        if self.on_agent_speaking is not None:
            self.on_agent_speaking(False)

    async def _respond_inner(self, user_text: str = "") -> str:
        """Generate a turn, speaking each sentence as it completes.

        Waiting for the whole response before touching TTS is the largest
        avoidable latency in a cascaded pipeline: the caller hears nothing
        while the model finishes writing. Flushing on sentence boundaries
        overlaps generation with synthesis. Boundaries matter — token-level
        flushing destroys prosody, and a lone "Yes." is not worth a chunk.
        """
        tools = None if self._failed_tools else self._tools_for_turn(user_text)
        state = _Turn()

        try:
            if tools is not None:
                await self._stream_turn(
                    self._tool_stream(tools), state, with_tools=True
                )
            else:
                await self._stream_turn(self._plain_stream(), state)
        except RateLimitError as exc:
            # A rate limit says nothing about whether the model can follow a
            # tool schema, so the tools must NOT be dropped. Doing so took the
            # booking path away for every remaining turn of the call: the
            # model answered fluently and never once emitted book_meeting.
            if GroqLLM.is_daily_exhausted(exc):
                logger.warning("Daily token limit reached; using hold line")
                return await self._respond_hold()
            for attempt in range(1, 4):
                wait = GroqLLM._suggested_wait(exc, 2.0 * attempt)
                logger.info("Rate limited (%s); backing off %.1fs (try %d)",
                            GroqLLM.limit_kind(exc) or "unknown", wait,
                            attempt)
                if self.retry_backoff:
                    await asyncio.sleep(wait * self.retry_backoff)
                state = _Turn()
                try:
                    await self._stream_turn(self._tool_stream(tools), state,
                                            with_tools=True)
                    break
                except RateLimitError as retry_exc:
                    exc = retry_exc
                except MalformedToolCall:
                    break
            else:
                logger.warning("Rate limited after retries; using hold line")
                return await self._respond_hold()
        except MalformedToolCall:
            # A tool call the model could not parse is a real reason to answer
            # this turn without tools — but not to disable them for the call.
            logger.warning("Malformed tool call; replying without tools")
            self._failed_tools = True
            state = _Turn()
            try:
                await self._stream_turn(self._plain_stream(), state)
            except RateLimitError:
                logger.warning("LLM unavailable; using hold line")
                return await self._respond_hold()

        # Tools come back for the next turn: one bad parse should not disable
        # them for the rest of the call.
        self._failed_tools = False
        return await self._finish_turn(state.text, spoken=state.spoke)

    def _plain_stream(self):
        return self._llm_text_stream(
            self.llm.generate(self._trimmed_conversation(), stream=True)
        )

    def _tool_stream(self, tools):
        return self._llm_text_stream(
            self.llm.generate_with_tools(
                self._trimmed_conversation(), tools, self._executor
            )
        )

    def _executor(self, name: str, args: dict) -> dict:
        result_holder: dict = {}

        async def run() -> dict:
            result = await self.tools.execute(name, args)
            if result.get("success"):
                self._successful_tools.add(name)
                logger.info("tool %s ok", name)
            else:
                logger.warning("tool %s failed: %s", name, result.get("error"))
            result_holder["r"] = result
            return result

        return run()

    async def _llm_text_stream(self, source):
        """Normalize a response stream to text deltas, tracking tool calls."""
        async for resp in source:
            if resp.tool_calls:
                call = resp.tool_calls[0]
                name = call["function"]["name"]
                if name == "honor_dnc":
                    logger.info("Prospect asked to be removed from the list")
                    self._call_ended = True
                elif name == "log_disposition" and "end_call" in \
                        call["function"]["arguments"]:
                    self._call_ended = True
            if resp.text:
                yield resp.text

    async def _stream_turn(self, deltas, state, with_tools: bool = False) -> None:
        """Speak each completed sentence while the model is still writing."""
        if self._turn_spacing:
            await asyncio.sleep(self._turn_spacing)

        # Measured from the moment this turn was claimed — as close to
        # "prospect stopped talking" as this pipeline can observe, since
        # endpointing and transcription happen upstream in the STT.
        self._latency = Latency()
        self._latency.mark("to_first_audio")

        buffer = ""
        async for delta in deltas:
            state.text += delta
            buffer += delta

            parts = SENTENCE_SPLIT.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    await self._flush_sentence(sentence, state)
                buffer = parts[-1]

        await self._flush_sentence(buffer, state, force=True)

    async def _flush_sentence(self, sentence: str, state, force: bool = False):
        """Synthesize one sentence onto the output queue."""
        sentence = sentence.strip()
        if not sentence:
            return
        # Don't spend a chunk on a fragment; hold it for the next flush.
        if not force and not state.spoke and len(sentence) < MIN_FLUSH_CHARS:
            return
        if not state.spoke:
            state.spoke = True
        pcm = await self.tts.synthesize_pcm(sentence)
        if not pcm or self._audio_out is None:
            return
        if self.lock is not None:
            self.lock.note_speech()
        if self.recorder is not None:
            self.recorder.add_agent(pcm)
        await self._audio_out.put(("agent", pcm))
        if not state.audio_started:
            state.audio_started = True
            self._latency.stop("to_first_audio")

    async def _finish_turn(self, response_text: str, spoken: bool = False) -> str:
        """Record the turn. Audio was already emitted as it streamed."""
        response_text = response_text.strip()
        if not response_text:
            return ""
        self._conversation.append(Message(role="assistant", content=response_text))
        self._transcript.append(
            ConversationTurn(role="assistant", content=response_text)
        )
        self._record("agent", response_text)
        if not spoken:
            # Nothing streamed: empty generation, or the hold line.
            await self._speak(response_text)
        if self._latency.stages:
            logger.info("turn latency: %s", self._latency.summary())
        return response_text

    async def _respond_hold(self) -> str:
        if self._turn_spacing:
            await asyncio.sleep(self._turn_spacing)
        return await self._finish_turn(self.hold_line)

    async def _speak(self, text: str, is_opening: bool = False) -> None:
        """Synthesize sentence-by-sentence onto the output queue.

        Audio out goes wherever the media bridge takes it (LiveKit track here).

        The line is held across synthesis *and* playback. An earlier version
        of this function ran its synthesis loop twice, once in the body and
        again inside the `finally`, which doubled every line of audio.
        """
        if self.lock is not None:
            # The opening line goes out before the prospect has spoken, so it
            # cannot wait for silence that will never come.
            await self.lock.acquire(Speaker.AGENT, bypass_silence=is_opening)
        try:
            for sentence in [s for s in SENTENCE_SPLIT.split(text) if s.strip()]:
                pcm = await self.tts.synthesize_pcm(sentence.strip())
                if pcm and self._audio_out is not None:
                    if self.lock is not None:
                        self.lock.note_speech()
                    if self.recorder is not None:
                        self.recorder.add_agent(pcm)
                    # Tagged so the consumer can timestamp at playback time.
                    await self._audio_out.put(("agent", pcm))
        finally:
            # Synthesis is not playback. Wait for the audio to actually play
            # out, then leave a beat of silence, before releasing the line —
            # otherwise the prospect starts talking over the tail of the
            # agent's reply and two voices are audible at once.
            if self.recorder is not None:
                await self.recorder.await_agent_playback()
            if self.turn_gap:
                await asyncio.sleep(self.turn_gap)
            if self.lock is not None:
                self.lock.release(Speaker.AGENT)

    def _trimmed_conversation(self) -> list[Message]:
        """System prompt plus the most recent turns.

        The hosted tier allows 7000 input tokens per minute and every turn
        resends the history, so an unbounded transcript exhausts the budget
        mid-call and the agent dies on a 429. Phone calls stay short; a
        rolling window is also more faithful to what an agent would remember.
        """
        history = self._conversation[1:]  # keep the system prompt pinned
        if len(history) <= self.max_history_turns:
            return self._conversation
        return [self._conversation[0]] + history[-self.max_history_turns:]

    def _record(self, speaker: str, text: str) -> None:
        if self.recorder is not None:
            self.recorder.note(speaker, text)

    def _close(self):
        """Release per-call resources. The in-process server outlives calls."""
        pass

    def get_transcript(self) -> list[ConversationTurn]:
        """Get the full conversation transcript."""
        return self._transcript

    def get_disposition(self) -> str:
        """Determine the call disposition from the conversation."""
        full_text = " ".join(t.content.lower() for t in self._transcript)

        # A tool result is ground truth: the booking either happened or it
        # did not. Never infer "booked" from the conversation alone, or the
        # agent can claim a meeting that no calendar ever received.
        if "book_meeting" in self._successful_tools:
            return "booked"
        if "mark_callback" in self._successful_tools:
            return "callback"
        if "honor_dnc" in self._successful_tools:
            return "dnc"

        if any(
            w in full_text
            for w in ["voicemail", "leave a message", "not available"]
        ):
            return "voicemail"
        if any(
            w in full_text
            for w in ["wrong number", "not the right", "who is this"]
        ):
            return "wrong_number"
        if any(w in full_text for w in ["not interested", "no thanks"]):
            return "not_interested"

        return "no_outcome"
