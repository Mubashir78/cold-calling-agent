"""Behaviour when the model cannot be reached.

The org-wide token budget is shared, so throttling is an expected condition
rather than an edge case. A voice agent that goes silent reads as a dropped
call, so there must always be something to say.
"""

import asyncio

import pytest
from groq import RateLimitError

from llm.client import LLMResponse
from pipeline import VoicePipeline
from stt.engine import TranscriptSegment
from tools.registry import ToolRegistry


def _rate_limited():
    """Build a RateLimitError.

    groq's exception reads `response.request` in its constructor, so a stub
    response with a `request` attribute is required.
    """

    class _Response:
        status_code = 429
        headers: dict = {}
        request = object()

        def json(self):
            return {"error": {"message": "Rate limit reached. try again in 30s"}}

    return RateLimitError(
        "Rate limit reached", response=_Response(),
        body={"error": {"message": "Rate limit reached. try again in 30s"}},
    )


class _STT:
    def __init__(self, utterances):
        self._u = list(utterances)

    async def initialize(self):
        pass

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        async for _ in audio_in:
            pass
        for t in self._u:
            yield TranscriptSegment(t, 0.0, 0.0, True, 0.0)


class _TTS:
    def __init__(self):
        self.spoken = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return b"\x00" * 640


class _AllThrottledLLM:
    """Every entry point raises, as it would with the budget exhausted."""

    def generate(self, messages, tools=None, stream=True):
        return self._fail()

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._fail()

    async def _fail(self):
        raise _rate_limited()
        yield  # pragma: no cover


class _ToolPathThrottledOnce:
    """The tool path 429s once, then works.

    Used to pin a regression: the old code answered a rate-limited tool turn
    with the *plain* path and set `_failed_tools`, which silently removed the
    booking path for every remaining turn of the call. The agent stayed
    fluent and never once emitted book_meeting, so the call booked nothing.
    A 429 says nothing about whether the model can follow a tool schema.
    """

    def __init__(self):
        self.tool_calls = 0
        self.plain_calls = 0

    def generate(self, messages, tools=None, stream=True):
        return self._plain()

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._tools()

    async def _tools(self):
        self.tool_calls += 1
        if self.tool_calls == 1:
            raise _rate_limited()
        yield LLMResponse(text="Let me check that for you.")

    async def _plain(self):
        self.plain_calls += 1
        yield LLMResponse(text="Let me check that for you.")


def _pipeline(llm, tts=None, **kw):
    return VoicePipeline(
        stt=_STT(["Tuesday at two works."]),
        llm=llm,
        tts=tts or _TTS(),
        tools=ToolRegistry(),
        system_prompt="s",
        turn_spacing=0.0,
        # These tests assert which path was taken, not how long the wait was.
        # Sleeping out a real backoff schedule would add ~12s per throttled
        # turn and push the suite past its timeout.
        retry_backoff=0.0,
        **kw,
    )


async def _empty():
    return
    yield b""


async def _drain(q):
    while True:
        await q.get()


async def test_fully_throttled_still_speaks():
    """Dead air reads as a dropped call; the agent must say something."""
    tts = _TTS()
    p = _pipeline(_AllThrottledLLM(), tts=tts)

    q = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    try:
        await p.run_call(_empty(), q, opening="Hi, this is an AI call.")
    finally:
        task.cancel()

    assert len(tts.spoken) >= 2, (
        f"agent only spoke {len(tts.spoken)} time(s) under total throttling"
    )


async def test_rate_limit_does_not_cost_the_call_its_tools():
    """A throttled turn must be retried with the tools still attached."""
    llm = _ToolPathThrottledOnce()
    tts = _TTS()
    p = _pipeline(llm, tts=tts)

    q = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    try:
        await p.run_call(_empty(), q, opening="Opening.")
    finally:
        task.cancel()

    assert llm.tool_calls >= 2, (
        "the tool path was abandoned after one 429 instead of retried"
    )
    assert llm.plain_calls == 0, (
        "a rate limit sent the turn down the plain path, which drops the "
        "booking for the rest of the call"
    )
    assert "check that" in " ".join(tts.spoken)


async def test_hold_line_is_recorded_in_transcript():
    p = _pipeline(_AllThrottledLLM())
    q = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    try:
        await p.run_call(_empty(), q, opening="Opening.")
    finally:
        task.cancel()

    texts = [t.content for t in p.get_transcript()]
    assert p.hold_line in texts, "hold line missing from transcript"


def test_hold_line_is_plausible_for_a_sales_call():
    """It must not sound like an error message."""
    p = _pipeline(_AllThrottledLLM())
    line = p.hold_line.lower()
    assert "sorry" in line or "one moment" in line
    for bad in ("error", "exception", "429", "rate limit", "unavailable"):
        assert bad not in line, f"hold line leaks internals: {bad}"


async def test_recording_is_written_even_when_the_call_fails():
    """Regression: 74s of audio was produced, then discarded on exception."""
    from sim.recorder import CallRecording

    rec = CallRecording("Dana", "Northwind", "+14155550101")
    rec.start()
    p = _pipeline(_AllThrottledLLM())
    p.recorder = rec

    q = asyncio.Queue()
    consumer = asyncio.create_task(_consume(q, rec))
    try:
        await p.run_call(_empty(), q, opening="Opening line here.")
    finally:
        consumer.cancel()

    # Whatever happened, the recorder must hold both sides.
    assert rec.agent or rec.prospect
    assert rec.transcript, "nothing recorded"


async def _consume(q: asyncio.Queue, rec) -> None:
    """Stand in for playback, so audio can drain as fast as it is queued."""
    while True:
        item = await q.get()
        speaker, pcm = item if isinstance(item, tuple) else ("agent", item)
        rec.add_playback(speaker, pcm)
        rec.mark_playback_finished(speaker, len(pcm) // 2)
