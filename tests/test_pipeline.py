"""Pipeline turn-taking, with fakes for STT, LLM and TTS.

No model, no network, no API key. This pins the behaviour a real call
depends on: the agent speaks first, and the disposition comes from tool
results rather than from keyword-matching the conversation.
"""

import asyncio
import json

import pytest

from llm.client import LLMResponse
from pipeline import VoicePipeline
from stt.engine import TranscriptSegment
from tools.registry import ToolRegistry


class FakeSTT:
    """Replays a fixed list of utterances, then goes quiet."""

    def __init__(self, utterances):
        self._utterances = list(utterances)

    async def initialize(self):
        pass

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kwargs):
        for text in self._utterances:
            yield TranscriptSegment(text, 0.0, 0.0, True, 0.0)


class FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return b"\x00" * 320


class ScriptedLLM:
    """Emits a fixed sequence of (text, tool_calls) per turn.

    Implements both entry points: the pipeline skips tool schemas on
    conversational turns and calls `generate` instead of
    `generate_with_tools`.
    """

    def __init__(self, turns):
        # turns: list of (text, [(tool_name, args), ...])
        self._turns = list(turns)
        self._i = 0
        self.executed: list[str] = []

    def generate(self, messages, tools=None, stream=True):
        return self._run(messages, None)

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._run(messages, executor)

    async def _run(self, messages, executor):
        if self._i >= len(self._turns):
            return
        text, tool_calls = self._turns[self._i]
        self._i += 1
        if text:
            yield LLMResponse(text=text)
        for name, args in tool_calls:
            # The real client executes the call before the next round; the
            # pipeline learns the outcome through the executor's return.
            if executor is not None:
                self.executed.append(name)
                await executor(name, args)
            yield LLMResponse(
                text="",
                tool_calls=[{
                    "id": f"c{self._i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }],
            )


def _stub(reg: ToolRegistry, name: str, result: dict) -> None:
    """Replace a tool's handler.

    Patching the bound method on the instance is not enough: ToolDefinition
    captured the original handler at registration time.
    """
    async def handler(**kwargs):
        handler.calls.append(kwargs)
        return result

    handler.calls = []
    reg._tools[name].handler = handler


async def _empty_stream():
    return
    yield b""  # pragma: no cover


async def _drain(q: asyncio.Queue) -> None:
    while True:
        await q.get()


def _registry() -> ToolRegistry:
    reg = ToolRegistry(
        crm_webhook="http://127.0.0.1:9/api/crm",
        calendar_webhook="http://127.0.0.1:9/api/calendar",
        dnc_webhook="http://127.0.0.1:9/api/dnc",
    )
    reg.start_call()
    reg.bind_call("+14155550101", "Dana Whitfield", "Northwind Logistics")
    return reg


async def _run(pipeline: VoicePipeline, opening: str = "") -> None:
    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    try:
        await pipeline.run_call(_empty_stream(), q, opening=opening)
    finally:
        task.cancel()


async def test_agent_speaks_first():
    """A real call opens with the agent, before the prospect says anything."""
    tts = FakeTTS()
    pipeline = VoicePipeline(
        stt=FakeSTT(["Yeah, what is this about?"]),
        llm=ScriptedLLM([("Sure, happy to explain.", [])]),
        tts=tts,
        tools=_registry(),
        system_prompt="test",
    )
    await _run(pipeline, opening="Hi, this is an AI call from Acme Corp.")

    spoken = " ".join(tts.spoken)
    assert "AI call from Acme Corp" in spoken
    # The opening is the first thing synthesized, before any reply.
    assert tts.spoken[0].startswith("Hi, this is an AI call")
    assert pipeline._conversation[1].role == "assistant"
    assert pipeline._conversation[1].content.startswith("Hi, this is an AI call")


async def test_successful_booking_yields_booked():
    reg = _registry()
    _stub(reg, "book_meeting", {"status": "ok", "event_id": "evt_1"})

    pipeline = VoicePipeline(
        stt=FakeSTT(["Tuesday at two works."]),
        llm=ScriptedLLM([
            ("", [("book_meeting", {
                "prospect_name": "Dana",
                "time_slot": "2031-01-05T14:00:00",
            })]),
            ("Booked for Tuesday at two.", []),
        ]),
        tts=FakeTTS(),
        tools=reg,
        system_prompt="test",
    )
    await _run(pipeline)

    assert "book_meeting" in pipeline._successful_tools
    assert pipeline.get_disposition() == "booked"


async def test_failed_booking_is_not_reported_as_booked():
    """Regression: a run reported 'booked' with zero meetings in the calendar.

    The tool points at a dead port, so the booking never happens. The
    disposition must not claim otherwise.
    """
    pipeline = VoicePipeline(
        stt=FakeSTT(["Tuesday at two works."]),
        llm=ScriptedLLM([
            ("", [("book_meeting", {
                "prospect_name": "Dana",
                "time_slot": "2031-01-05T14:00:00",
            })]),
            ("Let me get that booked for you.", []),
        ]),
        tts=FakeTTS(),
        tools=_registry(),
        system_prompt="test",
    )
    await _run(pipeline)

    assert "book_meeting" not in pipeline._successful_tools
    assert pipeline.get_disposition() != "booked"


async def test_dnc_request_sets_dnc_disposition():
    reg = _registry()
    _stub(reg, "honor_dnc", {"status": "ok"})

    pipeline = VoicePipeline(
        stt=FakeSTT(["Take me off your list."]),
        llm=ScriptedLLM([
            ("", [("honor_dnc", {"reason": "asked to be removed"})]),
            ("You're removed. Sorry to bother you.", []),
        ]),
        tts=FakeTTS(),
        tools=reg,
        system_prompt="test",
    )
    await _run(pipeline)

    assert "honor_dnc" in pipeline._successful_tools
    assert pipeline.get_disposition() == "dnc"
    # honor_dnc ends the call, so the prospect gets no further reply.
    assert pipeline._call_ended is True


async def test_transcript_records_both_sides():
    pipeline = VoicePipeline(
        stt=FakeSTT(["Yeah, what is this about?"]),
        llm=ScriptedLLM([("Let me explain quickly.", [])]),
        tts=FakeTTS(),
        tools=_registry(),
        system_prompt="test",
    )
    await _run(pipeline, opening="Hi, this is an AI call.")

    roles = [t.role for t in pipeline.get_transcript()]
    contents = [t.content for t in pipeline.get_transcript()]
    assert roles[0] == "assistant"          # the opening
    assert "user" in roles
    assert "assistant" in roles
    assert any("what is this about" in c for c in contents)
