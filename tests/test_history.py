"""Conversation history windowing and rate-limit resilience."""

import asyncio
import json

import pytest

from llm.client import LLMResponse, Message
from pipeline import VoicePipeline
from stt.engine import TranscriptSegment


class _FakeSTT:
    def __init__(self, utterances):
        self._u = list(utterances)

    async def initialize(self):
        pass

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        for t in self._u:
            yield TranscriptSegment(t, 0.0, 0.0, True, 0.0)


class _FakeTTS:
    async def synthesize_pcm(self, text):
        return b"\x00" * 320


class _NLLM:
    """Replies with `n` distinct messages, recording the prompt each time."""

    def __init__(self, n=20):
        self.n = n
        self.prompt_sizes = []
        self._i = 0

    def generate(self, messages, tools=None, stream=True):
        return self._run(messages)

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._run(messages)

    async def _run(self, messages):
        self.prompt_sizes.append(sum(len(m.content) for m in messages))
        if self._i >= self.n:
            return
        self._i += 1
        yield LLMResponse(text=f"reply number {self._i}")


def _pipeline(llm, **kw):
    from tools.registry import ToolRegistry
    reg = ToolRegistry()
    return VoicePipeline(
        stt=_FakeSTT([]),
        llm=llm,
        tts=_FakeTTS(),
        tools=reg,
        system_prompt="SYSTEM",
        **kw,
    )


def test_short_history_is_untouched():
    p = _pipeline(_NLLM())
    p._conversation = [Message(role="system", content="SYSTEM")]
    for i in range(3):
        p._conversation.append(Message(role="user", content=f"u{i}"))
        p._conversation.append(Message(role="assistant", content=f"a{i}"))
    assert p._trimmed_conversation() == p._conversation


def test_long_history_is_windowed():
    """Regression: an unbounded transcript exhausted the 7000 ITPM budget
    and the agent died on a 429 partway through a call."""
    p = _pipeline(_NLLM(), max_history_turns=6)
    p._conversation = [Message(role="system", content="SYSTEM")]
    for i in range(40):
        p._conversation.append(Message(role="user", content=f"u{i}"))
    trimmed = p._trimmed_conversation()
    assert len(trimmed) <= 7
    assert trimmed[0].role == "system", "system prompt must stay pinned"
    assert trimmed[0].content == "SYSTEM"


def test_window_keeps_the_most_recent_turns():
    p = _pipeline(_NLLM(), max_history_turns=4)
    p._conversation = [Message(role="system", content="SYSTEM")]
    for i in range(10):
        p._conversation.append(Message(role="user", content=f"u{i}"))
    trimmed = p._trimmed_conversation()
    assert trimmed[-1].content == "u9"


def test_prompt_size_stops_growing():
    """The property that actually matters: tokens per turn must not grow
    without bound as the call goes on."""
    llm = _NLLM(n=200)
    p = _pipeline(llm, max_history_turns=6)
    p._conversation = [Message(role="system", content="SYSTEM")]

    sizes = []
    for i in range(60):
        p._conversation.append(Message(role="user", content=f"u{i}" * 40))
        sizes.append(sum(len(m.content) for m in p._trimmed_conversation()))

    assert max(sizes) - min(sizes) < 2000, "prompt grew without bound"
    assert sizes[-1] < sizes[len(sizes) // 2] * 2


def test_agent_replies_reach_the_recorder():
    """Regression: replies were spoken and logged but never recorded, so a
    transcript showed the agent speaking exactly once."""
    from tools.registry import ToolRegistry
    from sim.recorder import CallRecording

    reg = ToolRegistry()
    rec = CallRecording("A", "B", "+14155550101")
    p = VoicePipeline(
        stt=_FakeSTT(["Hello?", "And pricing?"]),
        llm=_NLLM(n=5),
        tts=_FakeTTS(),
        tools=reg,
        system_prompt="SYSTEM",
        recorder=rec,
    )

    async def go():
        q = asyncio.Queue()
        task = asyncio.create_task(_drain(q))
        try:
            await p.run_call(_empty(), q, opening="Opening line.")
        finally:
            task.cancel()

    asyncio.run(go())

    speakers = [s for s, _ in rec.transcript]
    assert speakers.count("agent") >= 2, (
        f"agent spoke {speakers.count('agent')} time(s); replies not recorded"
    )
    assert speakers[0] == "agent", "opening must be first"


async def _drain(q):
    while True:
        await q.get()


async def _empty():
    return
    yield b""
