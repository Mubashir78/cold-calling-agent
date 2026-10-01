"""Tool availability.

Regression worth guarding: tool schemas were withheld on turns whose wording
did not match a keyword list, and the model called a tool it had never been
offered. The SDK raised "Tool choice is none, but model called a tool" and
the call died, losing the booking path to save ~200 tokens.
"""

import asyncio

import pytest
from groq import APIError

from llm.client import LLMResponse
from pipeline import VoicePipeline
from stt.engine import TranscriptSegment
from tools.registry import ToolRegistry


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


class _LLM:
    def __init__(self, tool_calls_on=()):
        self.tool_calls_on = set(tool_calls_on)
        self.saw_tools: list[bool] = []

    def generate(self, messages, tools=None, stream=True):
        return self._run(tools)

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._run(tools)

    async def _run(self, tools):
        self.saw_tools.append(bool(tools))
        yield LLMResponse(text="Got it.")


def _pipeline(llm, tts=None, **kw):
    return VoicePipeline(
        stt=_STT(["Sure, that works for me."]),
        llm=llm,
        tts=tts or _TTS(),
        tools=ToolRegistry(),
        system_prompt="s",
        turn_spacing=0.0,
        **kw,
    )


# Wording that carries no booking keyword but is plainly agreement.
AGREEMENT_PHRASES = [
    "That sounds reasonable. Yeah, let's do it.",
    "Sure, that works for me.",
    "Alright, go ahead.",
    "Sounds good to me.",
    "Perfect, let's set it up.",
    "Yes, that's fine.",
    "Deal.",
    "I'm in.",
]


@pytest.mark.parametrize("phrase", AGREEMENT_PHRASES)
def test_tools_are_offered_on_plain_agreement(phrase):
    """The regression: keyword gating hid the schemas and killed the call."""
    p = _pipeline(_LLM())
    assert p._tools_for_turn(phrase) is not None, (
        f"no tools offered for agreement phrase: {phrase!r}"
    )


def test_booking_schema_withdrawn_after_a_booking():
    p = _pipeline(_LLM())
    p._successful_tools.add("book_meeting")
    tools = p._tools_for_turn("great, see you then")
    names = {t["function"]["name"] for t in tools}
    assert "book_meeting" not in names
    assert "log_disposition" in names, "other tools must stay available"


def test_no_tools_offered_once_the_call_has_ended():
    p = _pipeline(_LLM())
    p._call_ended = True
    assert p._tools_for_turn("anything") is None


def _api_error(message: str) -> APIError:
    """APIError takes (message, request, *, body) — see its __init__."""
    import httpx

    return APIError(
        message,
        httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        body={"error": {"message": message}},
    )


TOOL_SHAPE_MESSAGES = [
    "Failed to parse tool call arguments as JSON",
    "Tool choice is none, but model called a tool",
    "invalid tool choice",
    "Tool call validation failed: attempted to call tool 'book_meeting' "
    "which was not provided",
]


@pytest.mark.parametrize("message", TOOL_SHAPE_MESSAGES)
async def test_client_converts_tool_shape_errors(message):
    """The SDK words these three failures differently; all must convert."""
    from llm.client import GroqLLM, MalformedToolCall

    llm = GroqLLM(api_key="x")
    err = _api_error(message)

    async def boom():
        raise err

    with pytest.raises(MalformedToolCall):
        await llm._with_backoff(boom)


async def test_unrelated_api_errors_are_not_swallowed():
    """A server error must not be mistaken for a recoverable tool glitch."""
    from llm.client import GroqLLM

    llm = GroqLLM(api_key="x")
    err = _api_error("Internal server error")

    async def boom():
        raise err

    with pytest.raises(APIError):
        await llm._with_backoff(boom)


async def test_call_survives_a_tool_shape_error():
    """End to end through the pipeline: a bad tool turn must not end the call."""
    from llm.client import MalformedToolCall

    class _Flaky:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor,
                                max_tool_rounds=3):
            return self._run()

        async def _run(self):
            self.calls += 1
            if self.calls == 1:
                raise MalformedToolCall("bad tool call")
            yield LLMResponse(text="Got it.")

    tts = _TTS()
    p = _pipeline(_Flaky(), tts=tts)

    out = await p._run_turn(p._reply_stream("Yes, let's do it."))

    assert out == "Got it.", "no reply after a tool-shape error"
    assert tts.spoken, "no audio after a tool-shape error"


async def _empty():
    return
    yield b""
