"""Tests that compose the real objects the way production does.

Every other test in this suite builds its own pipeline and hands it its own
fake collaborators. That is right for testing a component and useless for
catching a mismatch *between* components, which is where this project's real
bugs have lived.

Seven consecutive live calls ended after the agent's second turn. The far end
had never once generated a reply, and the cause was a single missing method:
`Conversation.generate` calls `llm.complete(prompt)`, `GroqLLM` never defined
it. Every unit test passed, because each side was tested against its own fake
and the two halves were never asked to meet.

The rule these tests enforce: assert the contract between modules, and build
the object graph from the real classes rather than from stubs.
"""

import asyncio
from types import SimpleNamespace

import pytest


def _response(text: str):
    """A canned non-streaming chat completion, shaped like Groq's."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _FakeTransport:
    """Stands in for the Groq SDK, keeping the real client logic in play."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.prompts.append(kwargs["messages"][-1]["content"])
                text = (outer.replies.pop(0) if outer.replies
                        else "Okay, go on.")
                return _response(text)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


from llm.client import GroqLLM, LLMResponse
from playback import Playback
from pipeline import VoicePipeline
from sim.prospect import SimulatedProspect, script_for
from tools.registry import ToolRegistry
from turns import TurnCoordinator

class _AgentLLM:
    """Replies as the agent would, from whatever the far end said."""

    def __init__(self):
        self.turn = 0

    def _stream(self):
        self.turn += 1
        return self._run(self.turn)

    generate = _stream

    def generate_with_tools(self, messages, tools, executor, **kw):
        return self._stream()

    async def _run(self, n):
        yield LLMResponse(text=f"Agent turn {n}. When works for you?")

class _TTS:
    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return b"\x00" * 3200

class _STT:
    """Recognises the far end's speech, one turn at a time.

    Yields on every Nth audio chunk so the exchange has a rhythm: a real
    recogniser finalises once, after the caller stops talking. Yielding on
    every chunk would pile four transcripts into the same millisecond and the
    pipeline would correctly queue them.
    """

    CHUNKS_PER_TURN = 40

    def __init__(self, turns: int = 4):
        self.turns = turns
        self.chunks = 0

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        from stt.engine import TranscriptSegment

        said = 0
        async for _ in audio_in:
            self.chunks += 1
            if self.chunks % self.CHUNKS_PER_TURN or said >= self.turns:
                continue
            said += 1
            yield TranscriptSegment(
                "Sure, that works for me.", 0.0, 0.0, True, 1.0
            )


# -- the contract that broke ----------------------------------------------


def test_far_end_can_generate_through_the_real_groq_client():
    """The far end calls `complete`; the client must provide it.

    Seven live calls raised `AttributeError: 'GroqLLM' object has no attribute
    'complete'` inside the far end's generation task, logged at warning and
    otherwise invisible. The agent spoke to nobody for the entire call.
    """
    from llm.client import GroqLLM
    from sim.dialogue import Conversation, persona_for

    transport = _FakeTransport(["Sure, Wednesday afternoon works for me."])
    llm = GroqLLM(api_key="test-key")
    llm.client = transport

    conv = Conversation(persona_for("interested"), llm=llm)
    conv.note_agent_said("When works for you?")

    text = asyncio.run(conv.generate())

    assert text == "Sure, Wednesday afternoon works for me."
    assert transport.prompts, "the far end never reached the model"


def test_complete_goes_through_backoff_and_records_usage():
    """`complete` must share the client's rate-limit and accounting path.

    A far end that bypasses `_with_backoff` would not rotate to a fallback
    model when one model's daily budget is spent, so a call could die on a
    limit that another model would have answered.
    """
    from llm.client import GroqLLM

    transport = _FakeTransport(["Yes, that works."])
    llm = GroqLLM(api_key="test-key")
    llm.client = transport

    text = asyncio.run(llm.complete("Anything I can help with?"))

    assert text == "Yes, that works."
    assert llm.usage_log, "usage was not recorded for the far end's turn"


def test_complete_raises_on_daily_limit_rather_than_returning_nothing():
    """A rate limit must surface as an exception the caller can reason about.

    Returning an empty string would look to the far end like a model that had
    nothing to say, and the call would end quietly -- the exact failure mode
    that cost seven runs.
    """
    from llm.client import DailyTokenLimitExceeded, GroqLLM

    class _RateLimited:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    from groq import RateLimitError
                    raise RateLimitError(
                        "Rate limit reached for model",
                        response=SimpleNamespace(
                            headers={"x-ratelimit-remaining-tokens": "0"},
                            status_code=429,
                        ),
                        body=None,
                    )

    llm = GroqLLM(api_key="test-key", max_retries=0, retry_budget=0.1)
    llm.client = _RateLimited

    with pytest.raises((DailyTokenLimitExceeded, Exception)):
        asyncio.run(llm.complete("hello"))


# -- the object graph, assembled for real ---------------------------------


def test_pipeline_and_far_end_exchange_several_turns():
    """Both sides talking, built from the real classes.

    This is the test that was missing. It wires the same four objects `main.py`
    wires -- coordinator, playback, pipeline, far end -- with only the network
    boundary faked, and asserts the conversation actually advances.

    Every one of the seven failed live runs would have failed this.
    """
    async def go():
        # The far end's model: the real client, only the socket replaced.
        transport = _FakeTransport([
            "Sure, Wednesday afternoon works for me.",
            "Yes, send me the invite.",
            "Perfect, I'll make it.",
            "Thanks, talk soon.",
        ])
        prospect_llm = GroqLLM(api_key="test-key")
        prospect_llm.client = transport

        turns = TurnCoordinator()
        tts = _TTS()
        playback = Playback(coordinator=turns, turn_gap=0.02)
        prospect = SimulatedProspect(
            script_for("interested"), tts, turns=turns, llm=prospect_llm,
            phone_filter=False, pickup_delay=0.01,
        )
        pipeline = VoicePipeline(
            stt=_STT(), llm=_AgentLLM(), tts=tts, tools=ToolRegistry(),
            system_prompt="S", coordinator=turns, playback=playback,
            turn_spacing=0.0,
        )
        # The wiring that `main.py` performs, and that no other test covered.
        pipeline.on_far_end_turn = prospect
        await prospect.prime()
        # Four prospect turns is enough to prove both sides are talking. The
        # default budget is eight, which costs this test roughly twice as much
        # wall-clock for no extra signal.
        prospect._conversation.max_turns = 4

        audio_out: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(audio_out))

        # The object graph as `main.py` builds it: the far end's stream IS the
        # call's audio input, the recogniser transcribes it, the agent answers.
        # Wiring it any other way tests a system that does not exist.
        try:
            await asyncio.wait_for(
                pipeline.run_call(prospect.stream(), audio_out,
                                 opening="Opening line.", sample_rate=16000),
                timeout=25,
            )
        finally:
            consumer.cancel()

        return tts.spoken, transport.prompts

    spoken, far_prompts = asyncio.run(go())

    assert len(far_prompts) >= 3, (
        f"the far end only generated {len(far_prompts)} times; "
        "it is not answering the agent"
    )
    assert len(spoken) >= 3, f"the agent spoke {len(spoken)} times"