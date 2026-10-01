"""Rate-limit handling.

Regression: every 429 was retried on a fixed 6s delay, exhausting all five
attempts in 30s while the server had explicitly said to wait 30s. The SDK's
own retries also fired first, so the backoff never saw the real error.
"""

import pytest

from llm.client import GroqLLM


class _FakeRateLimit(Exception):
    """Stands in for groq.RateLimitError, which needs an HTTP response."""

    def __init__(self, body):
        super().__init__("429")
        self.body = body


def test_parses_server_suggested_wait():
    body = {
        "error": {
            "message": "Rate limit reached for model `qwen`. "
                       "Please try again in 30.14s. Need more tokens?",
        }
    }
    wait = GroqLLM._suggested_wait(_FakeRateLimit(body), fallback=5.0)
    assert wait == pytest.approx(30.14), "server's own delay was discarded"


def test_parses_from_a_json_string_body():
    wait = GroqLLM._suggested_wait(
        _FakeRateLimit('{"error":{"message":"try again in 12.5s"}}'), 5.0
    )
    assert wait == pytest.approx(12.5)


def test_falls_back_when_no_delay_is_given():
    body = {"error": {"message": "Rate limit reached."}}
    assert GroqLLM._suggested_wait(_FakeRateLimit(body), 7.0) == 7.0


def test_falls_back_on_unparseable_body():
    assert GroqLLM._suggested_wait(_FakeRateLimit(None), 4.0) == 4.0


def test_fallback_ramps_when_body_is_absent():
    """Each attempt waits longer than the last when the server says nothing."""
    llm = GroqLLM(api_key="x", retry_delay=5.0)
    waits = [
        llm._suggested_wait(_FakeRateLimit(None), llm.retry_delay * (i + 1))
        for i in range(4)
    ]
    assert waits == [5.0, 10.0, 15.0, 20.0], waits


def test_sdk_does_not_retry_underneath_us():
    """SDK retries would fire before our backoff and eat the budget."""
    llm = GroqLLM(api_key="x")
    assert llm.client.max_retries == 0, (
        "SDK retries are on; our backoff never sees the error"
    )


def test_default_attempts_allow_a_full_minute_window():
    llm = GroqLLM(api_key="x")
    assert llm.max_retries >= 6, "too few attempts to outlast a 30s window"


async def test_turn_spacing_is_applied():
    """A burst of turns starves the org-wide per-minute budget."""
    import asyncio
    import time

    from llm.client import LLMResponse
    from stt.engine import TranscriptSegment
    from tools.registry import ToolRegistry

    class _STT:
        async def initialize(self):
            pass

        async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
            async for _ in audio_in:
                pass
            for t in ["one", "two"]:
                yield TranscriptSegment(t, 0.0, 0.0, True, 0.0)

    class _TTS:
        async def synthesize_pcm(self, text):
            return b"\x00" * 320

    class _LLM:
        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
            return self._run()

        async def _run(self):
            yield LLMResponse(text="ok")

    from pipeline import VoicePipeline

    p = VoicePipeline(
        stt=_STT(), llm=_LLM(), tts=_TTS(), tools=ToolRegistry(),
        system_prompt="s", turn_spacing=0.4,
    )

    async def empty():
        return
        yield b""

    q = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    start = time.monotonic()
    try:
        await p.run_call(empty(), q, opening="hi")
    finally:
        task.cancel()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.8, f"spacing not applied; call took {elapsed:.2f}s"


async def _drain(q):
    while True:
        await q.get()


def _daily_429():
    """A 429 whose body names the per-day budget."""
    from test_throttle_fallback import _rate_limited
    exc = _rate_limited()
    # The real body says "per day"; the classifier keys off that phrase.
    return exc


def test_daily_exhaustion_rotates_to_the_next_model():
    """One model's spent budget must not end the call.

    Groq budgets are per model, so qwen exhausting its daily allowance says
    nothing about gpt-oss. Before this, the agent answered four turns in a row
    with the hold line and booked nothing.
    """
    from llm.client import GroqLLM

    llm = GroqLLM(
        api_key="x",
        model="qwen/qwen3.8-27b",
        fallback_models=["openai/gpt-oss-120b"],
        max_retries=0,
    )
    assert llm.active_model == "qwen/qwen3.8-27b"

    assert llm._rotate_model() is True
    assert llm.active_model == "openai/gpt-oss-120b"
    assert "qwen/qwen3.8-27b" in llm._exhausted

    # The exhausted model must not be retried.
    assert llm._rotate_model() is False
    assert llm.active_model == "openai/gpt-oss-120b"


def test_rotation_skips_models_already_exhausted():
    from llm.client import GroqLLM

    llm = GroqLLM(
        api_key="x",
        model="a",
        fallback_models=["b", "c"],
        max_retries=0,
    )
    llm._exhausted.add("b")
    assert llm._rotate_model() is True
    assert llm.active_model == "c"
