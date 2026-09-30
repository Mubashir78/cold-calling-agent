"""The two sides must never speak at the same time.

This is the guarantee a listener notices immediately, and it was unguarded
in one direction: the agent began replying the moment STT returned a segment,
with no check that the prospect had actually stopped talking.
"""

import asyncio

import numpy as np
import pytest

from pipeline import VoicePipeline
from stt.engine import TranscriptSegment
from turnlock import Speaker, TurnLock


def _pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


def _speech(seconds: float) -> bytes:
    t = np.arange(int(seconds * 16000)) / 16000.0
    return _pcm(0.4 * np.sin(2 * np.pi * 300 * t))


def _silence(seconds: float) -> bytes:
    return b"\x00" * int(seconds * 16000 * 2)


class _FakeTTS:
    def __init__(self):
        self.spoken = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return _speech(0.4)


class _ScriptedSTT:
    """Emits segments, and records when each was produced."""

    def __init__(self, utterances):
        self._u = list(utterances)
        self.emitted = []

    async def initialize(self):
        pass

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        async for chunk in audio_in:
            pass
        for text in self._u:
            self.emitted.append(text)
            yield TranscriptSegment(text, 0.0, 0.0, True, 0.0)


class _LLM:
    def __init__(self, n=20):
        self.n = n
        self._i = 0

    def generate(self, messages, tools=None, stream=True):
        return self._run()

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._run()

    async def _run(self):
        from llm.client import LLMResponse
        if self._i >= self.n:
            return
        self._i += 1
        yield LLMResponse(text=f"reply {self._i}")


def _pipeline(lock, stt=None, tts=None):
    from tools.registry import ToolRegistry
    return VoicePipeline(
        stt=stt or _ScriptedSTT(["Hello?", "And pricing?"]),
        llm=_LLM(),
        tts=tts or _FakeTTS(),
        tools=ToolRegistry(),
        system_prompt="S",
        lock=lock,
        # No inter-turn gap here: these tests measure the silence check, and a
        # deliberate pause would dominate the timing assertions.
        turn_gap=0.0,
    )


async def _drain(q):
    while True:
        await q.get()


async def _empty():
    return
    yield b""


async def test_opening_bypasses_the_silence_requirement():
    """The opening cannot wait for silence: nobody has spoken yet.

    Regression: requiring silence on the opening deadlocked the first turn,
    and the agent burned the whole lock timeout before speaking.
    """
    lock = TurnLock(silence_required=0.3, timeout=0.5)
    tts = _FakeTTS()
    p = _pipeline(lock, tts=tts)

    loop = asyncio.get_event_loop()
    start = loop.time()
    # The caller owns the line; `_speak` no longer touches the lock.
    await lock.acquire(Speaker.AGENT, bypass_silence=True)
    await p._speak("Hi, this is an AI call.")
    elapsed = loop.time() - start

    assert tts.spoken, "opening line was never spoken"
    # Must not wait for silence that will never come, but the inter-turn gap
    # is deliberate and always applies.
    assert elapsed < 0.25 + lock.timeout / 10, (
        f"opening waited {elapsed:.2f}s; it should bypass the silence check"
    )
    assert not any("timeout" in o for o in lock.overlaps)


async def test_reply_waits_for_the_prospect_to_finish():
    """The hazard: a reply that starts mid-sentence."""
    lock = TurnLock(silence_required=0.3, timeout=0.5)
    tts = _FakeTTS()
    p = _pipeline(lock, tts=tts)

    lock.note_speech()      # prospect is talking
    lock.note_silence(0.05)  # a breath, not a real pause

    await lock.acquire(Speaker.AGENT)
    await p._speak("Sure, here's what we do.")

    assert any("timeout" in o for o in lock.overlaps), (
        "agent took the line on a 0.05s breath, mid-sentence"
    )


async def test_reply_proceeds_after_real_silence():
    lock = TurnLock(silence_required=0.3, timeout=1.0)
    tts = _FakeTTS()
    p = _pipeline(lock, tts=tts)
    p.turn_gap = 0.05

    lock.note_speech()
    lock.note_silence(0.6)  # genuine pause

    loop = asyncio.get_event_loop()
    start = loop.time()
    await lock.acquire(Speaker.AGENT)
    await p._speak("Sure, here's what we do.")
    elapsed = loop.time() - start

    assert tts.spoken
    # No forced timeout, and no waiting on silence that was already given.
    assert not any("timeout" in o for o in lock.overlaps)
    assert elapsed < p.turn_gap + 0.2, (
        f"waited {elapsed:.2f}s despite 0.6s of silence"
    )


async def test_lock_is_released_after_the_agent_finishes():
    lock = TurnLock()
    p = _pipeline(lock)
    await p._speak("Hello there.")
    assert lock.is_free(), "agent kept the line after finishing"


async def test_prospect_cannot_take_a_busy_line():
    lock = TurnLock(silence_required=0.2, timeout=0.3)
    await lock.acquire(Speaker.AGENT, require_silence=False)
    await lock.acquire(Speaker.PROSPECT, require_silence=False)
    assert any("timeout" in o for o in lock.overlaps), (
        "prospect barged straight into a held line"
    )


async def test_prospect_waits_through_model_generation():
    """The race that caused talking over each other.

    The agent spends 1-2s generating before any audio exists. If the line is
    only claimed while synthesizing, the prospect sees an idle line during
    generation and starts talking over the reply.
    """
    from sim.prospect import SimulatedProspect, script_for

    prospect = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)
    spoke_during_generation = []

    async def fake_llm():
        # The agent claims the line, then takes time to think.
        prospect.agent_started_speaking()
        for _ in range(12):
            if not prospect.agent_turn_active:
                spoke_during_generation.append(False)
            await asyncio.sleep(0.02)
        prospect.agent_finished_speaking()

    async def consume():
        async for _ in prospect._wait_for_agent_chunks(0):
            pass

    async def go():
        await asyncio.gather(consume(), fake_llm())

    await go()

    assert not spoke_during_generation, (
        "prospect considered talking while the agent was generating"
    )
    assert prospect.agent_turn_active is False


async def test_prospect_talks_once_the_turn_is_released():
    from sim.prospect import SimulatedProspect, script_for

    prospect = SimulatedProspect(script_for("interested"), tts=None, turn_gap=0.0)
    prospect.agent_started_speaking()

    chunks = []

    async def consume():
        async for c in prospect._wait_for_agent_chunks(0):
            chunks.append(c)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    assert chunks, "prospect stopped feeding the STT while the agent held the line"
    prospect.agent_finished_speaking()
    await asyncio.wait_for(task, timeout=2.0)


async def test_generation_window_is_covered_by_the_turn_flag():
    """The flag must be set before the model is called, not during TTS."""
    order = []

    class _LLM:
        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
            return self._run()

        async def _run(self):
            from llm.client import LLMResponse
            order.append(("generated", p._agent_turn_active))
            yield LLMResponse(text="One two three.")

    p = _pipeline(None, tts=_FakeTTS())
    p.llm = _LLM()
    p._respond_started = []
    original = p._begin_agent_turn

    def spy():
        original()
        order.append(("turn_claimed", p._agent_turn_active))

    p._begin_agent_turn = spy

    q = asyncio.Queue()
    task = asyncio.create_task(_drain(q))
    try:
        await p._respond("Tuesday at two works.")
    finally:
        task.cancel()

    assert order[0][0] == "turn_claimed", (
        f"turn not claimed before generation: {order}"
    )
    assert order[1] == ("generated", True), (
        f"line was not held during generation: {order}"
    )


async def test_speech_prevents_the_agent_from_taking_the_line():
    """Short silence must not hand the line over mid-sentence."""
    lock = TurnLock(silence_required=0.35, timeout=0.4)
    lock.note_speech()
    lock.note_silence(0.10)
    await lock.acquire(Speaker.AGENT)
    assert any("timeout" in o for o in lock.overlaps), (
        "agent took the line on a 0.10s breath"
    )


def test_concurrent_sides_never_both_hold():
    """Run both at once and assert mutual exclusion held throughout."""
    violations = []
    lock = TurnLock(timeout=0.15)

    async def side(name, chunks):
        await lock.acquire(name, require_silence=False)
        try:
            for _ in range(chunks):
                if lock.holder != name:
                    violations.append((name.value, lock.holder))
                await asyncio.sleep(0.01)
        finally:
            lock.release(name)

    async def go():
        await asyncio.gather(
            side(Speaker.AGENT, 5),
            side(Speaker.PROSPECT, 5),
        )

    asyncio.run(go())
    assert not violations, f"both sides held the line: {violations}"


def test_silence_tracking_detects_speech():
    lock = TurnLock()
    assert not lock.is_quiet()
    lock.note_silence(0.5)
    assert lock.is_quiet()
    lock.note_speech()
    assert not lock.is_quiet(), "speech did not clear the silence flag"


def test_stt_keeps_consuming_audio_while_the_agent_turn_runs():
    """Regression: ~12s of dead air between every single turn.

    `transcribe_stream` is a pull-based async generator, and `audio_in` is
    only advanced by it. Awaiting the reply *inside* the `async for` froze the
    entire chain for the length of the agent's turn — generation plus the full
    playback. The prospect's generator was frozen mid-wait, so when it resumed
    the agent's turn was already over, `agent_turn_active` read False, and it
    sat out the whole wait timeout waiting for a turn that had already been
    spoken. That is the gap a listener hears.

    The reply must therefore run as a task while this loop keeps pulling.
    """
    from llm.client import LLMResponse
    from tools.registry import ToolRegistry

    async def go():
        turn_running = asyncio.Event()
        turn_done = asyncio.Event()
        release_turn = asyncio.Event()
        state = {"pulled": 0, "during_turn": 0}

        async def audio_in():
            while True:
                state["pulled"] += 1
                if turn_running.is_set():
                    state["during_turn"] += 1
                    # The stream is only advanced by whoever pulls it, so this
                    # firing proves the STT was not frozen.
                    release_turn.set()
                yield _speech(0.02)
                # Real audio arrives in real time. Without this the fake
                # stream drains synchronously and no other task can run.
                await asyncio.sleep(0.002)

        class _LiveSTT:
            """Emits one segment, then keeps draining like a real STT."""

            def __init__(self):
                self.sent = False
                self.tail = 0

            async def transcribe_stream(self, stream, sample_rate=16000, **kw):
                async for _ in stream:
                    if not self.sent:
                        self.sent = True
                        yield TranscriptSegment("Hello?", 0.0, 0.0, True, 0.0)
                    elif turn_done.is_set():
                        self.tail += 1
                        if self.tail > 20:
                            return

        class _BlockingLLM:
            """Blocks until something else proves the stream still moves."""

            def generate(self, messages, tools=None, stream=True):
                return self._run()

            def generate_with_tools(self, messages, tools, executor,
                                    max_tool_rounds=3):
                return self._run()

            async def _run(self):
                try:
                    await asyncio.wait_for(release_turn.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                yield LLMResponse(text="Happy to help with that.")

        p = VoicePipeline(
            stt=_LiveSTT(),
            llm=_BlockingLLM(),
            tts=_FakeTTS(),
            tools=ToolRegistry(),
            system_prompt="S",
            turn_spacing=0.0,
            turn_gap=0.0,
        )
        inner_respond = p._respond

        async def respond(text=""):
            turn_running.set()
            try:
                return await inner_respond(text)
            finally:
                turn_running.clear()
                turn_done.set()

        p._respond = respond

        q: asyncio.Queue = asyncio.Queue()

        async def drain():
            while True:
                await q.get()

        consumer = asyncio.create_task(drain())
        try:
            await p.run_call(audio_in(), q, opening="", sample_rate=16000)
        finally:
            consumer.cancel()
        return state

    state = asyncio.run(go())
    assert state["during_turn"] > 0, (
        "audio_in was not advanced during the agent's turn: the STT was "
        f"frozen ({state['pulled']} chunks pulled, 0 during the turn)"
    )
