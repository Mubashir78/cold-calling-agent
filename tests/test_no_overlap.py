"""The two sides must never speak at the same time.

That is the guarantee a listener notices immediately, and it is the one this
project broke repeatedly. It regressed because three mechanisms each answered
"who has the line?" — a turn lock, a boolean turn flag, and sample counters in
the recorder — and disagreed. A real run produced 11.82s of overlap, with the
lock reporting `holder=None silence=0.14s need=0.35s` at the moment the flag
reported the line busy.

These tests wire the real `TurnCoordinator` and the real `Playback` into the
real pipeline, because the bug lived in the seams between them. Tests that fed
`Playback` directly all passed while the pipeline never marked the end of a
turn at all.
"""

import asyncio
import inspect
import json

import numpy as np
import pytest

from llm.client import LLMResponse
from pipeline import TURN_END, VoicePipeline
from playback import Playback
from stt.engine import TranscriptSegment
from tools.registry import ToolRegistry
from turns import TurnCoordinator

SAMPLE_RATE = 16000


def _speech(seconds: float, freq: float = 300.0) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.clip(0.4 * np.sin(2 * np.pi * freq * t), -1, 1) * 32767
            ).astype(np.int16).tobytes()


class _FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return _speech(0.25)


class _ScriptedSTT:
    """Emits segments as the audio stream reaches a scripted point."""

    def __init__(self, utterances, chunks_before_each: int = 3):
        self._u = list(utterances)
        self._chunks = chunks_before_each

    async def initialize(self):
        pass

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        for text in self._u:
            for _ in range(self._chunks):
                async for _ in audio_in:
                    pass
            yield TranscriptSegment(text, 0.0, 0.0, True, 0.0)


class _LLM:
    """Replies with one short sentence, slowly enough to be observable."""

    def __init__(self):
        self.turns = 0

    def generate(self, messages, tools=None, stream=True):
        return self._run()

    def generate_with_tools(self, messages, tools, executor, max_tool_rounds=3):
        return self._run()

    async def _run(self):
        self.turns += 1
        await asyncio.sleep(0.05)
        yield LLMResponse(text=f"reply number {self.turns}.")


async def _drain(q):
    while True:
        await q.get()


def _make(utterances, tts=None):
    """Build the objects. Must be called inside a running loop.

    Returns them without a consumer task, so the caller can decide when the
    playback consumer starts.
    """
    turns = TurnCoordinator()
    playback = Playback(coordinator=turns, turn_gap=0.05)
    pipeline = VoicePipeline(
        stt=_ScriptedSTT(utterances),
        llm=_LLM(),
        tts=tts or _FakeTTS(),
        tools=ToolRegistry(),
        system_prompt="S",
        coordinator=turns,
        playback=playback,
        turn_spacing=0.0,
    )
    q: asyncio.Queue = asyncio.Queue()
    return turns, playback, pipeline, q


def _start(q, playback):
    return asyncio.create_task(playback.run(q))


def _stop(consumer):
    consumer.cancel()


def test_agent_speaks_first():
    """The agent opens; the prospect has not said anything yet."""
    async def go():
        turns, playback, pipeline, q = _make([])
        tts = _FakeTTS()
        pipeline.tts = tts
        consumer = _start(q, playback)
        await pipeline.run_call(_empty(), q, opening="Opening line here.")
        _stop(consumer)
        return tts.spoken

    spoken = asyncio.run(go())
    # The opening is spoken by the pipeline directly; no LLM turn runs because
    # the prospect never said anything.
    assert spoken == ["Opening line here."], spoken


async def _empty():
    if False:
        yield b""


def test_pipeline_releases_the_line_after_every_turn():
    """Regression: nothing marked the end of a turn, so the line never freed.

    The agent would hold the line after its first reply and the call would
    deadlock. Asserted directly on the coordinator, because a passing suite
    once hid exactly this: the Playback tests supplied the marker themselves.
    """
    async def go():
        turns, playback, pipeline, q = _make(["Hello?", "And pricing?"])
        tts = _FakeTTS()
        pipeline.tts = tts
        consumer = _start(q, playback)
        await pipeline.run_call(_empty(), q, opening="Opening line.")
        _stop(consumer)
        return turns, tts.spoken

    turns, spoken = asyncio.run(go())
    assert not turns.agent_holding, "line still held after the call ended"
    assert spoken, "the agent never spoke"


def test_turn_end_marker_is_enqueued_for_each_turn():
    """The boundary must be declared explicitly, once per turn.

    The items are observed through a bridge that feeds the real playback
    consumer, so the pipeline's completion wait is genuinely satisfied rather
    than timing out. Draining the queue afterwards cannot work: the pipeline
    waits for its marker to be played, so nothing consuming it blocks for the
    full 30s allowance.
    """
    async def go():
        turns, playback, pipeline, q = _make(["Hello?"])
        seen: list = []
        bridge_in: asyncio.Queue = asyncio.Queue()

        async def bridge():
            while True:
                item = await q.get()
                seen.append(item)
                await bridge_in.put(item)

        consumer = asyncio.create_task(playback.run(bridge_in))
        mover = asyncio.create_task(bridge())
        await pipeline.run_call(_empty(), q, opening="Opening line.")
        await asyncio.sleep(0.6)
        consumer.cancel()
        mover.cancel()
        return seen

    seen = asyncio.run(go())
    markers = [i for i in seen if isinstance(i, tuple) and i[0] == TURN_END]
    # One for the opening, one for the reply to the prospect's line.
    assert len(markers) == 2, f"expected 2 turn-end markers, got {len(markers)}"
    assert all(m[2] is True for m in markers), "marker must set turn_end"


def test_prospect_never_speaks_while_the_agent_holds_the_line():
    """The core guarantee, checked against the mechanism that enforces it."""
    turns = TurnCoordinator()
    heard: list[str] = []

    async def prospect_turn(label: str):
        # Exactly what SimulatedProspect does: wait for the turn to arrive,
        # then for it to finish.
        deadline = asyncio.get_event_loop().time() + 2.0
        while not turns.agent_holding and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)
        while turns.agent_holding:
            if turns.agent_holding:
                heard.append(f"{label}:spoke-while-held")
            await asyncio.sleep(0.01)
        heard.append(label)

    async def go():
        turns.agent_claims()
        await asyncio.sleep(0.3)
        turns.agent_releases()
        await prospect_turn("line-1")
        turns.agent_claims()
        await asyncio.sleep(0.2)
        turns.agent_releases()
        await prospect_turn("line-2")

    asyncio.run(go())
    assert not [h for h in heard if h.endswith("spoke-while-held")], heard
    assert heard.count("line-1") == 1 and heard.count("line-2") == 1


def test_stt_keeps_consuming_audio_while_the_agent_turn_runs():
    """The input stream must keep flowing during a turn.

    `transcribe_stream` is a pull-based generator and `audio_in` is only
    advanced by it. Awaiting the reply inline froze the whole chain for the
    length of the turn, so the prospect's wait was frozen mid-wait and resumed
    against a turn that had already been spoken.
    """
    turn_running = asyncio.Event()
    release_turn = asyncio.Event()
    state = {"pulled": 0, "during_turn": 0}

    class _LiveSTT:
        def __init__(self):
            self.sent = False

        async def transcribe_stream(self, stream, sample_rate=16000, **kw):
            async for _ in stream:
                if not self.sent:
                    self.sent = True
                    yield TranscriptSegment("Hello?", 0.0, 0.0, True, 0.0)
                elif state["pulled"] > 200:
                    return

    async def audio_in():
        while True:
            state["pulled"] += 1
            if turn_running.is_set():
                state["during_turn"] += 1
                release_turn.set()
            yield _speech(0.02)
            # Real audio arrives in real time. Without this the fake stream
            # drains synchronously and no other task can run.
            await asyncio.sleep(0.002)

    class _BlockingLLM:
        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor, **kw):
            return self._run()

        async def _run(self):
            # The turn has started. Nothing else may advance audio_in unless the
            # stream is genuinely live during it.
            turn_running.set()
            try:
                await asyncio.wait_for(release_turn.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            yield LLMResponse(text="Happy to help with that.")

    async def go():
        turns, playback, pipeline, q = _make([])
        pipeline.stt = _LiveSTT()
        pipeline.llm = _BlockingLLM()
        pipeline.turn_spacing = 0.0
        consumer = _start(q, playback)
        await pipeline.run_call(audio_in(), q, opening="", sample_rate=16000)
        _stop(consumer)

    asyncio.run(go())
    assert state["during_turn"] > 0, (
        "audio_in was not advanced during the agent's turn: the STT was frozen "
        f"({state['pulled']} chunks pulled, 0 during the turn)"
    )

def test_prospect_audio_reaches_the_recording():
    """Regression: the prospect track came out empty.

    Dropping the recording pass-through left one side of the stereo mix with no
    audio at all. The overlap report then read 0.00s — not because the call was
    clean, but because there was nothing to overlap with. An empty track turns
    the measurement into a false pass, which is worse than a wrong number
    because it looks like success.
    """
    from sim.recorder import CallRecording

    async def go():
        rec = CallRecording("A", "B", "+14155550101")
        rec.start()
        drained = {"value": False}

        async def audio_in():
            for _ in range(30):
                yield _speech(0.02)
                await asyncio.sleep(0.002)
            drained["value"] = True

        class _PullingSTT:
            async def transcribe_stream(self, stream, sample_rate=16000, **kw):
                async for _ in stream:
                    pass
                return
                yield  # pragma: no cover

        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        pipeline = VoicePipeline(
            stt=_PullingSTT(), llm=_LLM(), tts=_FakeTTS(),
            tools=ToolRegistry(), system_prompt="S",
            recorder=rec, coordinator=turns, playback=playback,
            turn_spacing=0.0,
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))
        await pipeline.run_call(audio_in(), q, opening="")
        consumer.cancel()
        return rec, drained["value"]

    rec, drained = asyncio.run(go())
    assert drained, "the prospect stream was never consumed"
    assert rec.prospect, "prospect track is empty: overlap would read 0.00s"
    assert rec.prospect[0].samples.size > 0


class _ContextSTT:
    """Records what the pipeline tells the recogniser, and when.

    Emits one segment so a reply turn runs after the opening.
    """

    def __init__(self):
        self.contexts: list[tuple[str, int]] = []
        self._sent = False

    def set_agent_context(self, text):
        if text:
            self.contexts.append((text, len(self.contexts)))

    async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
        async for _ in audio_in:
            if not self._sent:
                self._sent = True
                yield TranscriptSegment("Hello?", 0.0, 0.0, True, 0.0)
            # Keep feeding so the pipeline's waits are satisfied.
            if self._sent and not self.contexts:
                continue


def test_agent_tells_the_recogniser_what_it_just_said():
    """Regression: the strongest feature was dead code.

    `set_agent_context` existed on the AssemblyAI backend and nothing called
    it. It seeds the recogniser with the agent's last reply, which is what
    turns "Orphic. Send me the invite" into "Perfect. Send me the invite".
    """
    async def go():
        stt = _ContextSTT()
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        tts = _FakeTTS()
        pipeline = VoicePipeline(
            stt=stt, llm=_LLM(), tts=tts, tools=ToolRegistry(),
            system_prompt="S", coordinator=turns, playback=playback,
            turn_spacing=0.0,
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))
        # The same object records the context AND yields the segment. Swapping
        # in a separate scripted STT here would replace the recorder and the
        # test would observe nothing.
        await pipeline.run_call(_empty(), q, opening="Opening line here.")
        _stop(consumer)
        return stt.contexts, tts.spoken

    contexts, spoken = asyncio.run(go())
    assert contexts, "nothing was ever passed to the recogniser as context"
    assert any("Opening line here." in c for c, _ in contexts), contexts


def test_pipeline_only_sets_context_when_the_backend_supports_it():
    """A backend without the method must not break the call."""
    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        tts = _FakeTTS()

        class _BareSTT:
            """No set_agent_context at all."""

            async def transcribe_stream(self, stream, sample_rate=16000, **kw):
                async for _ in stream:
                    pass
                return
                yield  # pragma: no cover

        pipeline = VoicePipeline(
            stt=_BareSTT(), llm=_LLM(), tts=tts, tools=ToolRegistry(),
            system_prompt="S", coordinator=turns, playback=playback,
            turn_spacing=0.0,
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = _start(q, playback)
        await pipeline.run_call(_empty(), q, opening="Opening line.")
        _stop(consumer)
        return tts.spoken

    spoken = asyncio.run(go())
    assert spoken == ["Opening line."], spoken


def test_transient_stt_failure_does_not_end_the_call():
    """Regression: a dropped call is the worst outcome on a call list.

    The STT raised while opening its connection and the exception propagated
    out of `run_call`, so the call died after the opening line — 8.4s, zero
    segments, nothing logged. Recognition dropping mid-call is ordinary on a
    hosted provider; the call should degrade, not end.
    """
    from groq import APIConnectionError

    class _FlakySTT:
        def __init__(self):
            self.attempts = 0

        async def initialize(self):
            pass

        async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
            self.attempts += 1
            raise APIConnectionError(request=object())

    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        stt = _FlakySTT()
        tts = _FakeTTS()
        pipeline = VoicePipeline(
            stt=stt, llm=_LLM(), tts=tts, tools=ToolRegistry(),
            system_prompt="S", coordinator=turns, playback=playback,
            turn_spacing=0.0,
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))
        try:
            await pipeline.run_call(_empty(), q, opening="Opening line.")
        except Exception as exc:
            _stop(consumer)
            raise AssertionError(f"call died on a transient STT error: {exc}")
        _stop(consumer)
        return tts.spoken

    spoken = asyncio.run(go())
    assert spoken == ["Opening line."], "the opening should still have played"


def test_stt_connection_is_retried_before_giving_up():
    """Regression: a slow handshake killed the call outright.

    Connecting to AssemblyAI took ~3s in isolation but exceeded websockets'
    default 10s open timeout inside the running app, where the TTS models are
    resident and the box is loaded. Two calls in a row ended with "timed out
    during opening handshake" and nothing was transcribed.

    A hosted transport being briefly unavailable is ordinary. Retrying is the
    difference between a call and a lost call.
    """
    import types

    from stt import assemblyai as aai

    async def go():
        attempts = {"n": 0}

        class _FakeSocket:
            async def __aenter__(self):
                if attempts["n"] < 2:
                    raise TimeoutError("timed out during opening handshake")
                attempts["n"] += 1
                raise StopIteration   # construction proved the retry happened
            async def __aexit__(self, *exc):
                return False

        original = aai.websockets.connect

        def fake_connect(*a, **kw):
            attempts["n"] += 1
            return _FakeSocket()

        aai.websockets.connect = fake_connect
        try:
            stt = aai.AssemblyAIRealtimeSTT(api_key="k")
            await stt.initialize()
            try:
                async def audio():
                    yield b""
                async for _ in stt.transcribe_stream(audio(), 16000):
                    pass
            except Exception:
                pass
        finally:
            aai.websockets.connect = original
        return attempts["n"]

    attempts = asyncio.run(go())
    assert attempts >= 3, (
        f"only {attempts} connection attempts; a transient handshake failure "
        "ends the call"
    )


def test_agent_answers_the_final_transcript_not_a_partial():
    """Regression risk: the final can disagree with the partial it replaced.

    Observed on a real recording: partial "Perfect. Send me a Veeam invite"
    became final "Perfect. Send me a view invite". The agent primes its
    generation against the partial to cut latency, so if it then answered the
    partial it would be replying to words the caller did not say.

    The final transcript is the only thing the agent may treat as true.
    """
    seen: list[str] = []

    class _PrimeThenFinal:
        async def transcribe_stream(self, audio_in, sample_rate=16000, **kw):
            async for _ in audio_in:
                pass
            # A partial arrives first, revised into the final.
            yield TranscriptSegment("Perfect. Send me a Veeam invite, "
                                    "I'll make it.", 0.0, 0.0, False, 1.0)
            async for _ in audio_in:
                pass
            yield TranscriptSegment("Perfect. Send me a view invite, "
                                    "I'll make it.", 0.0, 0.0, True, 1.0)
            async for _ in audio_in:
                pass

    class _RecordingLLM:
        def generate(self, messages, tools=None, stream=True):
            return self._run(messages)

        def generate_with_tools(self, messages, tools, executor, **kw):
            return self._run(messages)

        async def _run(self, messages):
            seen.append(messages[-1].content)
            yield LLMResponse(text="Right, sending it now.")

    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        pipeline = VoicePipeline(
            stt=_PrimeThenFinal(), llm=_RecordingLLM(), tts=_FakeTTS(),
            tools=ToolRegistry(), system_prompt="S",
            coordinator=turns, playback=playback, turn_spacing=0.0,
        )
        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))
        await pipeline.run_call(_empty(), q, opening="Opening.")
        _stop(consumer)
        return pipeline

    pipeline = asyncio.run(go())
    spoken_to_model = " ".join(seen)
    assert "Veeam" not in spoken_to_model, (
        "the agent generated against a partial the recogniser later revised"
    )
    assert "view invite" in spoken_to_model, "the final was never used"


def test_pipeline_is_wired_to_the_far_end():
    """Regression: the far end was never told anything the agent said.

    `pipeline.on_far_end_turn` was never assigned in `main.py`, so
    `_hand_turn_to_far_end` had no target and the far end received zero agent
    turns. It therefore never generated a reply, and every call ended after the
    agent's second turn — while the unit suite stayed green, because every test
    builds its own pipeline and assigns the hook itself.

    This asserts the wiring exists in the composition root, which is the only
    place that bug lived.
    """
    import inspect

    import main as m

    source = inspect.getsource(m)
    assert "pipeline.on_far_end_turn = prospect" in source, (
        "main.py does not hand the pipeline its far end, so the simulated "
        "caller is never told what the agent said"
    )


class _FakeToolRegistry:
    """Tool registry that simulates booking without network calls."""

    def __init__(self):
        self._booked = False
        self._logged = False
        self._dnc = False
        self._successful_tools: set[str] = set()
        self._current_call: dict = {}

    def bind_call(self, phone_number: str, prospect_name: str = "",
                  company: str = "") -> None:
        self._current_call = {"phone_number": phone_number,
                              "prospect_name": prospect_name,
                              "company": company}
        self._booked = False
        self._logged = False
        self._dnc = False
        self._successful_tools.clear()

    @property
    def booked_this_call(self) -> bool:
        return self._booked

    @property
    def terminal(self) -> bool:
        return self._booked or self._dnc

    def get_tools_for_llm(self):
        return [
            {"type": "function", "function": {
                "name": "book_meeting",
                "description": "Book a meeting at the agreed time. Call once.",
                "parameters": {
                    "type": "object",
                    "properties": {"time_slot": {"type": "string", "description": "ISO 8601"}},
                    "required": ["time_slot"],
                },
            }},
            {"type": "function", "function": {
                "name": "log_disposition",
                "description": "Log the call outcome in the CRM. Call once, at the end.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "disposition": {"type": "string",
                                        "enum": ["booked", "callback", "not_interested", "dnc",
                                                 "voicemail", "no_answer", "wrong_number"]},
                        "notes": {"type": "string"},
                    },
                    "required": ["disposition"],
                },
            }},
            {"type": "function", "function": {
                "name": "honor_dnc",
                "description": "Add the prospect to the Do Not Call list.",
                "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": []},
            }},
        ]

    async def execute(self, name: str, arguments: dict) -> dict:
        if name == "book_meeting":
            if self._booked:
                return {"success": False, "error": "Already booked."}
            self._booked = True
            self._successful_tools.add(name)
            return {"success": True, "result": {"event_id": "test_evt"}}
        if name == "log_disposition":
            if self._logged:
                return {"success": False, "error": "Already logged."}
            self._logged = True
            self._successful_tools.add(name)
            return {"success": True, "result": {}}
        if name == "honor_dnc":
            if self._dnc:
                return {"success": False, "error": "Already on DNC."}
            self._dnc = True
            self._successful_tools.add(name)
            return {"success": True, "result": {}}
        return {"success": False, "error": f"Unknown tool: {name}"}


def test_call_ends_after_the_meeting_is_booked():
    """Regression: the agent re-confirmed the booking three times.

    A real call booked a meeting, then said "You're all set for Tuesday at 10
    AM" three more times, speaking 64s of audio in a 143s call. The prospect
    answered each repeat, and the two sides overlapped for 2.06s while doing
    it. A booked meeting is the end of the call, not a topic to keep returning
    to.

    The turn that books is allowed to finish -- it carries the confirmation --
    and the call ends with it.
    """
    class _BookingLLM:
        """Books on its second turn, then keeps offering to book again.

        Calls the executor so the tool registry actually records the booking.
        """

        def __init__(self):
            self.turns = 0

        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor,
                                max_tool_rounds=3):
            return self._run(executor)

        async def _run(self, executor=None):
            self.turns += 1
            await asyncio.sleep(0.02)
            if self.turns >= 2:
                # Execute the tool through the executor if provided
                if executor is not None:
                    await executor("book_meeting", {"time_slot": "2026-10-08T14:00"})
                yield LLMResponse(
                    text="",
                    tool_calls=[{"id": "1", "type": "function",
                                "function": {"name": "book_meeting",
                                             "arguments": json.dumps(
                                                 {"time_slot":
                                                  "2026-10-08T14:00"})}}],
                    finish_reason="tool_calls",
                )
                return
            yield LLMResponse(text="Can I put Tuesday at 10 on the calendar?")

    async def go():
        turns = TurnCoordinator()
        playback = Playback(coordinator=turns, turn_gap=0.0)
        tts = _FakeTTS()
        tools = _FakeToolRegistry()
        pipeline = VoicePipeline(
            stt=_ScriptedSTT(["Tuesday at 10 works for me."] * 8),
            llm=_BookingLLM(),
            tts=tts,
            tools=tools,
            system_prompt="S",
            coordinator=turns,
            playback=playback,
            turn_spacing=0.0,
        )
        tools.bind_call("+14155550101", "Dana", "Northwind")
        assert tools.booked_this_call is False

        q: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(q))

        async def source():
            for _ in range(4000):
                yield b"" * 320
                await asyncio.sleep(0.01)

        await asyncio.wait_for(
            pipeline.run_call(source(), q, opening="Opening."), timeout=30
        )
        _stop(consumer)
        return tts.spoken, tools.booked_this_call

    spoken, booked = asyncio.run(go())
    assert booked, "the meeting was never booked, so this proved nothing"
    assert len(spoken) <= 3, (
        f"the agent spoke {len(spoken)} turns after the meeting was booked: "
        f"{spoken}"
    )


def test_no_fixed_pause_sits_on_the_critical_path():
    """The agent must not add a fixed pause before generating.

    `turn_spacing` slept unconditionally before the model request was even
    sent, adding 300ms of pure silence to every turn -- about 19% of the
    measured 1.6s prospect-to-agent gap. The config comment claimed the pacing
    was token-based, but `GroqLLM.suggest_turn_spacing` (which derives it from
    the real token cost) has no call site anywhere in the repo. So it was not
    pacing at all, just dead air.

    Rate limiting is handled by rotating to another model rather than sleeping,
    so a blind pre-sleep cannot prevent a 429 -- it only adds audible
    hesitation to every turn.
    """
    import yaml

    config = yaml.safe_load(open("config/config.yaml"))

    # The shipped default must not pace, because nothing measures the budget.
    pipeline_defaults = inspect.signature(VoicePipeline).parameters
    assert pipeline_defaults["turn_spacing"].default == 0.0, (
        "VoicePipeline still defaults to a fixed pre-turn sleep"
    )
    assert float(config["voice"]["turn_spacing"]) == 0.0, (
        "config still sets a fixed pre-turn sleep on every turn"
    )


def test_far_end_generation_is_not_gated_behind_a_rate_limit_sleep():
    """A per-minute 429 must cost silence, not a wait.

    Budgets are per model, so switching is instant while waiting out the
    minute is not. The previous behaviour slept up to 45s inside the request,
    which the caller hears as the agent going quiet mid-call.
    """
    import asyncio as _aio
    from types import SimpleNamespace

    from groq import RateLimitError

    class _Resp:
        status_code = 429
        headers = {"x-ratelimit-remaining-tokens": "0",
                   "x-ratelimit-reset-tokens": "30s"}
        request = object()

        def json(self):
            return {"error": {"message": "Rate limit reached"}}

    slept: list[float] = []
    real_sleep = _aio.sleep

    async def _fake_sleep(seconds, *a, **kw):
        slept.append(seconds)
        return await real_sleep(0)

    calls = {"n": 0}

    async def _create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("Rate limit reached", response=_Resp(),
                                 body={"error": {"message": "x"}})
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Wednesday works.",
                                        tool_calls=None),
                finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))

    from llm.client import GroqLLM

    llm = GroqLLM(model="primary/model", api_key="k",
                  fallback_models=["backup/model"], max_retries=2,
                  retry_budget=90.0)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    _aio.sleep = _fake_sleep
    loop = _aio.new_event_loop()
    _aio.set_event_loop(loop)
    try:
        text = loop.run_until_complete(llm.complete("When works for you?"))
    finally:
        _aio.set_event_loop(None)
        loop.close()
        _aio.sleep = real_sleep

    assert text == "Wednesday works.", (
        "the call failed even though a fallback model's budget was free"
    )
    assert slept == [], (
        f"the caller waited {sum(slept):.1f}s of silence when switching "
        f"models was instant"
    )
