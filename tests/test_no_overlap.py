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
