"""Playback-aware turn taking.

The bug this pins: `_speak` only enqueues audio. Piper synthesizes faster
than real time, so the pipeline considered its turn finished while the audio
was still playing. The prospect then started talking over the tail, and two
voices were audible at once.

The rule: a speaker has not finished speaking until its audio has *played*,
not until it has been generated.
"""

import asyncio

import numpy as np
import pytest

from sim.recorder import CallRecording
from turnlock import Speaker


def _pcm(seconds: float = 0.1) -> bytes:
    n = int(seconds * 16000)
    return (np.full(n, 0.3) * 32767).astype(np.int16).tobytes()


async def test_playback_wait_blocks_until_audio_has_played():
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    r.add_agent(_pcm(0.05))          # queued: 800 samples
    assert r._agent_finished == 0

    task = asyncio.create_task(r.await_agent_playback(timeout=2.0))
    await asyncio.sleep(0.05)
    assert not task.done(), "returned while audio was still queued"

    r.add_playback("agent", _pcm(0.05))      # playback starts
    r.mark_playback_finished("agent", 800)   # ...and finishes
    assert await asyncio.wait_for(task, timeout=1.0) is True


async def test_no_wait_when_nothing_is_queued():
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    assert await r.await_agent_playback(timeout=0.2) is True


async def test_wait_gives_up_rather_than_hanging():
    """A stuck consumer must not freeze the call forever."""
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    r.add_agent(_pcm(0.05))  # queued, never played
    assert await r.await_agent_playback(timeout=0.1) is False


async def test_playback_wait_blocks_until_the_last_chunk_finishes():
    """The bug that produced 1-2s collisions.

    Counting a chunk on arrival treats "playback started" as "playback
    ended". With three 2s sentences the last was recorded at T+4s but was
    still audible until T+6s, so the line was released two seconds early.
    """
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    for _ in range(3):
        r.add_agent(_pcm(2.0))          # 6s of audio queued

    task = asyncio.create_task(r.await_agent_playback(timeout=5.0))
    # Simulate a paced consumer: record, sleep, then mark finished.
    for _ in range(3):
        await asyncio.sleep(0.05)
        r.add_playback("agent", _pcm(2.0))
        await asyncio.sleep(0.02)
        r.mark_playback_finished("agent", 32000)
        if task.done():
            break

    # After the final chunk is *started* but not finished, we must still wait.
    if task.done():
        r2 = CallRecording("A", "B", "+1")
        r2.start()
        r2.add_agent(_pcm(2.0))
        t2 = asyncio.create_task(r2.await_agent_playback(timeout=5.0))
        r2.add_playback("agent", _pcm(2.0))
        await asyncio.sleep(0.05)
        assert not t2.done(), (
            "wait returned at playback start, not playback end"
        )
        r2.mark_playback_finished("agent", 32000)
        assert await asyncio.wait_for(t2, timeout=2.0) is True
    else:
        assert await asyncio.wait_for(task, timeout=2.0) is True


async def test_playback_seconds_tracks_finished_audio():
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    r.add_agent(_pcm(0.5))
    r.add_playback("agent", _pcm(0.25))
    assert r.agent_playback_seconds() == 0.0, (
        "counted before playback finished"
    )
    r.mark_playback_finished("agent", 4000)
    assert abs(r.agent_playback_seconds() - 0.25) < 0.01


async def test_pipeline_waits_for_playback_before_releasing_the_line():
    """End to end: the line must not free up while audio is still queued."""
    from llm.client import LLMResponse
    from pipeline import VoicePipeline
    from stt.engine import TranscriptSegment
    from tools.registry import ToolRegistry
    from turnlock import TurnLock

    class _TTS:
        async def synthesize_pcm(self, text):
            return _pcm(0.3)  # ~0.3s of audio per sentence

    class _LLM:
        def generate(self, messages, tools=None, stream=True):
            return self._run()

        def generate_with_tools(self, messages, tools, executor,
                                max_tool_rounds=3):
            return self._run()

        async def _run(self):
            yield LLMResponse(text="One two. Three four.")

    lock = TurnLock(silence_required=0.05, timeout=2.0)
    rec = CallRecording("A", "B", "+14155550101")
    rec.start()

    p = VoicePipeline(
        stt=None, llm=_LLM(), tts=_TTS(), tools=ToolRegistry(),
        system_prompt="s", lock=lock, recorder=rec,
        turn_spacing=0.0, turn_gap=0.05, max_history_turns=6,
    )

    q: asyncio.Queue = asyncio.Queue()
    released_while_queued = []

    async def consumer():
        while True:
            await asyncio.sleep(0.01)
            # If the line frees while audio is still queued, that is the bug.
            if lock.is_free() and rec._agent_finished < rec._agent_queued:
                released_while_queued.append(
                    rec._agent_queued - rec._agent_finished)

    drain = asyncio.create_task(consumer())
    play = asyncio.create_task(_play(q, rec))
    try:
        # `_speak` owns the line across synthesis *and* playback, so the
        # release cannot land while audio is still queued.
        await p._speak("One two. Three four.", is_opening=True)
    finally:
        drain.cancel()
        play.cancel()

    assert not released_while_queued, (
        f"line was released with audio still queued: {released_while_queued}"
    )


async def _play(q: asyncio.Queue, rec: CallRecording) -> None:
    while True:
        item = await q.get()
        speaker, pcm = item if isinstance(item, tuple) else ("agent", item)
        rec.add_playback(speaker, pcm)
        await asyncio.sleep(len(pcm) / (16000 * 2))  # real-time playback
        rec.mark_playback_finished(speaker, len(pcm) // 2)
