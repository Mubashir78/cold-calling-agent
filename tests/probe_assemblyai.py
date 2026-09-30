"""Does the AssemblyAI realtime backend actually work?

Drives `AssemblyAIRealtimeSTT` itself rather than a hand-rolled client, so a
pass means the production path works: auth, the v3 query-parameter handshake,
binary audio framing, and Turn parsing.

No LLM, no TTS, a few seconds of audio.

Note: chunks are paced at real time. The docs are explicit that streaming a
pre-recorded buffer in a tight loop produces inconsistent Turn messages, so a
tight loop here would test nothing.
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from stt.assemblyai import AssemblyAIRealtimeSTT  # noqa: E402

RATE = 16000
FRAME_MS = 100


def _tone(seconds: float, freq: float = 180.0) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    wave = 0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
    return (np.clip(wave, -1, 1) * 32767).astype(np.int16).tobytes()


def _silence(seconds: float) -> bytes:
    return b"\x00" * int(seconds * RATE * 2)


async def main() -> int:
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("FAIL: ASSEMBLYAI_API_KEY not set")
        return 1

    # A real recording is the honest test: a synthetic tone may simply not
    # register as speech, which tells us nothing about the backend.
    path = sys.argv[1] if len(sys.argv) > 1 else None

    stt = AssemblyAIRealtimeSTT(keyterms=["cold calling", "missed inbound calls"])
    await stt.initialize()

    frame = int(RATE * FRAME_MS / 1000) * 2

    async def audio():
        if path:
            import wave
            with wave.open(path, "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
            blocks = [pcm, _silence(2.0)]
            print(f"streaming {path} ({len(pcm) / (RATE * 2):.1f}s)")
        else:
            # Tone, then quiet: the quiet is what should end the turn.
            blocks = [_tone(2.5), _silence(2.0)]
        for block in blocks:
            for i in range(0, len(block), frame):
                yield block[i:i + frame]
                await asyncio.sleep(FRAME_MS / 1000)   # real time

    got: list[str] = []
    try:
        async for segment in stt.transcribe_stream(audio(), sample_rate=RATE):
            got.append(segment.text)
            print(f"  segment: {segment.text!r} "
                  f"({segment.start:.2f}-{segment.end:.2f}s)")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    if not got:
        print("FAIL: connected and streamed, but no Turn segments came back")
        print("      A synthetic tone may not register as speech. Before")
        print("      concluding it is broken, run it against a real recording:")
        print("      python tests/probe_assemblyai.py data/calls/*_prospect.wav")
        return 1

    print(f"\nOK: {len(got)} segment(s) via the production backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
