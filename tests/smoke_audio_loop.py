"""Smoke test: does Piper audio survive the whisper round trip?

Synthesizes speech with the prospect voice, pushes it through the exact STT
streaming path the pipeline uses, and asserts the transcript is recoverable.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sim.prospect import CHUNK_BYTES, SILENCE
from stt.engine import StreamingSTT
from tts.engine import PiperTTS, TTSConfig

PHRASES = [
    "Yeah, Tuesday afternoon after two works for me.",
    "Please take me off your calling list.",
    "Northwind Logistics, how may I direct your call?",
]


async def main() -> int:
    tts = PiperTTS(TTSConfig(voice="en_US-ryan-medium", speaking_rate=1.0))
    await tts.initialize()

    stt = StreamingSTT(model_size="small", device="cpu", compute_type="int8")
    await stt.initialize()

    failures = 0
    for phrase in PHRASES:
        pcm = await tts.synthesize_pcm(phrase)
        duration = len(pcm) / (16000 * 2)
        print(f"\nsynthesized {len(pcm)} bytes ({duration:.2f}s) for: {phrase}")

        async def stream():
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]
            for _ in range(50):  # 1s of trailing silence to trigger endpointing
                yield SILENCE

        heard = []
        async for seg in stt.transcribe_stream(stream(), sample_rate=16000):
            heard.append(seg.text)
            print(f"  transcript: {seg.text!r}")

        if not heard:
            print("  NO TRANSCRIPT")
            failures += 1

    print("\n" + ("PASS" if not failures else f"FAIL ({failures} phrases lost)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
