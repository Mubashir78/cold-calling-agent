"""Smoke test: Piper audio -> Groq whisper transcription.

Confirms the audio we synthesize is actually intelligible to the STT layer,
which is the one link that cannot be verified by inspection.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from sim.prospect import CHUNK_BYTES, SILENCE
from stt.groq_stt import GroqSTT
from tts.engine import PiperTTS, TTSConfig

load_dotenv(Path(__file__).parent.parent / ".env")

PHRASES = [
    "Yeah, Tuesday afternoon after two works for me.",
    "Please take me off your calling list.",
    "Northwind Logistics, how may I direct your call?",
]


async def main() -> int:
    import os

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("FAIL: GROQ_API_KEY not set")
        return 1

    tts = PiperTTS(TTSConfig(voice="en_US-ryan-medium"))
    await tts.initialize()

    stt = GroqSTT(model="whisper-large-v3-turbo", api_key=key)
    await stt.initialize()

    lost = 0
    for phrase in PHRASES:
        pcm = await tts.synthesize_pcm(phrase)
        print(f"\nsent: {phrase!r}  ({len(pcm) / 32000:.2f}s)")

        async def stream():
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]
            for _ in range(50):
                yield SILENCE

        heard = [s.text async for s in stt.transcribe_stream(stream(), sample_rate=16000)]
        print(f"heard: {heard}")

        if not heard:
            print("  LOST")
            lost += 1

    print("\n" + ("PASS" if not lost else f"FAIL: {lost} phrase(s) lost"))
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
