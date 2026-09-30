"""Does the degraded phone-line audio still transcribe?

Runs the same phrases through the full conditioning chain and scores them
against the known source text, so intelligibility is a number rather than an
opinion. Whisper is a forgiving model, so a hard pass rate can still hide
real word-level damage; the score is printed per phrase for inspection.
"""

import asyncio
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from sim.environment import band_limit
from sim.prospect import CHUNK_BYTES, SILENCE, SimulatedProspect
from stt.groq_stt import GroqSTT
from tts.engine import PiperTTS, TTSConfig

load_dotenv(Path(__file__).parent.parent / ".env")

PHRASES = [
    "Yeah, Tuesday afternoon after two works for me.",
    "Please take me off your calling list.",
    "Northwind Logistics, how may I direct your call?",
    "That sounds reasonable.",
    "Hi, you've reached Ironwood Manufacturing. Leave a message after the beep.",
    "I'm not sure, who's asking?",
    "Well, he's in meetings all afternoon.",
    "Can I take a message instead?",
]


def score(heard: str, said: str) -> float:
    """Word-level similarity, 0.0-1.0."""
    return difflib.SequenceMatcher(
        None, said.lower().split(), heard.lower().split()
    ).ratio()


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

    results = []
    for i, phrase in enumerate(PHRASES):
        # Different seed per phrase so the noise bed varies.
        from dataclasses import replace
        from sim.prospect import script_for

        prospect = SimulatedProspect(
            replace(script_for("interested"), company="Acme"),
            tts, turn_gap=0, seed=i,
        )
        await prospect.prime()
        pcm = await tts.synthesize_pcm(phrase)
        pcm = prospect._condition(pcm)

        async def stream():
            for j in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[j:j + CHUNK_BYTES]
            for _ in range(50):
                yield SILENCE

        heard = [s.text async for s in stt.transcribe_stream(stream(), sample_rate=16000)]
        text = " ".join(heard)
        s = score(text, phrase) if text else 0.0
        results.append(s)
        flag = "ok " if s >= 0.95 else ("~  " if s >= 0.6 else "BAD")
        print(f"{flag} {s:.2f}  said: {phrase}")
        print(f"          heard: {text or '(nothing)'}")

    mean = sum(results) / len(results)
    lost = sum(1 for r in results if r == 0.0)
    print(f"\nmean word accuracy: {mean:.3f}")
    print(f"utterances lost entirely: {lost}/{len(results)}")
    return 0 if mean >= 0.95 and lost == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
