"""Measure speech overlap in a recorded call.

Unit tests prove the lock works. This proves the *audio* has no overlap,
which is what a listener would actually hear. Reads the two-channel WAV and
reports every instant where both sides have energy at once.
"""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SPEECH_RMS = 0.012
FRAME_MS = 20


def load_stereo(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 2:
            raise ValueError(f"{path.name} is not stereo")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data[0::2].astype(np.float32) / 32768.0, \
           data[1::2].astype(np.float32) / 32768.0, sr


def overlap_report(agent: np.ndarray, prospect: np.ndarray, sr: int) -> dict:
    frame = max(1, sr * FRAME_MS // 1000)
    n = min(agent.size, prospect.size) // frame
    if n == 0:
        return {"frames": 0, "overlap_frames": 0, "overlap_seconds": 0.0}

    a = agent[:n * frame].reshape(n, frame)
    p = prospect[:n * frame].reshape(n, frame)
    a_rms = np.sqrt(np.mean(a**2, axis=1))
    p_rms = np.sqrt(np.mean(p**2, axis=1))

    a_speak = a_rms >= SPEECH_RMS
    p_speak = p_rms >= SPEECH_RMS
    both = a_speak & p_speak

    return {
        "frames": n,
        "agent_speech": int(a_speak.sum()),
        "prospect_speech": int(p_speak.sum()),
        "overlap_frames": int(both.sum()),
        "overlap_seconds": round(both.sum() * FRAME_MS / 1000.0, 3),
    }


def main() -> int:
    calls = sorted(Path("data/calls").glob("*_stereo.wav"))
    if not calls:
        print("No recordings found. Run a campaign first: python -m src.main")
        return 1

    worst = 0.0
    print(f"{'call':44} {'agent':>7} {'prospect':>9} {'overlap':>9}")
    for path in calls:
        agent, prospect, sr = load_stereo(path)
        report = overlap_report(agent, prospect, sr)
        worst = max(worst, report["overlap_seconds"])
        print(f"{path.stem[:44]:44} {report['agent_speech']:7d} "
              f"{report['prospect_speech']:9d} {report['overlap_seconds']:8.3f}s")

    print(f"\nworst overlap: {worst:.3f}s")
    if worst > 0.5:
        print("FAIL: the two sides talked over each other")
        return 1
    print("PASS: no meaningful simultaneous speech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
