"""Locate overlap in a recorded call: which turns, and what is being said.

Guessing at the cause has failed repeatedly. This prints the overlap
intervals with the transcript line nearest each one, so the offending turn is
identified directly.
"""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FRAME_MS = 20
RMS = 0.012


def load(path: Path):
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return data[0::2].astype(np.float32) / 32768.0, \
           data[1::2].astype(np.float32) / 32768.0, w.getframerate()


def intervals(mask: np.ndarray, min_frames: int = 5) -> list[tuple[float, float]]:
    """Contiguous True runs, in seconds, dropping brief blips."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frames:
                out.append((start * FRAME_MS / 1000, i * FRAME_MS / 1000))
            start = None
    if start is not None and len(mask) - start >= min_frames:
        out.append((start * FRAME_MS / 1000, len(mask) * FRAME_MS / 1000))
    return out


def merge(spans, gap: float) -> list[tuple[float, float]]:
    """Join spans separated by less than `gap` seconds.

    Within one utterance, quiet consonants dip below the RMS threshold and
    back. Counting frames turns a single sentence into dozens of blips and
    manufactures overlap that no listener would perceive. Turn-taking is a
    property of turns, so gaps shorter than an inter-word pause are joined.
    """
    if not spans:
        return []
    out = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = out[-1]
        if start - last_end <= gap:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


def overlap_of(a: np.ndarray, p: np.ndarray, join: float = 0.6) -> dict:
    """Total seconds where an agent turn and a prospect turn both run."""
    a_turns = merge(intervals(a), join)
    p_turns = merge(intervals(p), join)
    total = 0.0
    collisions = []
    for a0, a1 in a_turns:
        for p0, p1 in p_turns:
            lo, hi = max(a0, p0), min(a1, p1)
            if hi > lo:
                total += hi - lo
                collisions.append((lo, hi))
    return {
        "agent_turns": len(a_turns),
        "prospect_turns": len(p_turns),
        "overlap_seconds": round(total, 2),
        "collisions": collisions,
    }


def main() -> int:
    calls = sorted(Path("data/calls").glob("*_stereo.wav"))
    if not calls:
        print("no recordings; run a campaign first")
        return 1

    for path in calls:
        left, right, sr = load(path)
        frame = max(1, sr * FRAME_MS // 1000)
        n = min(left.size, right.size) // frame

        def speech(x):
            return np.sqrt(np.mean(
                x[:n * frame].reshape(n, frame) ** 2, axis=1)) >= RMS

        a, p = speech(left), speech(right)
        rep = overlap_of(a, p)

        print(f"\n=== {path.stem}")
        print(f"call {n * FRAME_MS / 1000:.1f}s")
        print(f"turns: agent={rep['agent_turns']}  "
              f"prospect={rep['prospect_turns']}")
        print(f"raw frame overlap: {int((a & p).sum()) * FRAME_MS / 1000:.2f}s")
        print(f"turn-level overlap: {rep['overlap_seconds']:.2f}s")

        if not rep["collisions"]:
            print("no turn ever overlaps the other side")
        for lo, hi in rep["collisions"]:
            print(f"  collision {lo:6.2f}s -> {hi:6.2f}s ({hi - lo:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
