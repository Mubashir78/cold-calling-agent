"""List recorded calls and open them for playback.

Usage:
  python -m src.listen              # list calls
  python -m src.listen play <name>  # play the stereo mix
  python -m src.listen <name>       # same as play
"""

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

CALLS_DIR = Path("data/calls")

# Tried in order; first one present wins.
PLAYERS = ["paplay", "aplay", "ffplay", "afplay", "mpv", "cvlc"]


def find_player() -> str | None:
    for player in PLAYERS:
        if shutil.which(player):
            return player
    return None


def calls() -> list[Path]:
    if not CALLS_DIR.exists():
        return []
    return sorted(p for p in CALLS_DIR.iterdir() if p.suffix == ".wav")


def describe(path: Path) -> str:
    try:
        with wave.open(str(path)) as w:
            seconds = w.getnframes() / w.getframerate()
            return f"{seconds:6.1f}s  {w.getnchannels()}ch  {path.name}"
    except Exception as exc:  # noqa: BLE001 - listing must not die on one bad file
        return f"   ????  unreadable ({exc})  {path.name}"


def list_calls() -> int:
    found = calls()
    if not found:
        print(f"No recordings in {CALLS_DIR}. Run a campaign first:")
        print("  python -m src.main")
        return 1
    print(f"{len(found)} file(s) in {CALLS_DIR}:\n")
    for path in found:
        print("  " + describe(path))
    transcripts = sorted(CALLS_DIR.glob("*.txt"))
    if transcripts:
        print("\nTranscripts:")
        for t in transcripts:
            print("  " + t.name)
    return 0


def resolve(name: str) -> Path | None:
    """Find a call by name fragment, preferring the stereo mix."""
    found = calls()
    matches = [p for p in found if name.lower() in p.name.lower()]
    if not matches:
        return None
    stereo = [p for p in matches if "stereo" in p.name]
    if stereo:
        return stereo[0]
    return sorted(matches, key=lambda p: "agent" not in p.name)[0]


def play(path: Path) -> int:
    player = find_player()
    if not player:
        print(f"No audio player found. Tried: {', '.join(PLAYERS)}")
        print(f"\nOpen this file instead: {path.resolve()}")
        return 2
    print(f"Playing {path.name} via {player}")
    try:
        return subprocess.call([player, str(path)])
    except KeyboardInterrupt:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recorded calls")
    parser.add_argument("action", nargs="?", default="list",
                        help="list, or play")
    parser.add_argument("name", nargs="?", help="call name fragment")
    args = parser.parse_args()

    if args.action in ("list", "ls"):
        return list_calls()

    name = args.name or args.action
    if args.action == "play" and not args.name:
        print("play needs a call name")
        return 1

    path = resolve(name)
    if path is None:
        print(f"No recording matching {name!r}. Run without arguments to list.")
        return 1
    return play(path)


if __name__ == "__main__":
    raise SystemExit(main())
