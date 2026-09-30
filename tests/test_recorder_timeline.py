"""Timeline correctness in the recorder.

The bug this pins: both sides were appended to separate lists and the stereo
mix concatenated them positionally. The two streams drift apart over a call,
so the result showed ~17% of the call as simultaneous speech when the
transcript showed clean turn-taking. The number was an artefact of the
measurement, not a defect in the agent.

These tests use injected offsets rather than sleeping, so they are exact.
"""

import numpy as np
import pytest

from sim.recorder import CallRecording, TimedChunk, SAMPLE_RATE


def _tone(seconds: float, freq: float = 400.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rec() -> CallRecording:
    r = CallRecording("A", "B", "+14155550101")
    r._t0 = 1000.0  # fixed clock so offsets are exact
    return r


def _at(r: CallRecording, side: str, offset: float, samples: np.ndarray) -> None:
    track = r.agent if side == "agent" else r.prospect
    track.append(TimedChunk(offset=offset, samples=samples))


def _play_at(r: CallRecording, side: str, seconds: float, **kw) -> None:
    """Record audio that *starts* `seconds` from the call start.

    Shifting the clock backwards is not a way to fake pacing — it moves the
    zero point and leaves a gap before every earlier chunk, which inflates
    the timeline instead of spacing chunks out.
    """
    _at(r, side, seconds, _tone(kw.get("tone", 1.0), **{
        k: v for k, v in kw.items() if k != "tone"
    }))


def test_non_overlapping_turns_measure_as_no_overlap():
    """Alternating 2s turns must report zero overlap."""
    r = _rec()
    for i in range(5):
        offset = i * 3.0
        _at(r, "agent" if i % 2 == 0 else "prospect", offset, _tone(2.0))

    report = r.overlap_report()
    assert report["overlap_seconds"] == 0.0, report


def test_drift_between_tracks_does_not_fake_overlap():
    """The regression: identical bursts, one side offset in time.

    Positionally concatenating these tracks would report every burst as
    simultaneous. Placed on a shared timeline, they never coincide.
    """
    r = _rec()
    for i in range(4):
        offset = i * 3.0
        _at(r, "agent", offset, _tone(1.0, 400.0))
        # Prospect is 1.5s later in the call, with a different length.
        _at(r, "prospect", offset + 1.5, _tone(0.8, 700.0))

    report = r.overlap_report()
    assert report["overlap_seconds"] == 0.0, (
        f"timeline drift was mistaken for overlap: {report}"
    )


def test_real_overlap_is_still_detected():
    """A genuinely simultaneous pair must be reported, not smoothed away."""
    r = _rec()
    _at(r, "agent", 1.0, _tone(1.0, 400.0))
    _at(r, "prospect", 1.0, _tone(1.0, 700.0))

    report = r.overlap_report()
    assert report["overlap_seconds"] > 0.8, (
        f"real overlap went undetected: {report}"
    )


def test_offset_is_preserved_in_the_rendered_track():
    r = _rec()
    _at(r, "agent", 2.0, _tone(0.5, 400.0))
    left, _ = r._tracks()
    start = int(2.0 * SAMPLE_RATE)
    before = np.max(np.abs(left[start - 100:start])) if start > 100 else 0.0
    during = np.max(np.abs(left[start:start + 8000]))
    after = np.max(np.abs(left[start + int(0.5 * SAMPLE_RATE):]))
    assert during > 0.1, "chunk was not placed at its timestamp"
    assert before < 0.01 and after < 0.01, "audio leaked outside the timestamp"


def test_timestamps_are_recorded_automatically():
    r = _rec()
    r.add_playback("agent", _tone(0.2))
    r.add_playback("prospect", _tone(0.2))
    assert r.agent and r.prospect
    assert all(c.offset >= 0.0 for c in r.agent)
    assert all(c.samples.dtype == np.float32 for c in r.prospect)
    r = _rec()
    _at(r, "agent", 5.0, _tone(1.0))
    assert r.duration() >= 5.9, r.duration()
    left, _ = r._tracks()
    # Tracks cover the whole call, plus a little padding so a trailing chunk
    # is never cut off.
    seconds = left.size / SAMPLE_RATE
    assert seconds >= r.duration()
    assert seconds < r.duration() + 1.5


def test_agent_audio_is_timestamped_at_playback_not_synthesis():
    """Regression: synthesis runs ahead of playback.

    Piper emits a burst of sentences faster than real time. Stamping them
    when queued packed the whole turn into a fraction of a second and
    reported ~8% phantom overlap against a real prospect turn.
    """
    r = _rec()
    # Queued back to back, as synthesis does.
    r.add_agent(_tone(0.5))
    r.add_agent(_tone(0.5))
    r.add_agent(_tone(0.5))
    assert r.agent == [], "queued audio must not be timestamped yet"

    # Played out later, spread over real time.
    r.add_playback("agent", _tone(0.5))
    r.add_playback("agent", _tone(0.5))
    r.add_playback("agent", _tone(0.5))

    assert len(r.agent) == 3
    offsets = [c.offset for c in r.agent]
    assert offsets == sorted(offsets)
    assert r._agent_queued == int(1.5 * SAMPLE_RATE)


def test_playback_keeps_turns_apart():
    """Agent speaks, then the prospect, with no overlap on the timeline."""
    r = _rec()
    _at(r, "agent", 0.0, _tone(1.0, 400.0))
    _at(r, "prospect", 1.5, _tone(1.0, 700.0))

    report = r.overlap_report()
    assert report["overlap_seconds"] == 0.0, report
    assert report["agent_seconds"] > 0.5
    assert report["prospect_seconds"] > 0.5


def test_queued_agent_audio_alone_reports_no_overlap():
    r = _rec()
    for _ in range(3):
        r.add_agent(_tone(0.5))
    report = r.overlap_report()
    assert report["overlap_seconds"] == 0.0, report


def test_agent_speech_spans_its_own_playback_duration():
    """A real track plays at 16 kHz; the timeline must reflect that.

    Regression: the consumer drained the queue instantly, so a 23-second
    reply was stamped within milliseconds and appeared to overlap a
    prospect speaking over the same wall-clock seconds.
    """
    r = _rec()
    # Three 0.5s chunks, played 0.5s apart, as a paced consumer would.
    for i in range(3):
        _at(r, "agent", i * 0.5, _tone(0.5))

    report = r.overlap_report()
    assert report["agent_seconds"] >= 1.4, (
        f"agent speech compressed to {report['agent_seconds']}s; "
        "playback was not paced"
    )


def test_turns_played_back_to_back_do_not_overlap():
    """Agent then prospect, each paced, on a shared timeline."""
    r = _rec()
    _at(r, "agent", 0.0, _tone(1.0, 400.0))
    _at(r, "prospect", 1.2, _tone(1.0, 700.0))

    report = r.overlap_report()
    assert report["overlap_seconds"] == 0.0, report
    assert report["agent_seconds"] > 0.8
    assert report["prospect_seconds"] > 0.8


def test_stereo_channels_carry_the_right_sides():
    import io
    import wave

    r = _rec()
    # Different amplitudes so the two channels are unambiguous.
    _at(r, "agent", 0.0, _tone(1.0, 400.0) * 0.8)
    _at(r, "prospect", 2.0, _tone(1.0, 700.0) * 0.2)

    with wave.open(io.BytesIO(r.stereo_wav())) as w:
        assert w.getnchannels() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    left = np.abs(data[0::2].astype(np.float32))
    right = np.abs(data[1::2].astype(np.float32))
    assert left.max() > right.max() * 2, (
        f"channels swapped or mixed: L={left.max()} R={right.max()}"
    )


def test_overlap_json_is_written(tmp_path):
    r = _rec()
    _at(r, "agent", 0.0, _tone(0.5))
    _at(r, "prospect", 3.0, _tone(0.5))
    paths = r.write(str(tmp_path))
    import json
    report = json.loads(open(paths["overlap"]).read())
    assert report["overlap_seconds"] == 0.0
    assert "total_seconds" in report
