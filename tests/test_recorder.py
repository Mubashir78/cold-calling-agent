"""Recording capture and WAV output."""

import numpy as np
import pytest

from sim.recorder import CallRecording, wrap_wav


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).astype(np.int16).tobytes()


def test_accepts_bytes():
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    r.add_playback("agent", _pcm(np.array([0.1, 0.2, 0.3])))
    assert len(r.agent) == 1
    assert r.agent[0].samples.dtype == np.float32


def test_accepts_numpy_array():
    """Regression: a synthesis path passed float samples, and numpy's
    truthiness check raised ValueError, killing the whole call."""
    r = CallRecording("A", "B", "+14155550101")
    r.start()
    r.add_prospect(np.array([0.1, 0.2], dtype=np.float32))
    assert len(r.prospect) == 1


def test_ignores_empty_input():
    r = CallRecording("A", "B", "+14155550101")
    r.add_agent(b"")
    r.add_agent(np.array([], dtype=np.float32))
    r.add_agent(None)
    assert r.agent == []


def test_stereo_has_two_channels():
    r = CallRecording("A", "B", "+14155550101")
    r.add_agent(_pcm(np.full(100, 0.5)))
    r.add_prospect(_pcm(np.full(50, 0.2)))
    wav = r.stereo_wav()
    assert wav[:4] == b"RIFF"
    # 'fmt ' chunk: channels at offset 22
    import wave, io
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 2
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000


def test_mono_wavs_are_single_channel():
    r = CallRecording("A", "B", "+14155550101")
    r.add_agent(_pcm(np.full(100, 0.5)))
    import wave, io
    with wave.open(io.BytesIO(r.agent_wav())) as w:
        assert w.getnchannels() == 1


def test_tracks_are_time_aligned_to_longer():
    """The shorter track is padded so both channels span the call."""
    r = CallRecording("A", "B", "+14155550101")
    r.add_agent(_pcm(np.full(1000, 0.5)))
    r.add_prospect(_pcm(np.full(100, 0.2)))
    import wave, io
    with wave.open(io.BytesIO(r.stereo_wav())) as w:
        # 1000 samples of agent, so stereo must hold 1000 frames minimum.
        assert w.getnframes() >= 1000


def test_write_produces_all_four_files(tmp_path):
    r = CallRecording("Dana Whitfield", "Northwind Logistics", "+14155550101")
    r.start()
    r.add_playback("agent", _pcm(np.full(1600, 0.4)))
    r.add_prospect(_pcm(np.full(1600, 0.2)))
    r.note("agent", "Hello, this is an AI call.")
    r.note("prospect", "Yeah, what is this?")

    paths = r.write(str(tmp_path))
    for key in ("agent", "prospect", "stereo", "transcript", "overlap"):
        assert key in paths
    for key, path in paths.items():
        expected = ".json" if key == "overlap" else None
        assert path.endswith(expected or (".wav", ".txt")), path

    text = (tmp_path / "Dana_Whitfield_Northwind_Logistics.txt").read_text()
    assert "agent: Hello, this is an AI call." in text
    assert "prospect: Yeah, what is this?" in text


def test_filenames_are_sanitized(tmp_path):
    r = CallRecording("O'Brien / Smith", "A & B Co.", "+14155550101")
    r.add_agent(_pcm(np.full(160, 0.1)))
    paths = r.write(str(tmp_path))
    for path in paths.values():
        assert " " not in path.split("/")[-1]
        assert "&" not in path.split("/")[-1]


def test_transcript_skips_blank_turns():
    r = CallRecording("A", "B", "+14155550101")
    r.note("agent", "  ")
    r.note("prospect", "Hello")
    assert r.transcript == [("prospect", "Hello")]
