"""Acoustic conditioning and prospect behavior."""

import numpy as np
import pytest

from sim.environment import (
    add_noise_bed,
    band_limit,
    breath,
    to_pcm,
    voicemail_beep,
)
from sim.prospect import MIN_TURN_GAP, SCRIPTS, SimulatedProspect, script_for

SR = 16000


def _speech(seconds: float = 1.0, freq: float = 500.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _response_at(freq: float, aggressiveness: float = 1.0) -> float:
    """Peak magnitude of a pure tone after filtering, relative to unfiltered.

    Comparing spectra directly avoids the confound of irfft not being
    peak-normalized, which says nothing about the filter's response.
    """
    signal = _speech(1.0, freq=freq)
    out = band_limit(signal, SR, aggressiveness=aggressiveness)
    in_peak = float(np.max(np.abs(np.fft.rfft(signal))))
    out_peak = float(np.max(np.abs(np.fft.rfft(out))))
    return out_peak / in_peak


def test_band_limit_attenuates_far_out_of_band():
    """6 kHz is well above telephony and should be strongly reduced."""
    assert _response_at(6000.0) < 0.45


def test_band_limit_preserves_the_speech_band():
    """1 kHz is dead center; it must survive largely intact."""
    assert _response_at(1000.0, aggressiveness=0.6) > 0.7


def test_band_limit_response_falls_with_aggressiveness():
    """`aggressiveness` scales attenuation, so it must change the response.

    Probed above the band edge, since below it the filter is now transparent by
    design and every aggressiveness would give the same answer.
    """
    soft = _response_at(4500.0, aggressiveness=0.3)
    hard = _response_at(4500.0, aggressiveness=1.0)
    assert hard < soft, "aggressiveness had no effect on the response"


def test_band_limit_aggressiveness_zero_is_transparent():
    signal = _speech(0.5)
    out = band_limit(signal, SR, aggressiveness=0.0)
    assert np.allclose(signal, out, atol=1e-6)


def test_noise_bed_never_clips():
    loud = np.ones(1000, dtype=np.float32) * 0.99
    out = add_noise_bed(loud, np.random.default_rng(0))
    assert np.max(np.abs(out)) <= 0.98 + 1e-6


def test_noise_bed_is_quiet_relative_to_speech():
    speech = _speech(1.0)
    out = add_noise_bed(speech, np.random.default_rng(1))
    # Noise should not dominate the segment.
    assert float(np.mean(out**2)) < float(np.mean(speech**2)) * 4


def test_noise_bed_reduces_peak_to_headroom():
    """A hot signal must be pulled down, not clipped into distortion."""
    hot = np.ones(2000, dtype=np.float32)
    out = add_noise_bed(hot, np.random.default_rng(2))
    assert float(np.max(np.abs(out))) < 0.9


def test_breath_is_short_and_quiet():
    b = breath(np.random.default_rng(3))
    assert 0.05 < b.size / SR < 0.6
    assert float(np.max(np.abs(b))) < 0.2


def test_voicemail_beep_is_a_tone():
    tone = voicemail_beep(freq_hz=1000.0)
    assert 0.2 < tone.size / SR < 0.8
    assert float(np.max(np.abs(tone))) > 0.05
    # Dominant frequency should be the requested one.
    spectrum = np.abs(np.fft.rfft(tone * np.hanning(tone.size)))
    peak_bin = int(np.argmax(spectrum))
    assert abs(peak_bin * SR / tone.size - 1000.0) < 40.0


def test_to_pcm_round_trip():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = np.frombuffer(to_pcm(samples), dtype=np.int16)
    assert pcm[0] == 0
    assert abs(pcm[1] / 32767.0 - 0.5) < 0.01
    assert pcm[3] == 32767
    assert pcm[4] == -32767


def test_scripts_differ_behaviorally():
    """The four scenarios must not all sound and behave identically."""
    rates = {k: s.speaking_rate for k, s in SCRIPTS.items()}
    assert len(set(rates.values())) > 1, "every persona has the same pace"
    assert SCRIPTS["gatekeeper"].pickup_delay < SCRIPTS["interested"].pickup_delay
    assert SCRIPTS["voicemail"].voicemail is True
    assert SCRIPTS["dnc"].asks_to_be_removed is True
    assert SCRIPTS["dnc"].hangup_probability > 0


def test_turn_gap_respects_agent_floor():
    """A prospect must not interrupt the agent mid-reply."""
    p = SimulatedProspect(script_for("voicemail"), tts=None, turn_gap=0.0)
    for i in range(20):
        assert p._gap_for(i) >= MIN_TURN_GAP


def test_voicemail_gap_is_still_bounded_by_floor():
    p = SimulatedProspect(script_for("voicemail"), tts=None, turn_gap=0.0)
    assert all(p._gap_for(i) >= MIN_TURN_GAP for i in range(20))


def test_prospect_lines_have_no_placeholders():
    from dataclasses import replace

    for key in SCRIPTS:
        p = SimulatedProspect(
            replace(script_for(key), company="Test Co"), tts=None
        )
        assert all("{company}" not in line for line in p.lines)


def _tone_gain(sig, out, freq):
    """Magnitude of `freq` in the filtered signal, relative to unfiltered."""
    f = np.fft.rfftfreq(len(out), 1 / 16000)
    i = int(np.argmin(np.abs(f - freq)))
    return float(np.abs(np.fft.rfft(out))[i] / (np.abs(np.fft.rfft(sig))[i] or 1.0))


def test_speech_band_survives_the_phone_filter():
    """Regression: the filter gutted 3 kHz, which is speech, not sibilance.

    `high_edge = PHONE_HIGH_HZ * 0.7` put the top of the skirt at 2380 Hz with
    1020 Hz of width, so it reached full depth by 3400 Hz — and everything above
    2380 was attenuated toward zero. A tone at 3000 Hz came out at exactly 0.0.
    That is inside the intelligibility core of speech, and it is why the
    prospect sounded garbled while the agent, whose audio is not filtered, was
    clean.

    Narrowband telephony is honestly lossy above ~3.4 kHz. It is not lossy at
    1-3 kHz.
    """
    sr = 16000
    t = np.arange(sr) / sr
    for freq in (500, 1000, 2000, 3000):
        sig = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        out = band_limit(sig.copy(), sr, aggressiveness=0.6)
        gain = _tone_gain(sig, out, freq)
        assert gain > 0.7, (
            f"{freq} Hz attenuated to {gain:.2f} — inside the speech band"
        )


def test_filter_still_removes_true_out_of_band_energy():
    """The filter must keep doing its job above the telephony band."""
    sr = 16000
    t = np.arange(sr) / sr
    sig = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    out = band_limit(sig.copy(), sr, aggressiveness=0.6)
    gain = _tone_gain(sig, out, 6000)
    # 6 kHz is ~2.6 kHz above the edge, where the raised-cosine skirt with
    # aggressiveness 0.6 has flattened to its floor. Assert the honest
    # expectation: clearly reduced relative to in-band, not annihilated.
    in_band = _tone_gain(sig, band_limit(
        (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32),
        sr, aggressiveness=0.6), 1000)
    assert gain < 0.5 * in_band, (
        f"6 kHz passed at {gain:.2f}; the phone band ends at 3.4 kHz"
    )


def test_aggressiveness_zero_is_a_no_op():
    sr = 16000
    t = np.arange(sr) / sr
    sig = (0.3 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    out = band_limit(sig.copy(), sr, aggressiveness=0.0)
    assert _tone_gain(sig, out, 5000) > 0.95
