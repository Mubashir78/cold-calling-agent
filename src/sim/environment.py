"""Acoustic environment for the simulated call.

A real phone line is not a clean digital recording: it has room noise, a
narrowband codec, breath before someone answers, and a beep on voicemail.
Simulating those makes the STT/LLM/TTS loop face the conditions it will
actually meet, instead of a studio-clean signal that flatters it.
"""

import numpy as np

# Narrowband telephony is roughly 300-3400 Hz. Codecs roll off hard at both
# ends, which is why sibilants are the first casualty.
PHONE_LOW_HZ = 300.0
PHONE_HIGH_HZ = 3400.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate room tone: more energy at low frequencies than white."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n).astype(np.float32)
    # Running sum of a white sequence approximates -3 dB/octave.
    smoothed = np.cumsum(white)
    smoothed -= smoothed.mean()
    peak = np.max(np.abs(smoothed)) or 1.0
    return smoothed / peak


def band_limit(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = PHONE_LOW_HZ,
    high_hz: float = PHONE_HIGH_HZ,
    aggressiveness: float = 0.6,
) -> np.ndarray:
    """Gentle narrowband shaping, in the frequency domain.

    `aggressiveness` scales how much out-of-band energy is removed; at 0
    everything passes through. The skirts start well inside the band edges
    and reach full depth outside them, so the middle of the speech band
    (roughly 500-2500 Hz) passes at unity gain. A hard cut is what a codec
    does, but it eats the sibilants, and the point of this is to degrade the
    signal the way a line does without making the transcript unreadable.
    """
    if audio.size < 8:
        return audio

    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)

    # Skirt start points sit AT the nominal band edges, not inside them.
    #
    # Placing them at 0.7x meant the top skirt began at 2380 Hz and reached full
    # depth by 3400, so a 3 kHz tone came out at 0.51 gain and anything above
    # 2380 was carved away. 3 kHz is inside the intelligibility core of speech,
    # which is why the prospect sounded garbled while the unfiltered agent did
    # not. Narrowband telephony is honestly lossy ABOVE 3.4 kHz, not at it.
    low_edge = low_hz
    high_edge = high_hz
    low_width = max(low_hz * 0.6, 1.0)
    high_width = max(high_hz * 0.6, 1.0)

    low = np.clip((low_edge - freqs) / low_width, 0.0, 1.0)
    high = np.clip((freqs - high_edge) / high_width, 0.0, 1.0)
    # Raised-cosine skirts: a brick wall rings, a real codec tapers.
    shape = aggressiveness * (
        np.sin(low * np.pi / 2) + np.sin(high * np.pi / 2)
    )
    gain = np.clip(1.0 - shape, 0.0, 1.0)

    out = np.fft.irfft(spectrum * gain, n=audio.size).astype(np.float32)
    return out


def room_tone(n: int, sample_rate: int, rng: np.random.Generator,
              level_db: float = -46.0) -> np.ndarray:
    """Continuous background hiss, quieter than the speech."""
    noise = pink_noise(n, rng)
    return noise * (10.0 ** (level_db / 20.0))


def add_noise_bed(audio: np.ndarray, rng: np.random.Generator,
                  level_db: float = -46.0) -> np.ndarray:
    """Mix room noise under a signal, then guard against clipping."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > 0.05:
        audio = audio * (0.7 / peak)
    mixed = audio + room_tone(audio.size, 16000, rng, level_db)
    return np.clip(mixed, -0.98, 0.98).astype(np.float32)


def breath(rng: np.random.Generator, duration_s: float = 0.22,
           level_db: float = -30.0) -> np.ndarray:
    """A short inhale, for the gap before someone answers."""
    n = int(duration_s * 16000)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / 16000.0
    envelope = np.sin(np.pi * np.clip(t / max(duration_s, 1e-6), 0, 1)) ** 2
    # Breath is broadband noise with a falling tilt.
    hiss = pink_noise(n, rng) * envelope
    return (hiss * (10.0 ** (level_db / 20.0))).astype(np.float32)


def voicemail_beep(freq_hz: float = 1000.0, duration_s: float = 0.45,
                   level_db: float = -20.0) -> np.ndarray:
    """The tone that precedes a voicemail message."""
    n = int(duration_s * 16000)
    t = np.arange(n) / 16000.0
    fade = int(0.01 * 16000)
    envelope = np.ones(n, dtype=np.float32)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    tone = np.sin(2 * np.pi * freq_hz * t) * envelope
    return (tone * (10.0 ** (level_db / 20.0))).astype(np.float32)


def to_pcm(audio: np.ndarray) -> bytes:
    """float32 [-1,1] to 16-bit mono little-endian."""
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
