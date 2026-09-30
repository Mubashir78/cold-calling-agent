"""Audio codec utilities — PCM, mu-law, and resampling."""

import numpy as np


def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 array to 16-bit PCM bytes."""
    return (audio * 32768.0).astype(np.int16).tobytes()


def mulaw_encode(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Encode audio to mu-law."""
    audio = np.clip(audio, -1.0, 1.0)
    mulaw = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    return mulaw


def mulaw_decode(mulaw: np.ndarray, mu: int = 255) -> np.ndarray:
    """Decode mu-law audio to float32."""
    audio = np.sign(mulaw) * (np.exp(np.abs(mulaw) * np.log1p(mu)) - 1) / mu
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)
