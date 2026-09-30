"""TTS engine using Piper (CPU, fully local).

Emits raw 16-bit mono PCM, not a WAV container, because that is what a
WebRTC/SIP track wants and what the STT side consumes.
"""

import asyncio
import io
import logging
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

VOICE_ROOT = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_paths(voice: str) -> tuple[str, str]:
    """Map a piper voice id to its (model, config) URLs.

    piper-voices layout is
        {family}/{code}/{name}/{quality}/{id}.{quality}.onnx
    so en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
    """
    try:
        family, rest = voice.split("_", 1)   # 'en' | 'US-lessac-medium'
        code_suffix, rest = rest.split("-", 1)  # 'US' | 'lessac-medium'
        name, quality = rest.rsplit("-", 1)     # 'lessac' | 'medium'
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse voice id {voice!r}; expected e.g. en_US-lessac-medium"
        ) from exc

    code = f"{family}_{code_suffix}"
    base = f"{VOICE_ROOT}/{family}/{code}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


@dataclass
class TTSConfig:
    voice: str = "en_US-lessac-medium"
    speaking_rate: float = 1.0
    target_sample_rate: int = 16000


class PiperTTS:
    """Text-to-speech using Piper."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._voice = None
        self._voice_dir: Optional[Path] = None

    async def initialize(self, voice_dir: str = "data/voices") -> None:
        """Download the voice if missing, then load it."""
        self._voice_dir = Path(voice_dir)
        self._voice_dir.mkdir(parents=True, exist_ok=True)

        model = self._voice_dir / f"{self.config.voice}.onnx"
        config_path = self._voice_dir / f"{self.config.voice}.onnx.json"

        if not (model.exists() and config_path.exists()):
            await self._download_voice()

        import piper

        self._voice = await asyncio.to_thread(
            piper.PiperVoice.load, str(model), str(config_path)
        )
        logger.info("Piper TTS ready with voice %s", self.config.voice)

    async def _download_voice(self) -> None:
        model_url, config_url = voice_paths(self.config.voice)
        assert self._voice_dir is not None
        targets = [
            (model_url, self._voice_dir / f"{self.config.voice}.onnx"),
            (config_url, self._voice_dir / f"{self.config.voice}.onnx.json"),
        ]
        for url, dest in targets:
            logger.info("Downloading %s", dest.name)
            data = await asyncio.to_thread(self._fetch, url)
            dest.write_bytes(data)

    @staticmethod
    def _fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read()

    def _synth_config(self):
        import piper

        # Piper speaks slower when length_scale is larger; invert the rate.
        return piper.SynthesisConfig(
            length_scale=1.0 / max(self.config.speaking_rate, 0.1)
        )

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Run Piper, return (float32 mono samples, source sample rate)."""
        chunks = list(self._voice.synthesize(text, syn_config=self._synth_config()))
        if not chunks:
            return np.array([], dtype=np.float32), self.config.target_sample_rate
        rate = chunks[0].sample_rate
        return (
            np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32),
            rate,
        )

    async def synthesize_pcm(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit mono PCM at the target sample rate."""
        if self._voice is None:
            raise RuntimeError("TTS engine not initialized")

        samples, rate = await asyncio.to_thread(self._synthesize_sync, text)
        if samples.size == 0:
            return b""

        if rate != self.config.target_sample_rate:
            from audio.codec import resample

            samples = resample(samples, rate, self.config.target_sample_rate)

        return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()

    async def synthesize_wav(self, text: str) -> bytes:
        """Synthesize text to a complete WAV file (for playback/debugging)."""
        pcm = await self.synthesize_pcm(text)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.target_sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    async def synthesize_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream PCM audio, flushing on sentence boundaries."""
        buffer = ""
        async for text in text_stream:
            buffer += text
            parts = buffer.replace("!", ".").replace("?", ".").split(". ")
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    if sentence.strip():
                        yield await self.synthesize_pcm(sentence.strip() + ".")
                buffer = parts[-1]

        if buffer.strip():
            yield await self.synthesize_pcm(buffer.strip())
