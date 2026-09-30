"""Voice Activity Detection using Silero VAD."""

import logging
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice activity detection using Silero VAD model."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None

    async def initialize(self):
        """Load Silero VAD model."""
        try:
            import torch
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            logger.info("Silero VAD loaded")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD: {e}. Using energy-based VAD.")
            self._model = None

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        if self._model is None:
            # Energy-based fallback
            energy = np.sqrt(np.mean(audio ** 2))
            return energy > 0.01

        import torch
        tensor = torch.from_numpy(audio).float()
        speech_prob = self._model(tensor, self.sample_rate).item()
        return speech_prob > self.threshold

    async def filter_silence(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """Filter out non-speech audio chunks."""
        async for chunk in audio_stream:
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if self.is_speech(audio):
                yield chunk
