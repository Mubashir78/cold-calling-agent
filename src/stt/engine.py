"""Streaming STT engine using faster-whisper (CPU-optimized)."""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
from faster_whisper import WhisperModel

from stt.endpointing import UtteranceSegmenter

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    is_final: bool
    confidence: float


class StreamingSTT:
    """Streaming speech-to-text using faster-whisper with VAD filtering."""

    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    async def initialize(self):
        """Load the whisper model."""
        logger.info(f"Loading whisper model: {self.model_size} on {self.device}")
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            ),
        )
        logger.info("Whisper model loaded")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        min_utterance: float = 0.4,
        end_turn_silence: float = 0.35,
        speech_rms: float = 0.012,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe a stream, emitting one segment per detected utterance.

        Segmentation is delegated to UtteranceSegmenter so this behaves
        identically to the hosted backend regardless of chunk size.
        """
        if not self._model:
            raise RuntimeError("STT engine not initialized")

        segmenter = UtteranceSegmenter(
            sample_rate=sample_rate,
            speech_rms=speech_rms,
            min_utterance=min_utterance,
            end_turn_silence=end_turn_silence,
        )

        async for chunk in audio_stream:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            for utterance in segmenter.push(samples):
                segment = await self._transcribe_chunk(utterance, sample_rate)
                if segment and segment.text.strip():
                    yield segment

        final = segmenter.flush()
        if final is not None:
            segment = await self._transcribe_chunk(final, sample_rate)
            if segment and segment.text.strip():
                yield segment

    async def _transcribe_chunk(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[TranscriptSegment]:
        """Transcribe a single chunk of audio."""
        if not self._model:
            return None

        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=True,
            ),
        )

        # Get the first segment
        for seg in segments:
            return TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                is_final=True,
                confidence=seg.avg_logprob,
            )
        return None

    async def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file (for testing)."""
        if not self._model:
            raise RuntimeError("STT engine not initialized")

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio_path,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            ),
        )
        return " ".join(seg.text for seg in segments).strip()
