"""Groq-hosted Whisper STT.

Same interface as the local faster-whisper engine, so the pipeline does not
care which one it has. Because the pipeline endpointing already accumulates
a whole utterance before transcribing, a single batch call per turn is the
natural fit — and it keeps a ~1 GB model off a box that has 5.6 GB of RAM
and no GPU to run 5 concurrent calls on.
"""

import asyncio
import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from groq import AsyncGroq, Groq

from stt.endpointing import UtteranceSegmenter
from stt.engine import TranscriptSegment

logger = logging.getLogger(__name__)


class GroqSTT:
    """Whisper transcription via Groq, one request per detected utterance."""

    def __init__(
        self,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        api_key: Optional[str] = None,
        prompt: str = "",
    ):
        self.model = model
        self.language = language
        self.prompt = prompt
        self._sync_client = Groq(api_key=api_key)

    async def initialize(self) -> None:
        logger.info("Groq STT ready with model %s", self.model)

    @staticmethod
    def _wrap_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def _transcribe_sync(self, wav: bytes, filename: str) -> str:
        """Blocking upload; runs in a worker thread.

        Uses the SDK's own multipart handling rather than assembling the
        body by hand, which is what produced the 403.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav)
            path = fh.name
        try:
            with open(path, "rb") as fh:
                result = self._sync_client.audio.transcriptions.create(
                    file=(filename, fh.read(), "audio/wav"),
                    model=self.model,
                    language=self.language,
                    response_format="verbose_json",
                )
            return (getattr(result, "text", "") or "").strip()
        finally:
            Path(path).unlink(missing_ok=True)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        min_utterance: float = 0.4,
        end_turn_silence: float = 0.35,
        speech_rms: float = 0.012,
    ) -> AsyncIterator[TranscriptSegment]:
        """Emit one segment per utterance, transcribing each with one request.

        `end_turn_silence` is the biggest latency control in the pipeline: the
        agent cannot begin generating until this much silence has passed.
        Production guidance budgets endpointing inside a 500ms mouth-to-ear
        target and warns that a 600ms silence threshold consumes the whole
        budget before the LLM sees a character. 350ms balances that against
        not cutting people off mid-sentence.
        """
        segmenter = UtteranceSegmenter(
            sample_rate=sample_rate,
            speech_rms=speech_rms,
            min_utterance=min_utterance,
            end_turn_silence=end_turn_silence,
        )

        async def emit(utterance: np.ndarray) -> Optional[TranscriptSegment]:
            wav = self._wrap_wav(utterance, sample_rate)
            text = await asyncio.to_thread(
                self._transcribe_sync, wav, "utterance.wav"
            )
            if not text:
                return None
            return TranscriptSegment(
                text=text, start=0.0, end=0.0,
                is_final=True, confidence=0.0,
            )

        async for chunk in audio_stream:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            for utterance in segmenter.push(samples):
                segment = await emit(utterance)
                if segment:
                    yield segment

        final = segmenter.flush()
        if final is not None:
            segment = await emit(final)
            if segment:
                yield segment

    async def transcribe_file(self, audio_path: str) -> str:
        with open(audio_path, "rb") as fh:
            return await asyncio.to_thread(
                self._transcribe_sync, fh.read(), audio_path
            )
