"""Voice activity detection and utterance endpointing.

Shared by every STT backend, so the local and hosted engines segment a
conversation identically.

Endpointing is frame-based rather than chunk-based on purpose: inspecting
only the tail of each chunk makes boundary detection depend on the caller's
chunk size, which is not something the engine should assume.
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_SPEECH_RMS = 0.012
DEFAULT_MIN_UTTERANCE = 0.4
DEFAULT_END_TURN_SILENCE = 0.7


@dataclass
class UtteranceSegmenter:
    """Split PCM into utterances, using trailing silence to end a turn.

    Emits only the speech-bearing span of each utterance. Leading and
    trailing silence are trimmed, because handing a transcriber a second of
    digital silence invites it to invent a short plausible sentence.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_ms: int = DEFAULT_FRAME_MS
    speech_rms: float = DEFAULT_SPEECH_RMS
    min_utterance: float = DEFAULT_MIN_UTTERANCE
    end_turn_silence: float = DEFAULT_END_TURN_SILENCE
    trim_leading_silence: bool = True

    _pending: List[np.ndarray] = field(default_factory=list, init=False)
    _pending_len: int = field(default=0, init=False)
    _utterance: List[np.ndarray] = field(default_factory=list, init=False)
    _utterance_len: int = field(default=0, init=False)
    _trailing_silence_frames: int = field(default=0, init=False)
    _in_utterance: bool = field(default=False, init=False)

    @property
    def frame_samples(self) -> int:
        return max(1, self.sample_rate * self.frame_ms // 1000)

    @property
    def min_speech_frames(self) -> int:
        return max(1, int(round(self.min_utterance * 1000 / self.frame_ms)))

    @property
    def silence_frames(self) -> int:
        return max(1, int(round(self.end_turn_silence * 1000 / self.frame_ms)))

    def _is_speech(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        return float(np.sqrt(np.mean(frame**2))) >= self.speech_rms

    def _append(self, target: List[np.ndarray], chunk: np.ndarray) -> int:
        target.append(chunk)
        return len(chunk)

    def _consume(self, chunk: np.ndarray) -> Iterator[np.ndarray]:
        """Split chunk into frames and update state, yielding finished turns."""
        step = self.frame_samples

        if self._pending_len:
            need = step - self._pending_len
            if chunk.size < need:
                self._append(self._pending, chunk)
                self._pending_len += chunk.size
                return
            self._append(self._pending, chunk[:need])
            frame = np.concatenate(self._pending)
            self._pending = []
            self._pending_len = 0
            rest = chunk[need:]
        else:
            frame, rest = chunk[: min(step, chunk.size)], chunk[step:]

        frames = [frame]
        if rest.size:
            whole = rest.size // step
            if whole:
                frames.extend(
                    rest[i * step:(i + 1) * step] for i in range(whole)
                )
            tail = rest[whole * step:]
            if tail.size:
                self._append(self._pending, tail)
                self._pending_len = tail.size

        for f in frames:
            if self._is_speech(f):
                self._trailing_silence_frames = 0
                if not self._in_utterance:
                    self._in_utterance = True
                    self._utterance = []
                    self._utterance_len = 0
                self._append(self._utterance, f)
                self._utterance_len += f.size
            elif self._in_utterance:
                self._trailing_silence_frames += 1
                if self._trailing_silence_frames >= self.silence_frames:
                    self._in_utterance = False
                    self._trailing_silence_frames = 0
                    utterance = self._close()
                    if utterance is not None:
                        yield utterance

    def _close(self) -> Optional[np.ndarray]:
        if not self._utterance:
            return None
        samples = np.concatenate(self._utterance)
        self._utterance = []
        self._utterance_len = 0
        return self._trim(samples)

    def _trim(self, samples: np.ndarray) -> Optional[np.ndarray]:
        if samples.size == 0:
            return None
        step = self.frame_samples
        count = samples.size // step
        if count == 0:
            return None
        framed = samples[:count * step].reshape(count, step)
        energy = np.sqrt(np.mean(framed**2, axis=1))
        speech = np.nonzero(energy >= self.speech_rms)[0]
        if speech.size == 0:
            return None
        if speech.size < self.min_speech_frames:
            return None
        if self.trim_leading_silence:
            start = int(speech[0]) * step
        else:
            start = 0
        end = int(speech[-1]) * step + step
        return samples[start:end]

    def push(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        """Feed samples; yield each utterance as it completes."""
        if samples.size == 0:
            return
        yield from self._consume(samples)

    def flush(self) -> Optional[np.ndarray]:
        """Return the utterance still open at end of stream, if any."""
        if self._pending_len:
            tail = np.concatenate(self._pending)
            self._pending = []
            self._pending_len = 0
            if self._is_speech(tail):
                if not self._in_utterance:
                    self._in_utterance = True
                    self._utterance = []
                self._append(self._utterance, tail)
        if not self._in_utterance:
            return None
        self._in_utterance = False
        self._trailing_silence_frames = 0
        return self._close()
