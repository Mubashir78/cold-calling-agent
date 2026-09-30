"""Call recorder with a shared timeline.

Turn-taking quality is measured as simultaneous speech: the full-duplex
literature (Sacks 1978) frames natural conversation as minimizing the time
both parties talk at once, and that is exactly what `overlap_report` counts.

Getting that number honest took three corrections, each of which had produced
a plausible but wrong figure:

1. Tracks were concatenated positionally. They drift, because the agent's
   TTS and the prospect's stream are independent iterators, so the channels
   stopped referring to the same instant. Chunks are now placed by timestamp.
2. Agent audio was stamped at synthesis. Piper runs faster than real time,
   so a whole reply landed inside a fraction of a second and appeared to
   overlap a prospect speaking over the same wall-clock span.
3. The playback consumer drained instantly. A real 16 kHz track plays at
   16 kHz; without pacing, a 23-second reply was stamped in milliseconds.

So: stamp at playback, and pace playback. Overlap from this file is real.
"""

import io
import json
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def wrap_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@dataclass
class TimedChunk:
    offset: float      # seconds from the start of the call
    samples: np.ndarray


@dataclass
class CallRecording:
    """Accumulates both sides of one call on a shared timeline."""

    lead_name: str
    company: str
    phone_number: str
    sample_rate: int = SAMPLE_RATE
    max_render_seconds: float = 1800.0

    agent: list[TimedChunk] = field(default_factory=list)
    prospect: list[TimedChunk] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    _t0: Optional[float] = field(default=None, init=False, repr=False)
    _agent_queued: int = field(default=0, init=False, repr=False)
    _agent_played: int = field(default=0, init=False, repr=False)
    _agent_finished: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Mark the call's zero point. Call before capturing anything."""
        self._t0 = time.monotonic()

    def _offset(self) -> float:
        if self._t0 is None:
            self._t0 = time.monotonic()
        return time.monotonic() - self._t0

    def _append(self, track: list[TimedChunk], pcm) -> None:
        samples = self._as_samples(pcm)
        if samples is None or samples.size == 0:
            return
        track.append(TimedChunk(offset=self._offset(), samples=samples))

    def add_agent(self, pcm) -> None:
        """Record synthesized agent audio.

        Deliberately NOT timestamped here: Piper synthesizes faster than
        real time, so a burst of sentences gets near-identical timestamps
        while playing out over several seconds. That compresses the agent's
        track against the prospect's and manufactures phantom overlap. Use
        `add_playback` from the consumer, where the real instant is known.
        """
        samples = self._as_samples(pcm)
        if samples is not None and samples.size:
            self._agent_queued += samples.size

    def add_prospect(self, pcm) -> None:
        """Record prospect audio as it is captured.

        The prospect's stream is consumed in real time, so capture time *is*
        playback time. No synthesis-ahead correction is needed here.
        """
        self.add_playback("prospect", pcm)

    def add_playback(self, speaker: str, pcm) -> None:
        """Record audio at the moment playback begins."""
        samples = self._as_samples(pcm)
        if samples is None or samples.size == 0:
            return
        track = self.agent if speaker == "agent" else self.prospect
        track.append(TimedChunk(offset=self._offset(), samples=samples))

    def mark_playback_finished(self, speaker: str, sample_count: int) -> None:
        """Count a chunk as played only once it has actually finished.

        Counting on arrival treats "playback started" as "playback ended",
        which releases the line while the last sentence is still audible.
        """
        if speaker == "agent":
            self._agent_finished += int(sample_count)

    def agent_playback_seconds(self) -> float:
        return self._agent_finished / self.sample_rate

    async def await_agent_playback(self, timeout: float = 30.0) -> bool:
        """Block until every queued sample has finished playing.

        Synthesis runs ahead of playback, so a caller that enqueues audio and
        returns has not finished speaking — it has finished *preparing* to
        speak. Handing the line back at that point is what makes two voices
        audible at once.
        """
        import asyncio as _asyncio

        loop = _asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while self._agent_finished < self._agent_queued:
            if loop.time() >= deadline:
                logger.warning("Agent playback did not drain within %.0fs", timeout)
                return False
            await _asyncio.sleep(0.005)
        return True

    def note(self, speaker: str, text: str) -> None:
        if text.strip():
            self.transcript.append((speaker, text.strip()))

    @staticmethod
    def _as_samples(pcm):
        """Accept bytes or float samples; return float32 or None.

        Never use `if not pcm` on a numpy array: its truthiness check raises.
        """
        if pcm is None:
            return None
        if isinstance(pcm, np.ndarray):
            return pcm.astype(np.float32, copy=False)
        if len(pcm) == 0:
            return None
        return pcm_to_float(bytes(pcm))

    def duration(self) -> float:
        end = 0.0
        for chunk in self.agent + self.prospect:
            end = max(end, chunk.offset + chunk.samples.size / self.sample_rate)
        return end

    def _render(self, track: list[TimedChunk], total_samples: int) -> np.ndarray:
        """Place each chunk at its timestamp on one timeline."""
        out = np.zeros(total_samples, dtype=np.float32)
        for chunk in track:
            start = int(chunk.offset * self.sample_rate)
            if start >= total_samples:
                continue
            end = min(start + chunk.samples.size, total_samples)
            out[start:end] += chunk.samples[: end - start]
        return np.clip(out, -0.98, 0.98)

    def _tracks(self) -> tuple[np.ndarray, np.ndarray]:
        total = max(
            int(self.duration() * self.sample_rate) + self.sample_rate,
            self.sample_rate,
        )
        # Each side is rendered in full, so a stray timestamp could allocate
        # gigabytes on a host with ~1GB of headroom. Cap it.
        limit = self.max_render_seconds * self.sample_rate
        if total > limit:
            logger.warning(
                "Render capped at %.0fs (timeline spans %.0fs)",
                self.max_render_seconds, total / self.sample_rate,
            )
            total = limit
        return self._render(self.agent, total), self._render(self.prospect, total)

    def agent_wav(self) -> bytes:
        left, _ = self._tracks()
        return wrap_wav(float_to_pcm(left), self.sample_rate)

    def prospect_wav(self) -> bytes:
        _, right = self._tracks()
        return wrap_wav(float_to_pcm(right), self.sample_rate)

    def stereo_wav(self) -> bytes:
        """Agent on the left channel, prospect on the right, time-aligned."""
        left, right = self._tracks()
        interleaved = np.empty(left.size * 2, dtype=np.float32)
        interleaved[0::2] = left
        interleaved[1::2] = right

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(self.sample_rate)
            wf.writeframes(float_to_pcm(interleaved))
        return buf.getvalue()

    def overlap_report(self, frame_ms: int = 20, speech_rms: float = 0.012) -> dict:
        """Measure simultaneous speech on the aligned timeline."""
        left, right = self._tracks()
        frame = max(1, self.sample_rate * frame_ms // 1000)
        n = min(left.size, right.size) // frame
        if n == 0:
            return {"frames": 0, "overlap_frames": 0, "overlap_seconds": 0.0,
                    "agent_seconds": 0.0, "prospect_seconds": 0.0}

        l_rms = np.sqrt(np.mean(
            left[:n * frame].reshape(n, frame) ** 2, axis=1))
        r_rms = np.sqrt(np.mean(
            right[:n * frame].reshape(n, frame) ** 2, axis=1))
        l_speak = l_rms >= speech_rms
        r_speak = r_rms >= speech_rms
        both = l_speak & r_speak
        return {
            "frames": n,
            "agent_frames": int(l_speak.sum()),
            "prospect_frames": int(r_speak.sum()),
            "overlap_frames": int(both.sum()),
            "overlap_seconds": round(both.sum() * frame_ms / 1000.0, 3),
            "agent_seconds": round(l_speak.sum() * frame_ms / 1000.0, 3),
            "prospect_seconds": round(r_speak.sum() * frame_ms / 1000.0, 3),
            "total_seconds": round(n * frame_ms / 1000.0, 3),
        }

    def write(self, out_dir: str = "data/calls") -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in f"{self.lead_name}_{self.company}"
        )
        paths = {
            "agent": out / f"{safe}_agent.wav",
            "prospect": out / f"{safe}_prospect.wav",
            "stereo": out / f"{safe}_stereo.wav",
            "transcript": out / f"{safe}.txt",
        }
        paths["agent"].write_bytes(self.agent_wav())
        paths["prospect"].write_bytes(self.prospect_wav())
        paths["stereo"].write_bytes(self.stereo_wav())

        lines = [f"{self.lead_name} — {self.company} ({self.phone_number})", ""]
        lines += [f"{speaker}: {text}" for speaker, text in self.transcript]
        paths["transcript"].write_text("\n".join(lines) + "\n")

        report = self.overlap_report()
        paths["overlap"] = out / f"{safe}_overlap.json"
        paths["overlap"].write_text(json.dumps(report, indent=2))
        logger.info("Recorded call to %s (overlap %.2fs of %.1fs)",
                    out / safe, report.get("overlap_seconds", 0.0),
                    report.get("total_seconds", 0.0))
        return {k: str(v) for k, v in paths.items()}
