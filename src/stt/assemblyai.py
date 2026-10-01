"""AssemblyAI realtime speech-to-text.

Implements the same streaming contract as the other STT backends, so it drops
in without touching the pipeline: `transcribe_stream` consumes PCM and yields
`TranscriptSegment` per finished utterance.

Unlike the batch backends this keeps ONE WebSocket open for the whole call and
streams audio as it arrives, which is what the realtime API is for: sub-second
transcription and server-side end-of-turn detection.

Two features carry most of the value for an outbound call:

`agent_context`
    AssemblyAI is told what the agent just said, so the prospect's reply is
    transcribed with that as context. Short or ambiguous replies -- "Tuesday
    afternoon", "send me the invite" -- are exactly the ones a context-free
    recogniser mangles, and this is the difference between booking the slot
    the prospect named and booking a garbled one.

`keyterms_prompt`
    Product and industry vocabulary is biased in, so proper nouns survive.

v3 takes its configuration as query parameters on the URL, not as a handshake
message; sending a JSON handshake is rejected as an invalid message type.
"""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import websockets

from stt.engine import TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_HOST = "wss://streaming.assemblyai.com/v3/ws"

# `universal-3-pro` is not a valid identifier; the streaming models are
# versioned. 3.5 Pro is the one that supports agent_context and keyterms.
DEFAULT_MODEL = "universal-3-5-pro"

# Milliseconds of audio per WebSocket message. The server accepts
# 50-1000ms; 100ms sits comfortably inside that.
FRAME_MS = 100


class AssemblyAIRealtimeSTT:
    """Streaming STT over AssemblyAI's realtime WebSocket API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        keyterms: Optional[list] = None,
        sample_rate: int = 16000,
        connect_timeout: float = 30.0,
        connect_retries: int = 3,
        **_: object,
    ):
        self._api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")
        self._model = model
        self._host = host
        self._keyterms = keyterms or []
        self._sample_rate = sample_rate
        self._socket = None
        self.connect_timeout = connect_timeout
        self.connect_retries = connect_retries

    def _url(self) -> str:
        params = {
            "sample_rate": str(self._sample_rate),
            "encoding": "pcm_s16le",
            "speech_model": self._model,
            # Only finished turns. Partials would make the agent answer the
            # same words several times as the recogniser revises them.
            # Partials are requested so generation can start before endpointing
            # confirms the turn. They are surfaced with `is_final=False`, and the
            # pipeline never records one as a turn — it uses them to prime the
            # model and nothing else.
            "include_partial_turns": "true",
            # Tight turn detection: a phone pause is short, and a 1.5s default
            # would leave the line dead while the server waits for certainty.
            "min_turn_silence": "120",
            "max_turn_silence": "450",
        }
        if self._keyterms:
            params["keyterms_prompt"] = json.dumps(self._keyterms)
        return f"{self._host}?{urlencode(params)}"

    async def initialize(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY is not set. Sign up at "
                "https://www.assemblyai.com/dashboard/signup and add it to .env"
            )
        logger.info("AssemblyAI realtime STT ready with model %s", self._model)

    def set_agent_context(self, text: str) -> None:
        """Tell the recogniser what the agent just said.

        Fire-and-forget: it takes effect on the next turn, so there is no
        reason to make the caller's hot path wait on the socket.
        """
        if not text or self._socket is None:
            return
        try:
            self._socket.send_nowait(json.dumps({
                "type": "UpdateConfiguration",
                "agent_context": text[:1750],
            }))
        except Exception as exc:
            logger.debug("agent_context not sent: %s", exc)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        **_: object,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream audio to AssemblyAI, yielding one segment per finished turn."""
        if not self._api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

        self._sample_rate = sample_rate
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        # A hosted transport being briefly unavailable is ordinary, and a lost
        # connection ends the call. Measured: connecting took ~3s on an idle box
        # but exceeded websockets' 10s default open_timeout inside the running
        # app, where the TTS models are resident. So: a longer timeout, and a
        # few attempts before giving up.
        connect_timeout = float(self.connect_timeout)
        attempts = max(1, int(self.connect_retries))
        last: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                logger.info("Recognition reconnect attempt %d/%d",
                            attempt, attempts)
                await asyncio.sleep(min(2.0 * (attempt - 1), 6.0))
            session = self._open(connect_timeout)
            try:
                socket = await session.__aenter__()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                logger.warning("Recognition connect failed (attempt %d/%d): %s",
                               attempt, attempts, exc)
                continue
            try:
                async for segment in self._pump(socket, queue, done, audio_stream):
                    yield segment
            finally:
                await session.__aexit__(None, None, None)
            return
        logger.error("Recognition unavailable after %d attempts: %s",
                     attempts, last)
        raise last  # type: ignore[misc]

    def _open(self, connect_timeout: float):
        return websockets.connect(
            self._url(),
            additional_headers={"Authorization": self._api_key},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=connect_timeout,
        )

    def set_agent_context(self, text: str) -> None:
        """Tell the recogniser what the agent just said.

        Fire-and-forget: it takes effect on the next turn, so there is no
        reason to make the caller's hot path wait on the socket.
        """
        if not text or self._socket is None:
            return
        try:
            self._socket.send_nowait(json.dumps({
                "type": "UpdateConfiguration",
                "agent_context": text[:1750],
            }))
        except Exception as exc:
            logger.debug("agent_context not sent: %s", exc)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        **_: object,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream audio to AssemblyAI, yielding one segment per finished turn."""
        if not self._api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

        self._sample_rate = sample_rate
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        # A hosted transport being briefly unavailable is ordinary, and a lost
        # connection ends the call. Measured: connecting took ~3s on an idle box
        # but exceeded websockets' 10s default open_timeout inside the running
        # app, where the TTS models are resident. So: a longer timeout, and a
        # few attempts before giving up.
        connect_timeout = float(self.connect_timeout)
        attempts = max(1, int(self.connect_retries))
        last: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                logger.info("Recognition reconnect attempt %d/%d",
                            attempt, attempts)
                await asyncio.sleep(min(2.0 * (attempt - 1), 6.0))
            session = self._open(connect_timeout)
            try:
                socket = await session.__aenter__()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                logger.warning("Recognition connect failed (attempt %d/%d): %s",
                               attempt, attempts, exc)
                continue
            try:
                async for segment in self._pump(socket, queue, done, audio_stream):
                    yield segment
            finally:
                await session.__aexit__(None, None, None)
            return
        logger.error("Recognition unavailable after %d attempts: %s",
                     attempts, last)
        raise last  # type: ignore[misc]

    def _open(self, connect_timeout: float):
        return websockets.connect(
            self._url(),
            additional_headers={"Authorization": self._api_key},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=connect_timeout,
        )

    async def _pump(self, socket, queue, done, audio_stream):
        """Run the read and write tasks over a live socket, yielding segments."""
        self._socket = socket
        logger.info("AssemblyAI streaming (%s)", self._model)

        async def reader() -> None:
            try:
                async for raw in socket:
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    kind = msg.get("type")
                    if kind == "Error":
                        logger.error("AssemblyAI error: %s", msg.get("error"))
                        continue
                    text = (msg.get("transcript") or "").strip()
                    if not text or kind != "Turn":
                        continue
                    # Partials are the recogniser still revising. They let
                    # the pipeline start generating before endpointing
                    # settles, and are flagged so nothing mistakes one for
                    # a completed turn.
                    is_final = bool(msg.get("end_of_turn"))
                    await queue.put(TranscriptSegment(
                        text=text,
                        start=float(msg.get("start", 0.0)) / 1000.0,
                        end=float(msg.get("end", 0.0)) / 1000.0,
                        is_final=is_final,
                        confidence=1.0,
                    ))
            except websockets.ConnectionClosed as exc:
                logger.info("AssemblyAI stream closed: %s", exc)
            except Exception:
                logger.exception("AssemblyAI reader failed")
            finally:
                await queue.put(done)

        async def writer() -> None:
            # AssemblyAI requires each binary message to hold 50-1000ms of
            # audio. The pipeline emits 20ms chunks, which is below the
            # minimum, and the server rejects the whole session with
            # "Input Duration Violation" — the call dies about a second in.
            # So re-frame on the way out. Deliberately no sleep: the
            # producer already paces in real time, and adding a delay here
            # would stretch the audio beyond its true duration.
            frame_bytes = int(self._sample_rate * FRAME_MS / 1000) * 2
            buf = bytearray()
            try:
                async for chunk in audio_stream:
                    if not chunk:
                        continue
                    buf += chunk
                    while len(buf) >= frame_bytes:
                        await socket.send(bytes(buf[:frame_bytes]))
                        del buf[:frame_bytes]
            except websockets.ConnectionClosed:
                logger.info("AssemblyAI socket closed while sending")
            except Exception:
                logger.exception("AssemblyAI writer failed")
            finally:
                if buf:
                    try:
                        await socket.send(bytes(buf))
                    except Exception:
                        pass
                # Ask the server to emit its last turn instead of dropping
                # whatever the prospect said as the line went dead. v3
                # requires an explicit `type` on client messages; the v2
                # `terminate_session` key is rejected as an invalid type.
                try:
                    await socket.send(json.dumps({"type": "Terminate"}))
                except Exception:
                    pass

        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item
        finally:
            self._socket = None
            for task in (reader_task, writer_task):
                task.cancel()
            await asyncio.gather(reader_task, writer_task,
                                 return_exceptions=True)
