"""Utterance segmentation.

Chunk-size independence is the property that matters here: the engine must
not care whether the transport hands it 20 ms frames or one big block, or a
conversation splits differently depending on the media stack.
"""

import numpy as np
import pytest

from stt.endpointing import UtteranceSegmenter

SR = 16000
FREQS = {"alpha": 300.0, "bravo": 700.0, "charlie": 1300.0, "delta": 2300.0}


def _speech(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _segment(**kw) -> UtteranceSegmenter:
    return UtteranceSegmenter(sample_rate=SR, **kw)


def _split(samples: np.ndarray, chunk_ms: int, **kw) -> list[np.ndarray]:
    seg = _segment(**kw)
    out: list[np.ndarray] = []
    step = SR * chunk_ms // 1000
    for i in range(0, samples.size, step):
        out.extend(seg.push(samples[i:i + step]))
    tail = seg.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_single_utterance():
    out = _split(np.concatenate([_speech(FREQS["alpha"], 1.5), _silence(1.0)]), 20)
    assert len(out) == 1
    assert out[0].size > 0


def test_two_utterances():
    out = _split(
        np.concatenate([
            _speech(FREQS["alpha"], 1.0), _silence(1.0),
            _speech(FREQS["bravo"], 1.0), _silence(1.0),
        ]), 20,
    )
    assert len(out) == 2


def test_three_utterances():
    out = _split(
        np.concatenate([
            _speech(FREQS["alpha"], 1.0), _silence(1.0),
            _speech(FREQS["bravo"], 1.0), _silence(1.0),
            _speech(FREQS["charlie"], 1.0), _silence(1.0),
        ]), 20,
    )
    assert len(out) == 3


@pytest.mark.parametrize("chunk_ms", [10, 20, 40, 160, 1000])
def test_result_is_independent_of_chunk_size(chunk_ms):
    """The regression: one big chunk collapsed three turns into one."""
    audio = np.concatenate([
        _speech(FREQS["alpha"], 1.0), _silence(1.0),
        _speech(FREQS["bravo"], 1.0), _silence(1.0),
        _speech(FREQS["charlie"], 1.0), _silence(1.0),
    ])
    out = _split(audio, chunk_ms)
    assert len(out) == 3, f"chunk_ms={chunk_ms} produced {len(out)} utterance(s)"


def test_pure_silence_yields_nothing():
    assert _split(_silence(3.0), 20) == []


def test_too_short_burst_is_dropped():
    """A 0.1s click is not an utterance."""
    out = _split(np.concatenate([_speech(FREQS["alpha"], 0.1), _silence(1.0)]), 20)
    assert out == []


def test_trailing_silence_is_trimmed():
    """Transcribers hallucinate on digital silence, so it must not be sent."""
    out = _split(
        np.concatenate([_speech(FREQS["alpha"], 1.0), _silence(1.0)]), 20
    )
    assert len(out) == 1
    # Speech is 16000 samples; anything much larger kept the silence.
    assert out[0].size < 16000 + 800


def test_leading_silence_is_trimmed():
    out = _split(
        np.concatenate([_silence(1.0), _speech(FREQS["alpha"], 1.0), _silence(1.0)]),
        20,
    )
    assert len(out) == 1
    assert out[0].size < 16000 + 800


def test_utterance_ending_at_stream_end_is_flushed():
    out = _split(_speech(FREQS["alpha"], 2.0), 20)
    assert len(out) == 1


def test_speech_without_any_trailing_silence_still_flushed():
    out = _split(
        np.concatenate([_speech(FREQS["alpha"], 1.0), _speech(FREQS["bravo"], 1.0)]),
        20,
    )
    # No silence boundary, so this is one continuous utterance.
    assert len(out) == 1


def test_very_long_utterance_stays_single_turn():
    out = _split(
        np.concatenate([_speech(FREQS["delta"], 12.0), _silence(1.0)]), 20
    )
    assert len(out) == 1


def test_empty_input_is_safe():
    seg = _segment()
    assert list(seg.push(np.array([], dtype=np.float32))) == []
    assert seg.flush() is None


def test_zero_length_chunks_are_ignored():
    seg = _segment()
    assert list(seg.push(np.zeros(0, dtype=np.float32))) == []
