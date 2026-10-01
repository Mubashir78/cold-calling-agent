# TODO - Cold Calling Agent

*Last updated: 2026-10-01*

## ✅ FIXED AND VERIFIED

- [x] **Agent spoke as the prospect / no voice contrast.** `pipeline._warm_far_end` passed its own `PiperTTS` into `far.prepare(tts)`, and the far end forwarded that object to its synthesizer. From the second utterance onward the prospect spoke in the agent's Lessac voice; the transcript still read `prospect:`, so nothing looked wrong except the audio. Fixed by removing the parameter, so the wrong voice cannot be passed. **Verified by A/B on the stereo files: channel spectral-centroid difference went from 76 Hz (14%) before to 212 Hz (31%) after.**
- [x] **300 ms of dead air on every turn.** `turn_spacing` slept unconditionally before the model request. The config comment claimed token-based pacing, but `GroqLLM.suggest_turn_spacing` has no call site anywhere — it was never pacing, just silence. Now 0.

## 🔴 BLOCKER — dead air is the hardware, not the code

Measured on the deployment host (aarch64, **2 cores**, Android):

- **Piper TTS runs at RTF 1.2–3.3× real time.** Steady state 1.24×; under load 2.3–3.3×. A 5.5 s reply takes 7–18 s to synthesize.
- **Measured gaps from the stereo file: P→A median 16.3 s, A→P median 22.9 s.**
- The 300 ms target is **physically unreachable** here. Synthesis alone exceeds the whole budget, so no LLM or endpointing improvement can close it.

Tested and **rejected** (do not retry without new evidence):
- TTS warm-up at init: A/B over 4 pairs gave cold 2.33× vs warmed 2.49× — no effect. Variance 1.61–3.31 swamps it.
- `OMP_NUM_THREADS` 1/2/4: no effect (2.65/2.65/2.67 s).
- `en_US-lessac-low` tier: **slower**, RTF 2.72 vs 1.28.
- `band_limit`: not the cause. Unity gain 300–3400 Hz, −3 dB at 242 Hz / 4060 Hz, −8 dB floor, zero-phase, once per utterance, no clipping path.

**Decision needed from the user:** run the agent on a host with more cores (the only route to conversational latency), or accept multi-second gaps on-device. Nothing in the code can synthesise faster than the CPU allows.

## 🟡 OPEN

- [ ] Groq **403** on `openai/gpt-oss-20b` — not available on this account. Transient 403s also hit qwen mid-call and killed two far-end generations (`dialogue.py:226` logs only `%s`, so the exception type is lost — log `type(exc).__name__` and walk `__cause__`).
- [ ] `APIConnectionError` gets zero retries: `_with_backoff` handles `RateLimitError` and `APIError` only, and `max_retries=0` at `client.py:90` disables the SDK's own transport retry. One blip permanently loses a turn.
- [ ] `_tightened_endpointing` is dead code — written at `pipeline.py:123,153,433`, **zero readers**. Left over from the speculative-priming removal.
- [ ] `test_call_ends_after_the_meeting_is_booked` times out under pytest, passes standalone. Undiagnosed.
- [ ] `cached: 0` every call — no prompt caching, so every turn bills full prompt price and drives the 429s.
- [ ] Far end invents a name for itself ("Marcus" when the lead is Dana Whitfield) then addresses the agent as "Dana".
- [ ] `onnxruntime pthread_setaffinity_np failed` at startup. Cosmetic so far; thread limits made no difference.

## 🟡 SUBMISSION

- [ ] README leads with AssemblyAI — omits realtime STT, `agent_context`, keyterms, partial transcripts.
- [ ] Cover image, video, slide deck. Not started.
- [ ] `bypass_calling_hours: true` must be off before any real dial.

## Process note

Nine failures this project were a measurement or log surface lying, not the code — an empty prospect track reporting 0.00 s overlap, frame-level RMS inventing collisions, positional concatenation inventing overlap, a probe streaming faster than real time producing garbage transcripts, a missing log line as the only reliable signal, a pure-tone test "proving" a filter was fine, a wall-clock test that passed because its "first" call was the second-ever call, and an A/B that showed warm-up does nothing. **Verify the instrument before believing the number.**