# Session state

Paused at host shutdown. Nothing running. Resume from "next steps".

## Status

- 106 tests; 1 failing at pause (`test_turn_spacing_is_applied`) — **fix already
  applied, not yet re-verified**. Cause: refactor moved `turn_spacing` into
  `_respond_with_tools` only, so the plain-reply path skipped it.
- Full campaign has never completed cleanly end-to-end with working
  turn-taking. Closest: run7 (4 min, 429s) and run8 (12 min, mostly backoff).

## Next steps, in order

1. `cd /root/cold-calling-agent && source .venv/bin/activate && python -m pytest -q`
   — expect 106 passed. Verify before anything else.
2. Run one call, not four, while the org token budget is shared/exhausted:
   `python -m src.main` after trimming `data/leads.csv` to the interested lead.
3. `python tests/check_overlap.py` — measures simultaneous speech in the
   two-channel WAV. **The user said they would verify overlap themselves by
   listening**, so ask before running.
4. Copy a good call to `/storage/emulated/0/Download/`.

## Live problems

**Shared rate limit is the blocker, not the code.** Groq returns
429 with `Limit 200000, Used 199805` — org-wide and consumed externally.
A whole campaign needs only ~17k tokens (~626/turn, 28 turns), so this
project is not the cause. Symptoms: waits escalate 35s → 55s, single calls
hang ~10 min.

Mitigations in place: honor the server's suggested wait, `max_retries=0`
on the SDK so our backoff sees errors, 90s `retry_budget` per call,
45s `rate_limit_retry_delay` at the orchestrator, and a throttled turn
degrades to a plain reply instead of killing the call.

**Groq's tier varies by account.** An earlier error said `Limit 7000`. Read
the real number from the error body, never hardcode it.

**Model is fixed by capability, not preference.** `qwen/qwen3.8-27b` is the
only model on this key that reliably tool-calls. `openai/gpt-oss-20b` and
`-120b` each hit 1/2 required tool calls (`tests/probe_models.py`).

## Gotchas hit repeatedly

- **`if not chunk` on a numpy array raises.** Bit me three times
  (`sim/recorder.py`, `pipeline._is_speech`, prospect filter). Use
  `is None` / `len() == 0` and coerce explicitly.
- **`pkill -f` matches its own command line** and kills the calling shell.
  Use `grep -E "pat[t]ern"` so the pattern cannot match itself.
- **`grep -vE ... > file` buffers**, so the log looks empty mid-run. Read
  `logs/agent.log` for live progress instead.
- **STT endpointing must be fed continuously.** Wrapping each chunk
  (`_single`) re-creates the segmenter and breaks it; the whole call must be
  one stream. `src/stt/endpointing.py` is chunk-size independent — keep it.

## Key facts

- Project root `/root/cold-calling-agent`, venv in `.venv`, `pythonpath = src`
- STT `whisper-large-v3-turbo` via Groq; TTS Piper local, two voices
  (`en_US-lessac-medium` agent, `en_US-ryan-medium` prospect), 121 MB in `data/voices/`
- `src/turnlock.py` enforces half-duplex turn taking, 350 ms silence
  threshold, `bypass_silence` for the opening line only
- Compliance fails closed; `bypass_calling_hours` is demo-only and can never
  bypass DNC (test-pinned)
- No tool accepts `phone_number`; the server injects the dialed number.
  Test asserts this — the LLM once wrote the literal string "unknown" to DNC.
- Storage flat at ~47 GB free; project 470 MB. Memory is the real constraint
  (~1.2 GB usable), which is why STT is hosted.

## Not real (simulation)

- PSTN leg: no SIP trunk, no LiveKit bridge. `handle_call` swaps in a
  simulated prospect.
- CRM/calendar/DNC: in-memory FastAPI stub on :8080, run in-process.
- All lead numbers are `555-01xx` (unassignable) and fictional names.
