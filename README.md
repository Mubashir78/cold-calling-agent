# Cold Calling Agent

AI-powered outbound voice agent. Fully open-source stack, built on AssemblyAI
realtime speech-to-text.

## Architecture

```
Lead List (CSV) → Dialer/Orchestrator → Voice Pipeline
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          │                     │                     │
                  STT (AssemblyAI)          LLM (Groq)           TTS (Piper)
                  realtime streaming        qwen3.8-27b          local voices
                  server-side turn detect   tool calling         2 distinct voices
                          │                     │                     │
                          └─────────────────────┴─────────────────────┘
                                                │
                              Simulated far end (also generates)
```

## Stack

| Component | Model | Notes |
|-----------|-------|-------|
| STT | AssemblyAI `universal-3-5-pro` (v3 realtime) | streaming, `agent_context`, keyterms, partial transcripts, server-side turn detection |
| LLM | `qwen/qwen3.8-27b` via Groq | fastest time-to-first-sentence of the models measured; tool calling |
| TTS | Piper `en_US-lessac-medium` (agent), `en_US-ryan-medium` (far end) | voices downloaded on first run |
| Far end | Same LLM class, separate voice | generates its own replies from a persona — not scripted |

Both sides of the call generate. The far end is conditioned on a persona, a
goal, and a turn budget, so it can push back, ask follow-ups, and refuse.

## Hardware requirement — read this before deploying

Turn latency is dominated by **local TTS synthesis**, not by the model or the
network. Piper is a neural voice and it must run faster than real time or it
cannot keep up with a live call.

Measured on the machine this was last run on — **aarch64, 2 cores**:

| Metric | Measured |
|--------|----------|
| Piper real-time factor (steady state) | **1.24×** |
| Piper real-time factor (under load) | **2.3–3.3×** |
| Prospect→agent gap (median) | **16.3 s** |
| Agent→prospect gap (median) | **22.9 s** |

A 5.5-second reply took 7–18 seconds to synthesize. On 2 cores the synthesis
step alone exceeds the entire latency budget, so no improvement to the LLM or
the recogniser can reach a 300 ms turn gap on that hardware.

Things that were measured and did **not** help, so you need not retry them:

- Warming the voice at init (cold 2.33× vs warmed 2.49× real time — noise)
- `OMP_NUM_THREADS` 1 / 2 / 4 (no measurable difference)
- The `low` Piper tier (slower: 2.72× vs 1.28× real time)
- `band_limit`, the simulated phone-line filter (unity gain 300–3400 Hz, an
  −8 dB floor — it is not what makes audio sound muffled)

**Deploy on a machine with more cores.** The agent→prospect direction is fast
(~0.22 s) because the far end pre-generates and pre-synthesizes while the agent
is still speaking. The agent side does not, and that is where synthesis sits on
the critical path.

## Quick Start

```bash
# 1. Install dependencies (includes piper-tts)
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: ASSEMBLYAI_API_KEY and GROQ_API_KEY are both required.
# Piper voice models download automatically on first run into data/voices/.

# 3. Add leads
# Edit data/leads.csv with your prospects

# 4. Check compliance settings before any real dial
# config/config.yaml sets bypass_calling_hours: true so a demo can run at any
# hour. Set it to false before calling anyone who has not opted in.

# 5. Run
python -m src.main
```

Recordings and transcripts are written to `data/calls/` (stereo WAV with the
agent on the left channel and the far end on the right, plus a text transcript).

## Project Structure

```
cold-calling-agent/
├── config/
│   └── config.yaml          # Main configuration
├── data/
│   ├── leads.csv            # Lead list
│   └── internal_dnc.txt     # Internal DNC list
├── src/
│   ├── audio/               # Audio processing
│   ├── stt/                 # Speech-to-text (AssemblyAI realtime)
│   ├── llm/                 # LLM client (Groq)
│   ├── tts/                 # Text-to-speech (Piper)
│   ├── compliance/          # Compliance gate (DNC, calling hours)
│   ├── dialer/              # Dialer orchestrator
│   ├── tools/               # Tool registry (CRM, calendar, DNC)
│   ├── webhooks/            # Webhook handlers
│   ├── pipeline.py          # Voice pipeline (STT → LLM → TTS)
│   └── main.py              # Entry point
├── logs/
└── tests/
```

## Compliance

The agent includes a compliance gate that runs before every dial:
- Internal DNC list checking
- Calling hours enforcement (8 AM - 9 PM recipient local time)
- AI disclosure in opener
- Recording disclosure
- Max call duration limit

## Tools

The LLM can call these tools during a conversation:
- `book_meeting` — Book a meeting with the prospect
- `log_disposition` — Log call outcome to CRM
- `honor_dnc` — Add prospect to DNC list
- `mark_callback` — Schedule a callback

## License

MIT
