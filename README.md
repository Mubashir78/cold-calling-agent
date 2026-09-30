# Cold Calling Agent

AI-powered outbound voice agent for vibe coding competitions. Fully open-source stack with sub-500ms latency.

## Architecture

```
Lead List (CSV) → Dialer/Orchestrator → LiveKit (WebRTC/SIP) → Voice Pipeline
                                                                    │
                                              ┌─────────────────────┤
                                              │                     │
                                         STT (Whisper)          TTS (Piper)
                                              │                     │
                                              └────── LLM (Groq) ──┘
```

## Stack

| Component | Model | License | Latency |
|-----------|-------|---------|---------|
| STT | faster-whisper (small) | MIT | ~200ms |
| LLM | Llama 3.1 8B (Groq) | Llama Community | ~150ms |
| TTS | Piper | MIT | ~100ms |
| **Total** | | | **~450ms** |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Piper TTS
pip install piper-tts

# 3. Configure
cp .env.example .env
# Edit .env with your Groq API key

# 4. Add leads
# Edit data/leads.csv with your prospects

# 5. Run
python -m src.main
```

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
│   ├── stt/                 # Speech-to-text (faster-whisper)
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
