"""Budget and usage reporting.

Reads Groq's rate-limit headers so a run can report what it actually cost
and what is left, instead of guessing from the number of calls made.

The headers are the only trustworthy source: published limits vary per model
and per organisation, and a run can 429 on any of four independent budgets
(RPM, RPD, TPM, TPD) while looking healthy on the others.

Costs one minimal request (~70 prompt tokens) to read the current state.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from groq import AsyncGroq

load_dotenv(Path(__file__).parent.parent / ".env")

MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# Per-model free-plan ceilings, from the published rate-limit table.
PUBLISHED = {
    "openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "openai/gpt-oss-20b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "qwen/qwen3.8-27b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "whisper-large-v3-turbo": {"rpm": 20, "rpd": 2000, "ash": 7200, "asd": 28800},
}


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _pct(remaining: int, limit: int) -> str:
    if remaining < 0 or limit <= 0:
        return "?"
    return f"{100.0 * remaining / limit:5.1f}%"


def parse_headers(headers) -> dict:
    """Pull the four budgets out of a response's rate-limit headers.

    Groq's header names are misleading: x-ratelimit-limit-requests is always
    per *day* and x-ratelimit-limit-tokens is always per *minute*, whatever
    the model's other ceilings are. Daily token headers carry a -day suffix.
    """
    h = {k.lower(): v for k, v in dict(headers or {}).items()}
    return {
        "rpd_remaining": _int(h.get("x-ratelimit-remaining-requests")),
        "rpd_limit": _int(h.get("x-ratelimit-limit-requests")),
        "tpm_remaining": _int(h.get("x-ratelimit-remaining-tokens")),
        "tpm_limit": _int(h.get("x-ratelimit-limit-tokens")),
        "tpd_remaining": _int(h.get("x-ratelimit-remaining-tokens-day")),
        "tpd_limit": _int(h.get("x-ratelimit-limit-tokens-day")),
        "reset_tokens": h.get("x-ratelimit-reset-tokens", ""),
        "reset_tokens_day": h.get("x-ratelimit-reset-tokens-day", ""),
        "retry_after": h.get("retry-after", ""),
    }


async def check(model: str, key: str) -> dict:
    """One minimal request, read the rate-limit headers off the response.

    Uses httpx directly: the SDK does not surface response headers on the
    parsed object, and the headers are the only authoritative budget signal.
    """
    import httpx

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
        except Exception as exc:
            return {"model": model, "state": type(exc).__name__,
                    "detail": str(exc)[:150]}

    if resp.status_code == 429:
        try:
            message = resp.json()["error"]["message"]
        except Exception:
            message = resp.text[:150]
        kind = ("TPD" if "per day" in message else
                "TPM" if "per minute" in message else
                "RPD" if "requests per day" in message else "429")
        state = parse_headers(resp.headers)
        state.update({"model": model, "state": kind, "detail": message[:200]})
        return state

    if resp.status_code >= 400:
        return {"model": model, "state": f"HTTP {resp.status_code}",
                "detail": resp.text[:150]}

    state = parse_headers(resp.headers)
    state["model"] = model
    state["state"] = "ok"
    try:
        state["probe_tokens"] = resp.json()["usage"]["prompt_tokens"]
    except Exception:
        state["probe_tokens"] = 0
    return state


async def main() -> int:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set")
        return 1

    print("Groq budget (limits are per model; each has its own allowance)\n")
    for model in MODELS:
        s = await check(model, key)
        if s["state"] != "ok":
            print(f"  {model:<24} EXHAUSTED on {s['state']}")
            continue
        pub = PUBLISHED.get(model, {})
        print(f"  {model}")
        tpd_r, tpd_l = s["tpd_remaining"], (s["tpd_limit"] or pub.get("tpd", 0))
        tpm_r, tpm_l = s["tpm_remaining"], (s["tpm_limit"] or pub.get("tpm", 0))
        rpd_r, rpd_l = s["rpd_remaining"], (s["rpd_limit"] or pub.get("rpd", 0))
        if tpd_l:
            print(f"      TPD  {tpd_r:>7} / {tpd_l:<7} left  {_pct(tpd_r, tpd_l)}"
                  f"   resets in {s['reset_tokens_day'] or 'n/a'}")
        if tpm_l:
            print(f"      TPM  {tpm_r:>7} / {tpm_l:<7} left  {_pct(tpm_r, tpm_l)}")
        if rpd_l:
            print(f"      RPD  {rpd_r:>7} / {rpd_l:<7} left  {_pct(rpd_r, rpd_l)}")
    print(f"\n  (probe cost ~{s.get('probe_tokens', 0)} tokens each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
