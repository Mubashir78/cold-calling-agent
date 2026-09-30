"""Why do both gpt-oss models miss the booking turn?

Hypothesis: book_meeting requires `prospect_name`, and the prospect's line
supplies no name, so the models decline to call rather than invent one. Qwen
guesses a name and succeeds.

If so, the fix is architectural, not prompt tuning: the system already knows
the lead, so name and company should be injected server-side exactly like the
phone number. The model then only has to supply a time.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv

from groq import AsyncGroq

load_dotenv(Path(__file__).parent.parent / ".env")

CANDIDATES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

SYSTEM = (
    "You are an AI sales agent for Acme Corp. Keep replies under 2 sentences. "
    "When the prospect agrees to a specific time, call book_meeting. "
    "If they ask to be removed, call honor_dnc immediately."
)

# Variant A: name required (today's schema)
BOOKING_WITH_NAME = {
    "type": "function",
    "function": {
        "name": "book_meeting",
        "description": "Book a meeting. Call once, after they agree to a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "prospect_name": {"type": "string"},
                "time_slot": {"type": "string", "description": "ISO 8601"},
                "duration_minutes": {"type": "integer", "default": 30},
            },
            "required": ["prospect_name", "time_slot"],
        },
    },
}

# Variant B: name and duration removed; system injects them
BOOKING_MINIMAL = {
    "type": "function",
    "function": {
        "name": "book_meeting",
        "description": "Book a meeting at the agreed time. Call once.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_slot": {"type": "string", "description": "ISO 8601"},
            },
            "required": ["time_slot"],
        },
    },
}

DNC = {
    "type": "function",
    "function": {
        "name": "honor_dnc",
        "description": "Add the prospect to the do-not-call list. Call the "
                       "moment they ask to be removed.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    },
}

TURN = "Great, Tuesday at 2pm after two works for me."


async def run(client, model: str, tools, label: str) -> tuple[int, float, int]:
    names, first, billed = [], None, 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TURN},
            ],
            tools=tools, tool_choice="auto", temperature=0.7,
            max_tokens=200, stream=True,
        )
        async for chunk in stream:
            if first is None:
                first = time.time()
            u = getattr(chunk, "usage", None)
            if u:
                billed += (u.prompt_tokens or 0) + (u.completion_tokens or 0)
            if not chunk.choices:
                continue
            for tc in (getattr(chunk.choices[0].delta, "tool_calls", None) or []):
                if tc.function and tc.function.name:
                    names.append(tc.function.name)
    except Exception as exc:
        msg = str(exc)
        kind = "TPD" if "per day" in msg else ("TPM" if "per minute" in msg
                                              else type(exc).__name__)
        return -1, kind, 0
    return (1 if "book_meeting" in names else 0), first, billed


async def main() -> int:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=90.0)
    for model in CANDIDATES:
        for tools, label in (
            ([BOOKING_WITH_NAME, DNC], "name-required"),
            ([BOOKING_MINIMAL, DNC], "minimal     "),
        ):
            hit, first, billed = await run(client, model, tools, label)
            flag = "CALLS" if hit == 1 else ("SKIPS" if hit == 0 else str(first))
            print(f"{model:<22} {label}  {flag:<6} billed={billed}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
