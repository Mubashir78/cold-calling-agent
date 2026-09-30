"""Re-probe tool calling across candidate models.

The first probe was single-turn and checked only the first tool call, which
under-reports reasoning models like gpt-oss that emit a `reasoning` field
before the actual function call. This one:

  * scans every chunk for tool calls, not just the first
  * uses three different conversation turns
  * reports TTFT (time to first token) separately from tool latency
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv

from groq import AsyncGroq

load_dotenv(Path(__file__).parent.parent / ".env")

CANDIDATES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting with the prospect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prospect_name": {"type": "string"},
                    "time_slot": {"type": "string", "description": "ISO 8601"},
                },
                "required": ["prospect_name", "time_slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "honor_dnc",
            "description": "Add the prospect to the do-not-call list.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = (
    "You are an AI sales agent for Acme Corp. Keep replies under 2 sentences. "
    "When the prospect agrees to a specific time, call book_meeting. "
    "If they ask to be removed from the call list, call honor_dnc immediately."
)

TURNS = [
    ("Great, Tuesday at 2pm after two works for me.", "book_meeting"),
    ("Actually no, take me off your calling list.", "honor_dnc"),
    ("What's this about?", None),  # no tool expected
]


async def probe(client, model: str) -> dict:
    row = {"model": model, "hits": 0, "expected": 0, "false_pos": 0,
           "ttft": [], "tok": 0, "err": ""}
    for user_text, expected in TURNS:
        start = time.time()
        first = None
        names, tok = [], 0
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=150,
                stream=True,
            )
            async for chunk in stream:
                if first is None:
                    first = time.time() - start
                usage = getattr(chunk, "usage", None)
                if usage:
                    tok += getattr(usage, "completion_tokens", 0) or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                for tc in (getattr(delta, "tool_calls", None) or []):
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        except Exception as exc:
            row["err"] = f"{type(exc).__name__}: {str(exc)[:90]}"
            continue
        if first:
            row["ttft"].append(first)
        row["tok"] += tok
        if expected:
            row["expected"] += 1
            if expected in names:
                row["hits"] += 1
        elif names:
            row["false_pos"] += 1
    return row


async def main() -> int:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
    for model in CANDIDATES:
        r = await probe(client, model)
        ttft = f"{sum(r['ttft'])/len(r['ttft']):.2f}" if r["ttft"] else "n/a"
        ok = r["hits"] == r["expected"] and r["false_pos"] == 0
        mark = "OK " if ok else "-- "
        print(f"{mark}{r['model']:<24} tool_hits={r['hits']}/{r['expected']} "
              f"false_pos={r['false_pos']} ttft={ttft}s out_tok={r['tok']}")
        if r["err"]:
            print(f"   err: {r['err']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
