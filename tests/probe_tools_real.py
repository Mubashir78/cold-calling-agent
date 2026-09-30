"""Tool-calling probe using the real registry schemas.

The earlier probe used a hand-rolled honor_dnc with empty properties, which is
unrepresentative: some models handle an empty schema poorly. This uses the
actual ToolRegistry definitions, so the answer is directly actionable.

Also accounts for gpt-oss being a reasoning model: reasoning lands in a
separate field, and tool calls may arrive in later chunks.
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

from tools.registry import ToolRegistry

load_dotenv(Path(__file__).parent.parent / ".env")

CANDIDATES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

SYSTEM = (
    "You are an AI sales agent for Acme Corp. Keep replies under 2 sentences. "
    "When the prospect agrees to a specific time, call book_meeting. "
    "If they ask to be removed, call honor_dnc immediately."
)

# (prospect says, tool that must fire)
TURNS = [
    ("Great, Tuesday at 2pm after two works for me.", "book_meeting"),
    ("Actually no, take me off your calling list.", "honor_dnc"),
    ("No thanks, we're all set.", None),
]


def real_tools() -> list[dict]:
    return ToolRegistry().get_tools_for_llm()


async def probe(client: AsyncGroq, model: str) -> dict:
    row = {"model": model, "hits": 0, "expected": 0, "false_pos": 0,
           "ttft": [], "billed": 0, "cached": 0, "err": ""}
    for user_text, expected in TURNS:
        start = time.time()
        first = None
        names = []
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                tools=real_tools(),
                tool_choice="auto",
                temperature=0.7,
                max_tokens=200,
                stream=True,
            )
            async for chunk in stream:
                if first is None:
                    first = time.time() - start
                u = getattr(chunk, "usage", None)
                if u:
                    d = getattr(u, "prompt_tokens_details", None)
                    cached = getattr(d, "cached_tokens", 0) or 0 if d else 0
                    row["billed"] += (u.prompt_tokens or 0) - cached + (u.completion_tokens or 0)
                    row["cached"] += cached
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                for tc in (getattr(delta, "tool_calls", None) or []):
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        except Exception as exc:
            msg = str(exc)
            kind = ("TPD" if "per day" in msg else
                    "TPM" if "per minute" in msg else type(exc).__name__)
            row["err"] = kind
            continue
        if first:
            row["ttft"].append(first)
        if expected:
            row["expected"] += 1
            if expected in names:
                row["hits"] += 1
            else:
                print(f"    MISS on {user_text[:34]!r} -> got {names}")
        elif names:
            row["false_pos"] += 1
            print(f"    FALSE POSITIVE on {user_text[:30]!r} -> {names}")
    return row


async def main() -> int:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=90.0)
    print("schema size (chars):", len(json.dumps(real_tools())), "\n")
    for model in CANDIDATES:
        r = await probe(client, model)
        ttft = f"{sum(r['ttft'])/len(r['ttft']):.2f}" if r["ttft"] else "n/a"
        ok = r["hits"] == r["expected"] and r["false_pos"] == 0 and not r["err"]
        print(f"{'OK ' if ok else '-- '}{r['model']:<22} "
              f"hits={r['hits']}/{r['expected']} fp={r['false_pos']} "
              f"ttft={ttft}s billed={r['billed']} cached={r['cached']} "
              f"{r['err']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
