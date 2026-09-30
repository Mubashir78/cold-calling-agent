"""Smoke test: Groq connectivity, streaming, and tool calling."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from llm.client import GroqLLM, Message
from tools.registry import ToolRegistry

load_dotenv(Path(__file__).parent.parent / ".env")

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
                    "datetime": {"type": "string"},
                },
                "required": ["prospect_name", "datetime"],
            },
        },
    }
]


async def main() -> int:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("FAIL: GROQ_API_KEY not set")
        return 1
    print(f"key present: {key[:7]}...{key[-4:]} ({len(key)} chars)")

    llm = GroqLLM(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=150, api_key=key)

    # 1. plain streaming
    messages = [
        Message(role="system", content="You are an AI sales agent for Acme Corp. Keep replies under 2 sentences."),
        Message(role="user", content="What's this about?"),
    ]
    out = ""
    async for resp in llm.generate(messages, stream=True):
        out += resp.text
    print(f"\n[streaming] {out!r}")
    if not out.strip():
        print("FAIL: empty completion")
        return 1

    # 2. tool calling
    called = {}

    async def executor(name, args):
        called[name] = args
        return {"status": "booked", "event_id": "evt_abc123"}

    tool_messages = [
        Message(role="system", content="You are an AI sales agent for Acme Corp. Book meetings when the prospect agrees."),
        Message(role="user", content="Great, Tuesday at 2pm after two works for me."),
    ]
    text = ""
    async for resp in llm.generate_with_tools(tool_messages, TOOLS, executor):
        text += resp.text

    print(f"\n[tools] text={text!r}")
    print(f"[tools] executed={called}")
    if not called:
        print("FAIL: model did not emit a tool call")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
