"""Where does the turn latency actually go?

One request per model. Measures time-to-first-token, time-to-first-sentence
(the point TTS can start), and TTS synthesis. Reasoning models spend tokens
before the answer, so TTFT alone can be misleading — a model can be fast to
first token and still slow to first sentence.
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv

from groq import AsyncGroq

load_dotenv(Path(__file__).parent.parent / ".env")

import main as M
from dialer.orchestrator import Lead
from tools.registry import ToolRegistry

SENT = re.compile(r"(?<=[.!?])\s+")

TOOLS = ToolRegistry().get_tools_for_llm()


async def measure(client, model: str, prompt_msgs, label: str) -> None:
    ttft = first_sentence = None
    t0 = time.monotonic()
    buf = ""
    try:
        stream = await client.chat.completions.create(
            model=model, messages=prompt_msgs, tools=TOOLS,
            tool_choice="auto", temperature=0.7, max_tokens=1024, stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            now = time.monotonic() - t0
            if ttft is None and (delta.content or delta.reasoning):
                ttft = now
            if delta.content:
                buf += delta.content
                if first_sentence is None and SENT.split(buf)[0].rstrip().endswith((".", "!", "?")):
                    first_sentence = now
    except Exception as exc:
        msg = str(exc)
        kind = "TPD" if "per day" in msg else ("TPM" if "per minute" in msg
                                              else type(exc).__name__)
        print(f"  {model:<22} {kind}")
        return
    print(f"  {model:<22} ttft={ttft if ttft else float('nan'):.2f}s  "
          f"first_sentence={first_sentence if first_sentence else float('nan'):.2f}s  {label}")


async def main() -> int:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=90.0)

    sp = M.SYSTEM_PROMPT.format(company_name=M.COMPANY_NAME, value_prop=M.VALUE_PROP)
    lead = Lead(phone_number="+14155550101", name="Dana Whitfield",
                company="Northwind Logistics")
    system = M._prompt_for(lead, sp)
    convo = system + (
        "\n\nprospect: Okay. Well, how much does it cost?\n"
        "agent: Pricing varies based on call volume.\n"
        "prospect: That sounds reasonable. Yeah, let's do it."
    )

    for model in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"):
        await measure(
            client, model,
            [{"role": "system", "content": system},
             {"role": "user", "content": convo[-600:]}],
            "real prompt",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
