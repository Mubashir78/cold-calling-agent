"""LLM client for conversation management using Groq API."""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import httpx
from groq import APIError, AsyncGroq, RateLimitError

logger = logging.getLogger(__name__)


class MalformedToolCall(RuntimeError):
    """The model emitted a tool call whose arguments were not valid JSON.

    Usually a reasoning model whose output was truncated before the
    arguments closed. Recoverable by replying without tool schemas.
    """


class DailyTokenLimitExceeded(RuntimeError):
    """The account's per-day token budget is spent.

    Distinct from a transient rate limit: no amount of waiting helps, so
    callers should stop immediately rather than hold a call open.
    """


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


class GroqLLM:
    """LLM client using Groq API with streaming and tool calling."""

    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",
        temperature: float = 0.7,
        max_tokens: int = 150,
        api_key: Optional[str] = None,
        max_retries: int = 12,
        retry_delay: float = 5.0,
        max_retry_wait: float = 45.0,
        retry_budget: float = 90.0,
        tokens_per_minute: int = 8000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_wait = max_retry_wait
        # Total wall-clock allowed for retries. A prospect will not hold a
        # line for ten minutes, so an unbounded backoff helps nobody: the
        # call dies either way, just later.
        self.retry_budget = retry_budget
        # Free-plan budget for this model, from the published rate-limit
        # table. Limits are per model, so each has its own allowance.
        self.tokens_per_minute = tokens_per_minute
        self.usage_log: list[dict] = []
        # No SDK-level retries: they fire on their own schedule and burn the
        # attempt budget before our backoff ever sees the error.
        self.client = AsyncGroq(
            api_key=api_key,
            max_retries=0,
            timeout=httpx.Timeout(60.0, read=90.0),
        )

    @staticmethod
    def _error_text(exc: Exception) -> str:
        """Flatten an API error into searchable text."""
        for source in (getattr(exc, "body", None), getattr(exc, "response", None)):
            if source is None:
                continue
            if isinstance(source, (bytes, bytearray)):
                source = source.decode("utf-8", "replace")
            if isinstance(source, str):
                return source
            try:
                return json.dumps(source)
            except (TypeError, ValueError):
                continue
        return str(exc)

    @classmethod
    def limit_kind(cls, exc: Exception) -> str:
        """Which budget was hit: 'day', 'minute', or 'unknown'.

        The distinction matters enormously. A per-minute limit clears in
        seconds and backoff is exactly right. A per-*day* limit does not clear
        on any useful timescale, so retrying just burns wall-clock time while
        the answer cannot change.
        """
        text = cls._error_text(exc).lower()
        if "tokens per day" in text or "(tpd)" in text:
            return "day"
        if "tokens per minute" in text or "itpm" in text:
            return "minute"
        if "requests per minute" in text or "rpm" in text:
            return "minute"
        return "unknown"

    @classmethod
    def is_daily_exhausted(cls, exc: Exception) -> bool:
        return cls.limit_kind(exc) == "day"

    @staticmethod
    def _suggested_wait(exc: Exception, fallback: float) -> float:
        """Pull the server's own retry delay out of the 429 body.

        Groq reports e.g. "Please try again in 30.14s". Ignoring that and
        retrying on a fixed short delay just burns the attempts. Note the
        server suggests a short wait even for daily limits, where no useful
        wait exists — callers must check `limit_kind` first.
        """
        text = GroqLLM._error_text(exc)
        if not text:
            return fallback
        match = re.search(r"try again in\s*([0-9.]+)\s*s", text, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return fallback
        return fallback

    async def _with_backoff(self, coro_factory):
        """Call the API, backing off on 429 rather than dying mid-call.

        A cold-calling agent that drops the prospect because of a rate limit
        is worse than one that waits: the call is already connected.
        """
        last: Exception | None = None
        started = asyncio.get_event_loop().time()
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except APIError as exc:
                # Recoverable tool-shape failures: drop the schemas for this
                # turn and reply in plain text. Losing a turn beats losing the
                # call. Groq words these several different ways, so match
                # loosely on the tool-shape family rather than one phrase.
                text = str(exc).lower()
                recoverable = (
                    "tool call" in text
                    or "called a tool" in text
                    or "tool choice" in text
                    or "tool call validation" in text
                )
                if recoverable:
                    logger.warning("Tool-shape error (%s); replying as text",
                                   text[:60])
                    raise MalformedToolCall(str(exc)) from exc
                raise
            except RateLimitError as exc:
                last = exc
                if self.is_daily_exhausted(exc):
                    # A daily token budget will not clear on any timescale a
                    # live call can wait for. Retrying is pure dead air.
                    logger.error(
                        "Daily token budget exhausted; not retrying. "
                        "The account's per-day limit resets on its own schedule."
                    )
                    raise DailyTokenLimitExceeded(
                        "Groq daily token budget exhausted"
                    ) from exc
                elapsed = asyncio.get_event_loop().time() - started
                if attempt == self.max_retries:
                    break
                if elapsed >= self.retry_budget:
                    logger.error(
                        "Rate limited for %.0fs; abandoning (budget %.0fs)",
                        elapsed, self.retry_budget,
                    )
                    break
                wait = self._suggested_wait(exc, self.retry_delay * (attempt + 1))
                wait = min(max(wait, 1.0), self.max_retry_wait)
                if elapsed + wait > self.retry_budget:
                    wait = max(0.0, self.retry_budget - elapsed)
                    if wait < 1.0:
                        break
                logger.warning(
                    "Rate limited; waiting %.1fs (attempt %d/%d, %.0fs elapsed)",
                    wait, attempt + 1, self.max_retries, elapsed,
                )
                await asyncio.sleep(wait)
        raise last  # type: ignore[misc]

    def suggest_turn_spacing(self, tokens_per_turn: int) -> float:
        """Seconds between turns that keep the call inside the TPM budget.

        Derived from the published limit rather than guessed: the free plan
        allows 8K tokens per minute for this model, and an uncached prompt is
        billed in full, so a turn every 1.5s will 429 a few turns in.
        """
        if tokens_per_turn <= 0:
            return 0.0
        turns_per_minute = max(1.0, self.tokens_per_minute / tokens_per_turn)
        return 60.0 / turns_per_minute

    def _record_usage(self, usage) -> None:
        if not usage:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.usage_log.append({
            "t": time.monotonic(),
            "prompt": prompt,
            "completion": completion,
            "cached": cached,
            "billed": prompt - cached + completion,
        })
        logger.debug(
            "usage prompt=%d completion=%d cached=%d billed=%d",
            prompt, completion, cached, prompt - cached + completion,
        )

    def usage_summary(self) -> dict:
        if not self.usage_log:
            return {"calls": 0, "billed": 0, "cached": 0, "max_prompt": 0}
        return {
            "calls": len(self.usage_log),
            "billed": sum(u["billed"] for u in self.usage_log),
            "cached": sum(u["cached"] for u in self.usage_log),
            "max_prompt": max(u["prompt"] for u in self.usage_log),
        }

    async def generate(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        stream: bool = True,
    ) -> AsyncIterator[LLMResponse]:
        """Generate a response from the LLM."""
        raw_messages = []
        for msg in messages:
            raw_msg = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                raw_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                raw_msg["tool_call_id"] = msg.tool_call_id
            raw_messages.append(raw_msg)

        kwargs = {
            "model": self.model,
            "messages": raw_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if stream:
            stream_resp = await self._with_backoff(
                lambda: self.client.chat.completions.create(
                    stream=True, **kwargs
                )
            )
            async for chunk in stream_resp:
                self._record_usage(getattr(chunk, "usage", None))
                delta = chunk.choices[0].delta
                tool_calls = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                yield LLMResponse(
                    text=delta.content or "",
                    tool_calls=tool_calls,
                    finish_reason=chunk.choices[0].finish_reason or "",
                    usage={
                        "prompt_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
                        "completion_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
                    },
                )
        else:
            resp = await self._with_backoff(
                lambda: self.client.chat.completions.create(**kwargs)
            )
            self._record_usage(getattr(resp, "usage", None))
            msg = resp.choices[0].message
            yield LLMResponse(
                text=msg.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
                finish_reason=resp.choices[0].finish_reason or "",
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                },
            )

    async def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        tool_executor: Callable,
        max_tool_rounds: int = 3,
    ) -> AsyncIterator[LLMResponse]:
        """Generate with tool calling loop."""
        for _ in range(max_tool_rounds):
            response_text = ""
            tool_calls = []

            async for resp in self.generate(messages, tools=tools, stream=True):
                response_text += resp.text
                tool_calls.extend(resp.tool_calls)
                yield resp

            if not tool_calls:
                return

            # Yield to the event loop before the tool round-trip. Without
            # this, a long tool sequence starves other tasks and, on a
            # 429 retry, the whole pipeline stalls behind one backoff.
            await asyncio.sleep(0)

            # Execute tool calls
            messages.append(Message(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls,
            ))

            for tc in tool_calls:
                try:
                    tool_args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    logger.warning("Unparseable tool args: %r", tc["function"]["arguments"])
                    tool_args = {}
                tool_name = tc["function"]["name"]
                logger.info(f"Tool call: {tool_name}({tool_args})")

                result = await tool_executor(tool_name, tool_args)

                messages.append(Message(
                    role="tool",
                    content=json.dumps(result),
                    tool_call_id=tc["id"],
                ))
