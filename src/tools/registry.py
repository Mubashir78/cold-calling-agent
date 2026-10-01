"""Tool registry — LLM function calling for CRM, calendar, and DNC operations."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Coroutine]


class ToolRegistry:
    """Registry of tools the LLM can call during a conversation."""

    def __init__(
        self,
        crm_webhook: str = "http://localhost:8080/api/crm",
        calendar_webhook: str = "http://localhost:8080/api/calendar",
        dnc_webhook: str = "http://localhost:8080/api/dnc",
        timeout: float = 5.0,
    ):
        self.crm_webhook = crm_webhook
        self.calendar_webhook = calendar_webhook
        self.dnc_webhook = dnc_webhook
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._tools: dict[str, ToolDefinition] = {}
        # Per-call state. Not safe for concurrent calls on one registry
        # instance; main.py constructs one per campaign with concurrency 1.
        self._current_call: dict = {}
        self._booked_this_call = False
        self._logged_this_call = False
        self._dnc_this_call = False
        self._register_default_tools()

    def start_call(self) -> None:
        """Reset per-call state before dialing."""
        self._booked_this_call = False
        self._logged_this_call = False
        self._dnc_this_call = False

    async def aclose(self) -> None:
        await self._client.aclose()

    def _register_default_tools(self):
        """Register the default cold-calling tools.

        Note: no tool takes a phone_number. The agent does not reliably know
        the number it dialed, and when it guesses it writes 'unknown' into
        the DNC list or books against the wrong record. The number is bound
        per-call by `bind_call` instead.
        """
        self.register(ToolDefinition(
            name="book_meeting",
            description="Book a meeting at the agreed time. Call once.",
            parameters={
                "type": "object",
                "properties": {
                    "time_slot": {"type": "string", "description": "ISO 8601"},
                },
                "required": ["time_slot"],
            },
            handler=self._book_meeting,
        ))

        self.register(ToolDefinition(
            name="log_disposition",
            description="Log the call outcome in the CRM. Call once, at the end.",
            parameters={
                "type": "object",
                "properties": {
                    "disposition": {
                        "type": "string",
                        "enum": ["booked", "callback", "not_interested", "dnc",
                                 "voicemail", "no_answer", "wrong_number"],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["disposition"],
            },
            handler=self._log_disposition,
        ))

        self.register(ToolDefinition(
            name="honor_dnc",
            description="Add the prospect to the Do Not Call list. Call the "
                        "moment they ask to be removed.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": [],
            },
            handler=self._honor_dnc,
        ))

        self.register(ToolDefinition(
            name="mark_callback",
            description="Schedule a callback.",
            parameters={
                "type": "object",
                "properties": {
                    "callback_time": {"type": "string", "description": "ISO 8601"},
                    "notes": {"type": "string"},
                },
                "required": ["callback_time"],
            },
            handler=self._mark_callback,
        ))

    def bind_call(self, phone_number: str, prospect_name: str = "", company: str = ""):
        """Bind the real number for the call in progress.

        Tools read this instead of taking a number as an argument, so the
        number written to the CRM or the DNC list is always the one dialed.
        """
        self._current_call = {
            "phone_number": phone_number,
            "prospect_name": prospect_name,
            "company": company,
        }

    def register(self, tool: ToolDefinition):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get_tools_for_llm(self) -> list[dict]:
        """Get tool definitions in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def _inject_call_context(self, name: str, arguments: dict) -> dict:
        """Add the dialed number and guard against duplicate bookings."""
        call = self._current_call
        number = call.get("phone_number")
        if not number:
            return {"success": False,
                    "error": "No call in progress; refusing to record a "
                             "placeholder number."}

        args = dict(arguments)
        if name == "book_meeting":
            if self._booked_this_call:
                return {"success": False,
                        "error": "A meeting is already booked for this call. "
                                 "Do not book a second one."}
            # The prospect's name and company are facts the system holds, not
            # facts the model can know. These *override* anything the model
            # supplied: a volunteered name is a guess, and booking a meeting
            # against a guess puts it on the wrong calendar.
            if call.get("prospect_name"):
                args["prospect_name"] = call["prospect_name"]
            if call.get("company"):
                args["company"] = call["company"]
            args.setdefault("duration_minutes", 30)
            try:
                self._validate_future_datetime(
                    args.get("time_slot", ""), "time_slot"
                )
            except ValueError as exc:
                return {"success": False, "error": str(exc)}
        elif name == "mark_callback":
            try:
                self._validate_future_datetime(
                    args.get("callback_time", ""), "callback_time"
                )
            except ValueError as exc:
                return {"success": False, "error": str(exc)}
        return args

    @property
    def booked_this_call(self) -> bool:
        """Whether a meeting has been booked on this call.

        Read by the pipeline to know the call has reached its purpose. Without
        a public accessor the pipeline would have to reach into a private
        field, and the "the call is over" decision would be invisible to
        anything else watching the call.
        """
        return self._booked_this_call

    @property
    def terminal(self) -> bool:
        """Whether the call has achieved a result that ends it.

        A booking or a DNC request is a completed outcome, not a topic. The
        agent still gets to finish the turn it is speaking -- that turn carries
        the confirmation -- but the call ends there rather than the agent
        returning to confirm a second and third time.
        """
        return self._booked_this_call or self._dnc_this_call

    async def execute(self, name: str, arguments: dict) -> dict:
        """Execute a tool call."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}

        tool = self._tools[name]

        if name in ("book_meeting", "log_disposition", "honor_dnc", "mark_callback"):
            if name == "book_meeting" and self._booked_this_call:
                return {"success": False,
                        "error": "Meeting already booked on this call."}
            if name == "honor_dnc" and self._dnc_this_call:
                return {"success": False, "error": "Already on the DNC list."}
            if name == "log_disposition" and self._logged_this_call:
                return {"success": False,
                        "error": "Disposition already logged on this call."}

            args = self._inject_call_context(name, arguments)
            if isinstance(args, dict) and "error" in args:
                return args
            arguments = args
            arguments["phone_number"] = self._current_call["phone_number"]

        try:
            result = await tool.handler(**arguments)
            if name == "book_meeting":
                self._booked_this_call = True
            if name == "log_disposition":
                self._logged_this_call = True
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {"success": False, "error": str(e)}

    async def _book_meeting(self, time_slot: str, **kwargs) -> dict:
        """Book a meeting.

        prospect_name and company arrive via _inject_call_context, which binds
        them from the lead actually dialed. The model is never asked for them:
        it cannot know them, and asking made it guess or decline to call.
        """
        payload = {"time_slot": time_slot, **kwargs}
        resp = await self._client.post(
            f"{self.calendar_webhook}/book", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _validate_future_datetime(value: str, field: str) -> str:
        """Reject a datetime the model invented in the past or malformed.

        The model has no reliable sense of today's date, so an unvalidated
        time_slot books meetings in 2024. This is a hard error, not a warning.
        """
        from datetime import datetime, timezone

        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be ISO 8601, got {value!r}"
            ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        if parsed < now:
            raise ValueError(
                f"{field} {value} is in the past (now is {now.isoformat()}). "
                f"Ask the prospect for a time and use the real current date."
            )
        return value

    async def _log_disposition(self, phone_number: str, disposition: str, **kwargs) -> dict:
        """Log disposition to CRM."""
        resp = await self._client.post(
            f"{self.crm_webhook}/disposition",
            json={
                "phone_number": phone_number,
                "disposition": disposition,
                **kwargs,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _honor_dnc(self, phone_number: str, **kwargs) -> dict:
        """Add number to DNC list."""
        resp = await self._client.post(
            f"{self.dnc_webhook}/add",
            json={"phone_number": phone_number, **kwargs},
        )
        resp.raise_for_status()
        return resp.json()

    async def _mark_callback(self, phone_number: str, callback_time: str, **kwargs) -> dict:
        """Schedule a callback."""
        resp = await self._client.post(
            f"{self.crm_webhook}/callback",
            json={
                "phone_number": phone_number,
                "callback_time": callback_time,
                **kwargs,
            },
        )
        resp.raise_for_status()
        return resp.json()
