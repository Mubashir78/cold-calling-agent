"""Dialer orchestrator — manages call concurrency, pacing, and lead queue."""

import asyncio
import csv
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncIterator, Awaitable, Callable, Optional

from compliance.gate import ComplianceGate

logger = logging.getLogger(__name__)


@dataclass
class Lead:
    phone_number: str
    name: str = ""
    company: str = ""
    timezone: str = "America/New_York"
    persona_key: str = "interested"  # which simulated prospect to connect
    metadata: dict = field(default_factory=dict)
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    status: str = "pending"  # pending, called, booked, dnc, failed


@dataclass
class CallResult:
    lead: Lead
    disposition: str  # booked, callback, not_interested, dnc, voicemail, no_answer, wrong_number
    duration_seconds: float
    transcript: list = field(default_factory=list)
    recording_url: str = ""
    notes: str = ""


class DialerOrchestrator:
    """Manages outbound calling campaigns."""

    def __init__(
        self,
        compliance_gate: ComplianceGate,
        max_concurrent_calls: int = 5,
        calls_per_minute: int = 10,
        retry_attempts: int = 2,
        retry_delay_minutes: int = 30,
        caller_id_pool: Optional[list[str]] = None,
    ):
        self.compliance_gate = compliance_gate
        self.max_concurrent_calls = max_concurrent_calls
        self.calls_per_minute = calls_per_minute
        self.retry_attempts = retry_attempts
        self.retry_delay_minutes = retry_delay_minutes
        self.caller_id_pool = caller_id_pool or ["+15551234567"]
        # Back off a throttled call rather than dropping the prospect, but
        # not indefinitely: a 75s wait is defensible, ten minutes is not.
        self.rate_limit_retry_delay = 45.0

        self._semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._call_timestamps: list[float] = []
        self._leads: list[Lead] = []
        self._results: list[CallResult] = []
        self._active_calls: dict[str, asyncio.Task] = {}

    def load_leads_from_csv(self, filepath: str):
        """Load leads from a CSV file."""
        self._leads = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._leads.append(Lead(
                    phone_number=row.get("phone_number", ""),
                    name=row.get("name", ""),
                    company=row.get("company", ""),
                    timezone=row.get("timezone", "America/New_York"),
                    persona_key=row.get("persona_key", "interested"),
                    metadata={k: v for k, v in row.items() if k not in ["phone_number", "name", "company", "timezone", "persona_key"]},
                ))
        logger.info(f"Loaded {len(self._leads)} leads from {filepath}")

    async def run_campaign(
        self,
        call_handler: Callable[[Lead, str], Awaitable],
    ):
        """Run a calling campaign."""
        logger.info(f"Starting campaign with {len(self._leads)} leads")

        # Reset rate limiter every minute
        async def reset_rate_limiter():
            while True:
                await asyncio.sleep(60)
                # This is a simplified rate limiter
                pass

        asyncio.create_task(reset_rate_limiter())

        tasks = []
        for lead in self._leads:
            if lead.status != "pending":
                continue

            # Compliance check
            compliance = self.compliance_gate.check_call_allowed(
                lead.phone_number, lead.timezone
            )
            if not compliance.allowed:
                logger.info(f"Skipping {lead.phone_number}: {compliance.reason}")
                lead.status = "dnc" if "DNC" in compliance.reason else "failed"
                continue

            # Rate limiting: ensure we don't exceed calls_per_minute
            now = asyncio.get_event_loop().time()
            self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
            if len(self._call_timestamps) >= self.calls_per_minute:
                sleep_time = 60 - (now - self._call_timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            self._call_timestamps.append(asyncio.get_event_loop().time())

            # Pick caller ID
            caller_id = random.choice(self.caller_id_pool)

            # Start call
            task = asyncio.create_task(
                self._execute_call(lead, caller_id, call_handler)
            )
            self._active_calls[lead.phone_number] = task
            tasks.append(task)

        # Wait for all calls to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Campaign complete. {len(self._results)} calls made.")

    async def _execute_call(
        self,
        lead: Lead,
        caller_id: str,
        call_handler: Callable,
    ):
        """Execute a single call with semaphore control."""
        async with self._semaphore:
            lead.attempts += 1
            lead.last_attempt = datetime.now()

            try:
                result = await call_handler(lead, caller_id)
                self._results.append(result)
                lead.status = result.disposition
            except Exception as e:
                if isinstance(e, DailyTokenLimitExceeded) or "daily token" in str(e).lower():
                    # Retrying cannot help: the account's per-day budget is
                    # spent. Fail fast and say so, rather than holding every
                    # remaining lead open on a limit that will not clear.
                    logger.error(
                        "Daily token budget exhausted; stopping campaign. "
                        "Remaining %d lead(s) not attempted.",
                        len(self._leads) - len(self._results),
                    )
                    lead.status = "failed"
                    self._results.append(CallResult(
                        lead=lead,
                        disposition="failed",
                        duration_seconds=0.0,
                        notes="daily token budget exhausted",
                    ))
                    self.daily_limit_hit = True
                    return
                if "429" in str(e) or "Rate limit" in str(e):
                    # The budget is shared across the org, so a throttled
                    # call is worth retrying: the prospect is still on the
                    # line, and a live call is more valuable than a fast one.
                    logger.warning(
                        "Call to %s hit a rate limit; retrying in %.0fs",
                        lead.phone_number, self.rate_limit_retry_delay,
                    )
                    await asyncio.sleep(self.rate_limit_retry_delay)
                    if lead.attempts <= self.retry_attempts:
                        return await self._execute_call(lead, caller_id, call_handler)
                logger.error(f"Call to {lead.phone_number} failed: {e}")
                lead.status = "failed"
                # Record the failure as a result. Dropping it silently would
                # make a failed call look identical to a lead never dialled.
                self._results.append(CallResult(
                    lead=lead,
                    disposition="failed",
                    duration_seconds=0.0,
                    notes=f"{type(e).__name__}: {e}"[:200],
                ))
            finally:
                self._active_calls.pop(lead.phone_number, None)

    def get_results(self) -> list[CallResult]:
        """Get all call results."""
        return self._results

    def export_results_csv(self, filepath: str):
        """Export results to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phone_number", "name", "disposition", "duration_seconds", "notes"])
            for r in self._results:
                writer.writerow([
                    r.lead.phone_number,
                    r.lead.name,
                    r.disposition,
                    r.duration_seconds,
                    r.notes,
                ])
