"""Compliance gate — runs before every dial. Fail closed."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    allowed: bool
    reason: str
    details: dict = None


class ComplianceGate:
    """Pre-dial compliance check. Every call must pass through here."""

    def __init__(
        self,
        internal_dnc_file: str = "data/internal_dnc.txt",
        calling_window_start: int = 8,
        calling_window_end: int = 21,
        ai_disclosure: str = "This is an AI-powered call from {company_name}.",
        recording_disclosure: str = "This call may be recorded for quality and compliance purposes.",
        bypass_calling_hours: bool = False,
    ):
        self.internal_dnc_file = Path(internal_dnc_file)
        self.calling_window_start = calling_window_start
        self.calling_window_end = calling_window_end
        self.ai_disclosure = ai_disclosure
        self.recording_disclosure = recording_disclosure
        # Demo affordance only: suppresses the hours check so a run is
        # possible outside 8am-9pm. DNC is NEVER bypassed.
        self.bypass_calling_hours = bypass_calling_hours
        self._dnc_numbers: set[str] = set()
        self._load_dnc_list()

    def _load_dnc_list(self):
        """Load internal DNC list."""
        if self.internal_dnc_file.exists():
            with open(self.internal_dnc_file) as f:
                self._dnc_numbers = {
                    line.strip()
                    for line in f
                    # '#' lines are section headers in the list file, not numbers
                    if line.strip() and not line.lstrip().startswith("#")
                }
            logger.info(f"Loaded {len(self._dnc_numbers)} DNC numbers")

    def add_to_dnc(self, phone_number: str):
        """Add a number to the internal DNC list."""
        normalized = self._normalize_number(phone_number)
        self._dnc_numbers.add(normalized)
        with open(self.internal_dnc_file, "a") as f:
            f.write(f"{normalized}\n")
        logger.info(f"Added {normalized} to DNC list")

    def check_call_allowed(
        self,
        phone_number: str,
        recipient_timezone: Optional[str] = None,
    ) -> ComplianceResult:
        """Check if a call is allowed. Fail closed on any uncertainty."""
        normalized = self._normalize_number(phone_number)

        # Check internal DNC
        if normalized in self._dnc_numbers:
            return ComplianceResult(
                allowed=False,
                reason="Number is on internal DNC list",
                details={"number": normalized},
            )

        # Check calling hours
        if recipient_timezone and not self.bypass_calling_hours:
            tz = pytz.timezone(recipient_timezone)
            now = datetime.now(tz)
            hour = now.hour
            if hour < self.calling_window_start or hour >= self.calling_window_end:
                return ComplianceResult(
                    allowed=False,
                    reason=f"Outside calling hours ({self.calling_window_start}:00-{self.calling_window_end}:00 {recipient_timezone})",
                    details={"current_hour": hour, "timezone": recipient_timezone},
                )

        return ComplianceResult(allowed=True, reason="Call permitted")

    def get_disclosure_text(self, company_name: str) -> str:
        """Get the AI disclosure text."""
        return self.ai_disclosure.format(company_name=company_name)

    def get_recording_disclosure(self) -> str:
        """Get the recording disclosure text."""
        return self.recording_disclosure

    @staticmethod
    def _normalize_number(phone_number: str) -> str:
        """Normalize phone number to E.164 format."""
        digits = re.sub(r"\D", "", phone_number)
        if len(digits) == 10:
            return f"+1{digits}"
        if len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"
        return f"+{digits}"
