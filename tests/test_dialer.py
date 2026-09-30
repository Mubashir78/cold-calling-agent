"""Tests for dialer orchestrator."""

import pytest
from dialer.orchestrator import DialerOrchestrator, Lead
from compliance.gate import ComplianceGate


@pytest.fixture
def orchestrator():
    gate = ComplianceGate(internal_dnc_file="data/internal_dnc.txt")
    return DialerOrchestrator(
        compliance_gate=gate,
        max_concurrent_calls=2,
        calls_per_minute=5,
    )


def test_load_leads(orchestrator):
    orchestrator.load_leads_from_csv("data/leads.csv")
    assert len(orchestrator._leads) > 0


def test_compliance_blocks_dnc(orchestrator):
    orchestrator.compliance_gate.add_to_dnc("+15550000000")
    lead = Lead(phone_number="+15550000000")
    orchestrator._leads = [lead]
    # The lead should be skipped
    assert lead.phone_number in orchestrator.compliance_gate._dnc_numbers
