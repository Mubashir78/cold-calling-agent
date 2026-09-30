"""Local CRM / calendar / DNC stub.

Stands in for the systems the agent would really call. Exists so the tool
layer is exercised end to end rather than silently failing to connect.
"""

import hashlib
import hmac
import json
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request

from compliance.gate import ComplianceGate

logger = logging.getLogger(__name__)


def normalize_number(phone_number: str) -> str:
    """Normalize to E.164.

    The DNC list is only useful if entries match what the gate looks up, and
    the gate compares E.164. Storing whatever the caller sent guarantees
    silent mismatches.
    """
    return ComplianceGate._normalize_number(phone_number)

app = FastAPI(title="Cold Calling Agent — Local Integrations")

# In-memory state; a real deployment would use Postgres + a calendar API.
MEETINGS: list[dict] = []
DISPOSITIONS: list[dict] = []
CALLBACKS: list[dict] = []
DNC: set[str] = set()


def _reset() -> None:
    MEETINGS.clear()
    DISPOSITIONS.clear()
    CALLBACKS.clear()
    DNC.clear()


def verify_signature(payload: bytes, signature: str, secret: str = "") -> bool:
    if not secret:
        return True
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhooks/call-ended")
async def call_ended(request: Request, x_signature: Optional[str] = Header(None)):
    """Telephony provider: a call finished."""
    payload = await request.body()
    if not verify_signature(payload, x_signature or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")
    data = json.loads(payload)
    logger.info("call.ended %s", data.get("call_id"))
    return {"status": "ok"}


@app.post("/api/calendar/book")
async def calendar_book(request: Request):
    data = await request.json()
    event_id = f"evt_{len(MEETINGS) + 1:04d}"
    MEETINGS.append({**data, "event_id": event_id})
    logger.info("BOOKED %s for %s at %s", event_id,
                data.get("prospect_name"), data.get("time_slot"))
    return {"status": "ok", "event_id": event_id, "meeting": data}


@app.post("/api/crm/disposition")
async def crm_disposition(request: Request):
    data = await request.json()
    data["phone_number"] = normalize_number(data.get("phone_number", ""))
    DISPOSITIONS.append(data)
    logger.info("DISPOSITION %s -> %s", data.get("phone_number"),
                data.get("disposition"))
    return {"status": "ok"}


@app.post("/api/crm/callback")
async def crm_callback(request: Request):
    data = await request.json()
    CALLBACKS.append(data)
    logger.info("CALLBACK %s at %s", data.get("phone_number"),
                data.get("callback_time"))
    return {"status": "ok"}


@app.post("/api/dnc/add")
async def dnc_add(request: Request):
    data = await request.json()
    number = normalize_number(data.get("phone_number", ""))
    if number:
        DNC.add(number)
    logger.info("DNC %s", number)
    return {"status": "ok", "phone_number": number}


@app.get("/api/state")
async def state():
    """Everything booked so far, for verifying a run."""
    return {
        "meetings": MEETINGS,
        "dispositions": DISPOSITIONS,
        "callbacks": CALLBACKS,
        "dnc": sorted(DNC),
    }


@app.post("/api/state/reset")
async def reset():
    _reset()
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
