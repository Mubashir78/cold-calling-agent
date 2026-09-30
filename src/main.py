"""Main entry point for the cold calling agent."""

import asyncio
import logging
import os
import sys
import time
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

from stt.assemblyai import AssemblyAIRealtimeSTT
from stt.engine import StreamingSTT
from stt.groq_stt import GroqSTT
from llm.client import GroqLLM
from tts.engine import PiperTTS, TTSConfig
from compliance.gate import ComplianceGate
from dialer.orchestrator import CallResult, DialerOrchestrator, Lead
from tools.registry import ToolRegistry
from pipeline import VoicePipeline
from sim.prospect import SAMPLE_RATE, SimulatedProspect, script_for
from sim.recorder import CallRecording
from turnlock import TurnLock

load_dotenv()

Path("logs").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from YAML."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


SYSTEM_PROMPT = """You are an AI sales agent calling for {company_name}. Goal: qualify and book a meeting.

You are on a phone call. Sound like a person, not a script.

Hard rules:
1. Open by disclosing you are an AI: "This is an AI-powered call from {company_name}."
2. One or two short sentences per turn. Never more.
3. Answer the question you were asked FIRST, then ask at most ONE new question.
   Never stack two questions in one turn.
4. Never repeat a question you already asked. If they answered it, move on.
5. If asked about price, say it depends on call volume and their setup, and offer
   a short call. Never invent a price, discount or contract term.
6. If asked to be removed, call honor_dnc immediately and end the call.
7. Only call book_meeting for a time the prospect THEMSELVES named or explicitly
   accepted. Never book a slot you offered yourself.
8. When they agree to something, call the matching tool in that same turn.
9. If they sound annoyed or give a short answer, stop selling and wrap up politely.
10. Mention what you sell in plain terms: {value_prop}.
"""

COMPANY_NAME = "Acme Corp"
VALUE_PROP = "cutting missed inbound calls in half"

OPENING = (
    "Hi, this is an AI-powered call from {company_name}. "
    "Is this the right person to talk to about your phone system?"
)


def _opening_for(lead: Lead) -> str:
    """First thing said after pickup. Always discloses the AI, as required."""
    return OPENING.format(company_name=COMPANY_NAME)


def _prompt_for(lead: Lead, base_prompt: str) -> str:
    """Per-lead system prompt.

    The current date is injected because the model cannot infer it, and it
    will otherwise book meetings in the past.
    """
    today = datetime.now().strftime("%A %d %B %Y")
    horizon = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    return (
        f"{base_prompt}\n\n"
        f"You are calling {lead.company}"
        + (f", speaking with {lead.name}" if lead.name else "")
        + f". Reference the company by name once, naturally.\n"
        f"Today is {today}. Any meeting you book must be in the future, "
        f"between now and {horizon}. Use that date range when picking times.\n"
        f"Never repeat a booking you have already made."
    )


async def _serve_integrations(port: int = 8080) -> asyncio.Task:
    """Run the CRM/calendar/DNC stub in-process.

    The tool layer is useless without it, and a separately-launched server
    dies with its parent shell — which is exactly how a run ended up
    reporting bookings that never happened.
    """
    import uvicorn

    from webhooks.handlers import app as integrations_app

    config = uvicorn.Config(
        integrations_app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    # Stop waiting on the lifespan task when we shut the server down, so
    # cancelling the serve task does not surface as an unhandled error.
    server._captive_lifespan_task = None
    task = asyncio.create_task(server.serve())
    task._server = server  # type: ignore[attr-defined]
    # Wait for the socket to actually accept connections before dialing.
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.1)
    if not server.started:
        raise RuntimeError(f"integrations server failed to bind port {port}")
    logger.info("Integrations server listening on 127.0.0.1:%d", port)
    return task


async def _drain(audio_out: asyncio.Queue, recorder=None) -> None:
    """Stand in for the media bridge that would publish this to a LiveKit track.

    Consumes at real time. A real media track plays 16 kHz audio at 16 kHz, so
    draining instantly would stamp a whole reply inside a few milliseconds
    and compress the agent's track against the prospect's — inventing overlap
    that no listener would ever hear. Sleeping for each chunk's own duration
    is what makes the recording a faithful timeline.
    """
    total = 0
    try:
        while True:
            item = await audio_out.get()
            speaker, pcm = item if isinstance(item, tuple) else ("agent", item)
            total += len(pcm)
            duration = len(pcm) / (SAMPLE_RATE * 2)
            if recorder is not None:
                # Record the instant playback starts...
                recorder.add_playback(speaker, pcm)
            # ...but only count the chunk as finished once it has finished.
            # Counting on arrival released the line while the last sentence
            # was still audible, which is exactly the overlap we measured.
            await asyncio.sleep(duration)
            if recorder is not None:
                recorder.mark_playback_finished(speaker, len(pcm))
    except asyncio.CancelledError:
        logger.info("Agent said %d bytes of audio (%.1fs)",
                    total, total / (SAMPLE_RATE * 2))
        raise


async def main():
    """Main entry point."""
    logger.info("Starting Cold Calling Agent")

    config = load_config()

    # Initialize components
    stt_provider = config["voice"].get("stt_provider", "groq")
    if stt_provider == "assemblyai":
        stt = AssemblyAIRealtimeSTT(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            model=config["voice"].get("stt_speech_model", "universal-3-5-pro"),
            # Bias the recogniser toward the vocabulary this call actually
            # uses; proper nouns are what a context-free recogniser loses.
            keyterms=config["voice"].get("stt_keyterms", []),
        )
    elif stt_provider == "groq":
        stt = GroqSTT(
            model=config["voice"]["stt_model"],
            language=config["voice"]["stt_language"],
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        stt = StreamingSTT(
            model_size=config["voice"]["stt_model"],
            language=config["voice"]["stt_language"],
            beam_size=config["voice"]["stt_beam_size"],
            vad_filter=config["voice"]["stt_vad_filter"],
        )
    await stt.initialize()

    llm = GroqLLM(
        model=config["voice"]["llm_model"],
        temperature=config["voice"]["llm_temperature"],
        max_tokens=config["voice"]["llm_max_tokens"],
        api_key=os.getenv("GROQ_API_KEY"),
    )

    tts = PiperTTS(TTSConfig(
        voice=config["voice"]["tts_voice"],
        speaking_rate=config["voice"]["tts_speaking_rate"],
    ))
    await tts.initialize()

    # A second voice, so the prospect is audibly a different speaker.
    prospect_tts = PiperTTS(TTSConfig(
        voice=config["voice"]["prospect_tts_voice"],
        speaking_rate=1.05,
    ))
    await prospect_tts.initialize()

    compliance = ComplianceGate(
        internal_dnc_file=config["compliance"]["internal_dnc_file"],
        calling_window_start=config["compliance"]["calling_window_start"],
        calling_window_end=config["compliance"]["calling_window_end"],
        ai_disclosure=config["compliance"]["ai_disclosure"],
        recording_disclosure=config["compliance"]["recording_disclosure"],
        bypass_calling_hours=config["compliance"].get("bypass_calling_hours", False),
    )

    tools = ToolRegistry(
        crm_webhook=config["tools"]["crm_webhook"],
        calendar_webhook=config["tools"]["calendar_webhook"],
        dnc_webhook=config["tools"]["dnc_webhook"],
    )

    orchestrator = DialerOrchestrator(
        compliance_gate=compliance,
        max_concurrent_calls=config["dialer"]["max_concurrent_calls"],
        calls_per_minute=config["dialer"]["calls_per_minute"],
        retry_attempts=config["dialer"]["retry_attempts"],
        retry_delay_minutes=config["dialer"]["retry_delay_minutes"],
        caller_id_pool=config["telephony"]["caller_id_pool"],
    )

    integrations = await _serve_integrations(config["tools"].get("port", 8080))

    # Load leads
    orchestrator.load_leads_from_csv(config["dialer"]["lead_file"])

    # Create voice pipeline
    system_prompt = SYSTEM_PROMPT.format(
        company_name=COMPANY_NAME, value_prop=VALUE_PROP
    )

    # Define call handler
    async def handle_call(lead: Lead, caller_id: str):
        """Place one call and run the voice pipeline on it.

        One VoicePipeline per call — the pipeline holds per-call state, so
        sharing it across concurrent leads would cross conversations.
        """
        logger.info("Calling %s from %s", lead.phone_number, caller_id)

        # In production this block is replaced by a LiveKit SIP room; the
        # pipeline contract is identical: async iterator in, queue out.
        # One lock per call: both sides arbitrate through it, so they cannot
        # talk over each other.
        lock = TurnLock(
            silence_required=config["voice"].get("turn_silence_ms", 350) / 1000.0,
            timeout=config["voice"].get("turn_lock_timeout", 30.0),
        )

        script = script_for(lead.persona_key)
        # The prospect answers as the company actually dialed. The script
        # supplies the persona; the lead supplies the identity.
        script = replace(script, company=lead.company)
        prospect = SimulatedProspect(
            script, prospect_tts,
            seed=abs(hash(lead.phone_number)) % 10_000,
            lock=lock,
        )
        await prospect.prime()

        tools.start_call()
        tools.bind_call(lead.phone_number, lead.name, lead.company)

        recording = CallRecording(
            lead_name=lead.name or "prospect",
            company=lead.company,
            phone_number=lead.phone_number,
        ) if config["recording"].get("enabled", True) else None
        if recording is not None:
            # Shared zero point for both tracks; without it the stereo mix
            # drifts and overlap measurements are meaningless.
            recording.start()

        pipeline = VoicePipeline(
            stt=stt,
            llm=llm,
            tts=tts,
            tools=tools,
            system_prompt=_prompt_for(lead, system_prompt),
            max_call_duration=config["compliance"]["max_call_duration"],
            recorder=recording,
            lock=lock,
            turn_spacing=config["voice"].get("turn_spacing", 1.5),
            turn_gap=config["voice"].get("turn_gap", 0.45),
        )
        # Let the prospect hold the line open until this reply is finished,
        # rather than talking over it after a guessed interval.
        def _on_agent_speaking(speaking: bool) -> None:
            if speaking:
                prospect.agent_started_speaking()
            else:
                prospect.agent_finished_speaking()

        pipeline.on_agent_speaking = _on_agent_speaking

        audio_out: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(_drain(audio_out, recording))

        started = time.time()
        try:
            await pipeline.run_call(
                prospect.stream(),
                audio_out,
                opening=_opening_for(lead),
                sample_rate=SAMPLE_RATE,
            )
        finally:
            consumer.cancel()
            # Write the recording even when the call blew up. A truncated
            # conversation is still worth hearing; losing it entirely because
            # of a rate limit is not acceptable.
            if recording is not None:
                paths = recording.write(
                    config["recording"].get("out_dir", "data/calls")
                )
                logger.info("Call audio: %s", paths["stereo"])
                logger.info("Transcript: %s", paths["transcript"])

        return CallResult(
            lead=lead,
            disposition=pipeline.get_disposition(),
            duration_seconds=round(time.time() - started, 1),
            transcript=pipeline.get_transcript(),
        )

    # Run campaign
    try:
        await orchestrator.run_campaign(handle_call)
    finally:
        server = getattr(integrations, "_server", None)
        if server is not None:
            server.should_exit = True
        with suppress(asyncio.CancelledError):
            await integrations

        # Report what actually landed in the integrations, so a run can be
        # checked against ground truth instead of the agent's own claims.
        from webhooks.handlers import CALLBACKS, DISPOSITIONS, DNC, MEETINGS

        logger.info("=== Integrations state ===")
        logger.info("Meetings booked: %d", len(MEETINGS))
        for m in MEETINGS:
            logger.info("  %s %s @ %s", m.get("prospect_name"),
                        m.get("company", ""), m.get("time_slot"))
        logger.info("Dispositions logged: %d", len(DISPOSITIONS))
        logger.info("Callbacks scheduled: %d", len(CALLBACKS))
        logger.info("DNC entries: %d", len(DNC))

    # Export results
    orchestrator.export_results_csv("data/results.csv")
    logger.info("Results exported to data/results.csv")
    logger.info("LLM usage: %s", llm.usage_summary())


if __name__ == "__main__":
    asyncio.run(main())
