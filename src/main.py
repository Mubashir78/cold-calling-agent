"""Main entry point for the cold calling agent.

Wiring only. Every policy decision lives in the module that owns it:

- who has the line ............ `turns.TurnCoordinator`
- when a turn has finished .... `playback.Playback`
- what the agent may do ....... `tools.registry.ToolRegistry`
- who may be called ........... `compliance.gate.ComplianceGate`
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

from compliance.gate import ComplianceGate
from dialer.orchestrator import CallResult, DialerOrchestrator, Lead
from llm.client import GroqLLM
from playback import Playback
from pipeline import VoicePipeline
from sim.prospect import SAMPLE_RATE, SimulatedProspect, script_for
from sim.recorder import CallRecording
from stt.assemblyai import AssemblyAIRealtimeSTT
from stt.engine import StreamingSTT
from stt.groq_stt import GroqSTT
from tools.registry import ToolRegistry
from turns import TurnCoordinator
from tts.engine import PiperTTS, TTSConfig

load_dotenv()

Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/agent.log")],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(Path("config/config.yaml")) as f:
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

    The current date is injected because the model cannot infer it, and will
    otherwise book meetings in the past.
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
    dies with its parent shell — which is how a run once reported bookings
    that never happened.
    """
    import uvicorn

    from webhooks.handlers import app as integrations_app

    server = uvicorn.Server(uvicorn.Config(
        integrations_app, host="127.0.0.1", port=port, log_level="warning"
    ))
    # Don't wait on the lifespan task at shutdown, so cancelling the serve task
    # does not surface as an unhandled error.
    server._captured_lifespan_task = None
    task = asyncio.create_task(server.serve())
    task._server = server  # type: ignore[attr-defined]
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.1)
    if not server.started:
        raise RuntimeError(f"integrations server failed to bind port {port}")
    logger.info("Integrations server listening on 127.0.0.1:%d", port)
    return task


def _build_stt(config: dict):
    provider = config["voice"].get("stt_provider", "assemblyai")
    if provider == "assemblyai":
        return AssemblyAIRealtimeSTT(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            model=config["voice"].get("stt_speech_model", "universal-3-5-pro"),
            # Bias the recogniser toward the vocabulary this call uses; proper
            # nouns are what a context-free recogniser loses.
            keyterms=config["voice"].get("stt_keyterms", []),
        )
    if provider == "groq":
        return GroqSTT(
            model=config["voice"]["stt_model"],
            language=config["voice"]["stt_language"],
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return StreamingSTT(
        model_size=config["voice"]["stt_model"],
        language=config["voice"]["stt_language"],
        beam_size=config["voice"]["stt_beam_size"],
        vad_filter=config["voice"]["stt_vad_filter"],
    )


async def main():
    logger.info("Starting Cold Calling Agent")
    config = load_config()
    voice = config["voice"]

    stt = _build_stt(config)
    await stt.initialize()

    llm = GroqLLM(
        model=voice["llm_model"],
        temperature=voice["llm_temperature"],
        max_tokens=voice["llm_max_tokens"],
        api_key=os.getenv("GROQ_API_KEY"),
        # Groq budgets are per model, so one model exhausting its daily
        # allowance should degrade to a slower one, not end the call.
        fallback_models=voice.get("llm_fallback_models", []),
    )

    tts = PiperTTS(TTSConfig(
        voice=voice["tts_voice"],
        speaking_rate=voice["tts_speaking_rate"],
    ))
    await tts.initialize()

    # A second voice, so the prospect is audibly a different speaker.
    prospect_tts = PiperTTS(TTSConfig(
        voice=voice["prospect_tts_voice"],
        speaking_rate=1.05,
    ))
    await prospect_tts.initialize()

    # The far end speaks too, so both sides of the call are generated. A small
    # model is enough for a one-or-two-sentence reply and keeps this off the
    # agent's token budget, which is the budget that decides whether a meeting
    # gets booked.
    prospect_llm = None
    if voice.get("prospect_generates", True):
        prospect_llm = GroqLLM(
            model=voice.get("prospect_llm_model", llm.model),
            temperature=float(voice.get("prospect_llm_temperature", 0.8)),
            max_tokens=int(voice.get("prospect_llm_max_tokens", 64)),
            api_key=os.getenv("GROQ_API_KEY"),
            fallback_models=voice.get("llm_fallback_models", []),
        )

    compliance = ComplianceGate(
        internal_dnc_file=config["compliance"]["internal_dnc_file"],
        calling_window_start=config["compliance"]["calling_window_start"],
        calling_window_end=config["compliance"]["calling_window_end"],
        ai_disclosure=config["compliance"]["ai_disclosure"],
        recording_disclosure=config["compliance"]["recording_disclosure"],
        bypass_calling_hours=config["compliance"].get("bypass_calling_hours"),
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
    orchestrator.load_leads_from_csv(config["dialer"]["lead_file"])

    async def handle_call(lead: Lead, caller_id: str):
        """Place one call and run the voice pipeline on it.

        In production this block is replaced by a LiveKit SIP room; the pipeline
        contract is identical: async iterator in, queue out.
        """
        logger.info("Calling %s from %s", lead.phone_number, caller_id)

        # One coordinator per call. Both sides read it, so they cannot disagree
        # about who has the line.
        turns = TurnCoordinator()

        script = replace(script_for(lead.persona_key), company=lead.company)
        prospect = SimulatedProspect(
            script, prospect_tts,
            seed=abs(hash(lead.phone_number)) % 10_000,
            turns=turns,
            # The far end is a participant, not a recording: it generates its
            # own lines, so the agent is evaluated against a caller that
            # actually answers what it was asked.
            llm=prospect_llm,
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

        # The playback consumer owns turn completion. It is constructed before
        # the pipeline because the pipeline awaits its signal.
        playback = Playback(
            recorder=recording,
            coordinator=turns,
            turn_gap=voice.get("turn_gap", 0.3),
        )

        pipeline = VoicePipeline(
            stt=stt,
            llm=llm,
            tts=tts,
            tools=tools,
            system_prompt=_prompt_for(
                lead, SYSTEM_PROMPT.format(
                    company_name=COMPANY_NAME, value_prop=VALUE_PROP)),
            max_call_duration=config["compliance"]["max_call_duration"],
            recorder=recording,
            coordinator=turns,
            playback=playback,
            turn_spacing=float(voice.get("turn_spacing", 0.0)),
        )

        audio_out: asyncio.Queue = asyncio.Queue()
        consumer = asyncio.create_task(playback.run(audio_out))

        # Hand the pipeline its far end. Without this the agent's turns go
        # nowhere: the caller is never told what was said, so it never has a
        # reason to reply and the call ends after the agent's second turn.
        pipeline.on_far_end_turn = prospect

        started = time.time()
        try:
            await pipeline.run_call(
                # The far end is told each agent turn so it can prepare an
                # answer while that turn is still playing. That overlap is what
                # keeps the gap short: otherwise it has to generate and
                # synthesize from scratch the moment the line frees.
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
                paths = recording.write(config["recording"].get("out_dir"))
                logger.info("Call audio: %s", paths["stereo"])
                logger.info("Transcript: %s", paths["transcript"])

        return CallResult(
            lead=lead,
            disposition=pipeline.get_disposition(),
            duration_seconds=round(time.time() - started, 1),
            transcript=pipeline.get_transcript(),
        )

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

    orchestrator.export_results_csv("data/results.csv")
    logger.info("Results exported to data/results.csv")
    logger.info("LLM usage: %s", llm.usage_summary())


if __name__ == "__main__":
    asyncio.run(main())