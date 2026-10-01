"""Number normalization and prospect identity.

Both exist for the same reason: a DNC entry that does not match the string
the gate looks up suppresses nobody, and a prospect answering as the wrong
company makes the agent log a bogus disposition.
"""

import asyncio
from dataclasses import replace

from compliance.gate import ComplianceGate
from sim.prospect import SCRIPTS, SimulatedProspect, script_for
from webhooks.handlers import normalize_number


def test_dnc_entry_normalizes_to_what_the_gate_looks_up():
    """A double-prefixed entry would never match, so the number is dropped."""
    gate = ComplianceGate(internal_dnc_file="data/internal_dnc.txt")
    gate._dnc_numbers.add(normalize_number("++14155550123"))
    result = gate.check_call_allowed("+14155550123")
    assert not result.allowed, "DNC entry did not match a plain E.164 lookup"


def test_various_formats_normalize_identically():
    target = "+14155550123"
    for raw in ["+14155550123", "+1-415-555-0123", "14155550123",
                "+1 415 555 0123", "(415) 555-0123"]:
        assert normalize_number(raw) == target, f"{raw!r} normalized wrong"


def test_normalize_matches_gate_normalize():
    for raw in ["+14155550123", "4155550123", "+1-415-555-0123"]:
        assert normalize_number(raw) == ComplianceGate._normalize_number(raw)


def test_prospect_uses_the_dialed_company():
    """Regression: the receptionist said Northwind while dialing Redwood."""
    script = replace(script_for("gatekeeper"), company="Redwood Health")
    prospect = SimulatedProspect(script, tts=None)
    assert prospect.lines[0].startswith("Redwood Health")
    assert "Northwind" not in prospect.lines[0]


def test_voicemail_uses_the_dialed_company():
    script = replace(script_for("voicemail"), company="Cobalt Systems")
    prospect = SimulatedProspect(script, tts=None)
    assert any("Cobalt Systems" in line for line in prospect.lines)


def test_no_script_line_leaves_an_unsubstituted_placeholder():
    for key in SCRIPTS:
        script = replace(script_for(key), company="Test Co")
        prospect = SimulatedProspect(script, tts=None)
        for line in prospect.lines:
            assert "{company}" not in line, f"{key} left a raw placeholder"

class _EchoLLM:
    """Returns a canned reply, so generation can be tested without a network."""

    def __init__(self, reply="Sure, that sounds fine to me."):
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_conversation_generates_its_own_reply():
    """The far end must generate, not read a fixed line.

    A scripted caller produces a scripted conversation: it answers a question
    that was never asked, because the line was next in the list.
    """
    import asyncio

    from sim.dialogue import Conversation, persona_for

    async def go():
        conv = Conversation(persona_for("interested"), llm=_EchoLLM())
        conv.note_agent_said("How much does it cost?")
        return await conv.generate()

    reply = asyncio.run(go())
    assert reply and reply.strip(), "the caller generated nothing"


def test_generated_reply_is_grounded_in_what_was_said():
    """The prompt must carry the agent's actual turn, or it is guessing."""
    import asyncio

    from sim.dialogue import Conversation, persona_for

    async def go():
        llm = _EchoLLM()
        conv = Conversation(persona_for("interested"), llm=llm)
        conv.note_agent_said("Does that come with onboarding and training?")
        await conv.generate()
        return llm.prompts[0]

    prompt = asyncio.run(go())
    assert "onboarding" in prompt.lower(), "the agent's turn was not passed"
    assert "Northwind" in prompt, "the persona's company is missing"


def test_conversation_stops_when_the_agent_closes():
    import asyncio

    from sim.dialogue import Conversation, persona_for

    async def go():
        conv = Conversation(persona_for("interested"), llm=_EchoLLM())
        conv.note_agent_said("Great, talk on Thursday then. Have a good one.")
        return conv.should_stop, await conv.generate()

    should_stop, reply = asyncio.run(go())
    assert should_stop is True, "the caller kept talking after a goodbye"
    assert reply is None


def test_conversation_is_bounded_by_a_turn_budget():
    """A free-running model must not talk forever."""
    import asyncio

    from sim.dialogue import Conversation, persona_for

    async def go():
        conv = Conversation(persona_for("interested"), llm=_EchoLLM(),
                           max_turns=2)
        out = []
        for _ in range(5):
            conv.note_agent_said("What next?")
            reply = await conv.generate()
            if reply is None:
                break
            conv.note_prospect_said(reply)
            out.append(reply)
        return out

    out = asyncio.run(go())
    assert len(out) <= 2, f"caller spoke {len(out)} times, budget was 2"


def test_prepared_reply_is_ready_before_the_line_frees():
    """Preparation must land during the agent's turn, not after it.

    This is the whole point: without it the far end cannot speak until it has
    generated *and* synthesized, which is seconds of dead air per turn.
    """
    import asyncio

    from sim.dialogue import Conversation, persona_for

    class _TTS:
        def __init__(self):
            self.calls = 0

        async def synthesize_pcm(self, text):
            self.calls += 1
            return b"\x00" * 320

    async def go():
        tts = _TTS()
        conv = Conversation(persona_for("interested"), llm=_EchoLLM())
        conv.note_agent_said("When works for you?")
        conv.prepare(tts)
        await asyncio.sleep(0.05)
        audio = await conv.take_prepared(tts)
        return audio, tts.calls

    audio, calls = asyncio.run(go())
    assert audio is not None, "no audio prepared during the agent's turn"
    assert calls == 1, "the reply was synthesized at speak time, not before"


class _StubTTS:
    """Returns silence so the prospect can be driven without a voice engine."""

    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize_pcm(self, text):
        self.spoken.append(text)
        return b"\x00" * 640


def test_prospect_stream_runs_end_to_end_with_an_llm():
    """Regression: the live call died on a missing attribute.

    `max_turns` was removed from `__init__` during the persona refactor but a
    reference survived in `stream()`, so the generator raised on its first turn.
    The far end then produced no audio, the recogniser got an empty stream, and
    the call ended after two turns having spoken once.

    No test caught it because every existing test drove the conversation object
    directly or built the prospect without the arguments production uses. This
    one runs the real stream, the way the pipeline does.
    """
    import asyncio

    from sim.prospect import SimulatedProspect, script_for
    from turns import TurnCoordinator

    class _LLM:
        async def complete(self, prompt: str) -> str:
            return "Yeah, that sounds fine. When works?"

    async def go():
        p = SimulatedProspect(
            script_for("interested"),
            _StubTTS(),
            turns=TurnCoordinator(),
            llm=_LLM(),
            phone_filter=False,
            pickup_delay=0.01,
        )
        await p.prime()
        chunks = []
        async for chunk in p.stream():
            chunks.append(chunk)
            if len(chunks) > 400:      # bounded: the stream is a live call
                break
        return chunks, p

    chunks, p = asyncio.run(go())
    assert chunks, "the prospect produced no audio at all"
    assert len(p._spoken) >= 1, "the prospect never spoke a line"


def test_prospect_speaks_again_after_the_agent_answers():
    """Regression: the call ended after the prospect's second turn.

    The prepared reply was read at the top of the loop, before the agent had
    said anything that could generate one. So the prospect waited for the
    agent's turn, the turn produced a reply in the background, and then nobody
    ever read it: the next pass found nothing prepared and the call ended.

    Preparation must therefore happen *during* the wait, so the audio is on hand
    the moment the line frees.
    """
    import asyncio

    from sim.prospect import SimulatedProspect, script_for
    from turns import TurnCoordinator

    class _TTS:
        def __init__(self):
            self.spoken: list[str] = []

        async def synthesize_pcm(self, text):
            self.spoken.append(text)
            return b"\x00" * 640

    class _LLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, prompt: str) -> str:
            self.calls += 1
            return f"Yes, reply number {self.calls} works."

    async def go():
        turns = TurnCoordinator()
        tts = _TTS()
        llm = _LLM()
        p = SimulatedProspect(
            script_for("interested"), tts, turns=turns, llm=llm,
            phone_filter=False, pickup_delay=0.01,
        )
        await p.prime()

        chunks = []

        async def agent_side():
            # Stand in for the pipeline: release the line, and tell the far end
            # what was said, a few times.
            await asyncio.sleep(0.25)
            for i in range(4):
                turns.agent_claims()
                await asyncio.sleep(0.05)
                p.on_agent_turn(f"Agent turn number {i}. What works for you?")
                await asyncio.sleep(0.4)
                turns.agent_releases()
                await asyncio.sleep(0.25)

        task = asyncio.create_task(agent_side())
        async for chunk in p.stream():
            chunks.append(chunk)
            if len(chunks) > 3000:
                break
        task.cancel()
        return len(p._spoken), llm.calls

    spoken, llm_calls = asyncio.run(go())
    assert spoken >= 3, f"prospect only spoke {spoken} times; the call stalls"
    assert llm_calls >= 2, "the far end never generated a reply"


def test_far_end_never_speaks_in_the_agents_voice():
    """The far end must synthesize with its own voice, always.

    The pipeline hands its far end a `prepare(tts)` hook and passed its OWN TTS
    into it. The far end forwarded that object to its synthesizer, so from the
    second utterance onward the prospect spoke in the agent's Lessac voice.
    Reported as "the agent started speaking as the prospect mid-call"; the
    transcript still read `prospect:`, so nothing looked wrong except the audio.

    The giveaway: the far end's own `tts` is never consulted on that path.
    Whichever object reaches `prepare` wins, and the only caller passed the
    wrong one.
    """
    from sim.prospect import SimulatedProspect, script_for
    from turns import TurnCoordinator

    class _VoiceTTS:
        def __init__(self, name):
            self.name = name
            self.spoken: list[tuple[str, str]] = []

        async def synthesize_pcm(self, text):
            self.spoken.append((self.name, text))
            # Valid PCM of the right shape; the voice identity is tracked in
            # `self.spoken`, not smuggled into the audio.
            return b"\x00\x01" * 320

    agent_tts = _VoiceTTS("AGENT_VOICE")
    prospect_tts = _VoiceTTS("PROSPECT_VOICE")

    async def go():
        turns = TurnCoordinator()
        p = SimulatedProspect(
            script_for("interested"), prospect_tts, turns=turns,
            llm=_GeneratingLLM(), phone_filter=False, pickup_delay=0.01,
        )
        await p.prime()

        # Exactly what the pipeline does now: start preparation, no voice.
        p.on_agent_turn("What works for you?")
        p.prepare()
        await p.warm_next_reply()
        await p.warm_next_reply()
        return prospect_tts.spoken, agent_tts.spoken, \
            p._line_cache.get("__prepared__")

    prospect_said, agent_said, cached = asyncio.run(go())

    assert "PROSPECT_VOICE" in str(prospect_said), (
        "the far end never used its own voice at all"
    )
    # The failure shows up in the AGENT's synthesizer being handed the far
    # end's line, not in the far end's: the wrong object won the race and
    # produced the audio nobody is checking here.
    assert not agent_said, (
        "the agent's voice synthesized the far end's reply: "
        f"{agent_said}"
    )


class _GeneratingLLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"Reply number {self.calls}."
