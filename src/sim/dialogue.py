"""The far end of the call, as a real participant.

This started as seven canned lines played in order, which read as a scripted
exchange: the agent asked about pricing and heard an unrelated line, because
that line was next in the list. Rule-matching improved that but was still a
lookup table — the prospect could only say what someone anticipated, so a
question it did not recognise got a shrug.

So the prospect now generates its own speech through a model, conditioned on the
persona, the conversation so far, and what the agent just said. Both sides are
then language models talking to each other, which is the honest shape of the
problem.

Two constraints shape the design.

Latency. The prospect cannot speak until it has generated and synthesized its
reply, and that would otherwise land as seconds of dead air after the agent
finishes. So replies are generated and synthesized *during* the agent's turn,
speculatively, and the prospect simply plays what is already prepared. When the
agent releases the line, the audio is already there.

Direction. A free-running model drifts: it will keep talking past the end of a
call, invent facts about its company, or agree to things the agent never
offered. Each persona therefore carries a goal and a hard turn budget, and the
reply prompt states where the conversation should be going. The prospect is
being steered, not left to improvise.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Persona:
    """Who this caller is and how they behave on a call."""

    key: str
    name: str
    company: str
    description: str
    # What this caller wants out of the call. The model is steered toward it so
    # the conversation has a direction instead of wandering.
    goal: str
    style: str
    # Lines that mean the call is over. When the agent says one, the persona
    # closes rather than inventing another topic.
    closers: tuple[str, ...] = ()
    max_turns: int = 8
    # True when this caller would ask to be removed. Kept as data so the agent's
    # honour_dnc path stays exercised.
    asks_to_be_removed: bool = False


PERSONAS: dict[str, Persona] = {
    "interested": Persona(
        key="interested",
        name="Interested",
        company="Northwind Logistics",
        description=(
            "You are a busy operations manager at a logistics company. You are "
            "polite but time-poor, and you say so."
        ),
        goal=(
            "You want to understand whether this helps with missed calls, and "
            "if it does, you want a short call booked. You are ready to agree "
            "to a time once you are satisfied the call is worth it."
        ),
        style=(
            "One or two short sentences. Sound like a person on a phone, not a "
            "script. Answer the question you were actually asked. Do not repeat "
            "yourself. Do not volunteer information the agent did not ask for."
        ),
        closers=("talk then", "have a good one", "goodbye", "talk on"),
        max_turns=8,
    ),
    "gatekeeper": Persona(
        key="gatekeeper",
        company="Redwood Health",
        name="Gatekeeper",
        description=(
            "You are a receptionist at a healthcare provider screening inbound "
            "calls. You are brisk and protective of staff time."
        ),
        goal=(
            "You want to end the call quickly and pass nothing to a clinician. "
            "You will not book a meeting yourself."
        ),
        style=(
            "Very short. Clipped. You are busy and slightly impatient."
        ),
        closers=("goodbye", "have a good", "no callback"),
        max_turns=4,
    ),
    "dnc": Persona(
        key="dnc",
        company="Cobalt Systems",
        name="Do Not Call",
        description=(
            "You are irritated. This company has called you before after you "
            "asked to be removed."
        ),
        goal=(
            "You want the calls to stop. Say clearly that you asked to be "
            "removed and demand they take you off the list."
        ),
        style=(
            "Short and blunt. You are not interested in what they are selling."
        ),
        closers=("goodbye", "fine.", "don't call"),
        max_turns=4,
        asks_to_be_removed=True,
    ),
}


def persona_for(key: str, company: Optional[str] = None) -> Persona:
    p = PERSONAS.get(key, PERSONAS["interested"])
    if company and company != p.company:
        # The lead supplies the identity: a receptionist answers as the
        # company that was actually dialed.
        p = Persona(**{**p.__dict__, "company": company})
    return p


# Told to the model on every turn. Constrains the *shape* of the reply without
# scripting the words.
REPLY_STYLE = """\
You are the person on the other end of a live phone call.

Rules:
1. Reply in ONE or TWO short sentences. This is a phone call, not email.
2. Answer ONLY what was just said to you. Do not change the subject.
3. Sound like a real person. Contractions, filler, natural phrasing.
4. Never repeat a question you have already answered.
5. WORD IT DIFFERENTLY EACH TIME. A model given the same prompt tends to
   return the same sentence, and a caller who says the identical thing twice
   in a row stops sounding like a person. Choose fresh wording every turn.
6. Never invent facts about your company beyond what you have been told.
7. Never agree to a specific time, price or commitment on the agent's behalf \
unless you have been asked about it and you want to say yes.
8. If the agent has said goodbye or wrapped up, close the call briefly.
9. If the agent has not asked you anything new, react to what they said \
rather than repeating yourself: "Right", "Okay, go on", "I see."
10. Output ONLY the words you would say. No stage directions, no narration, \
no quotes around your reply."""


class Conversation:
    """One side of the call: remembers what was said, decides what to say next."""

    def __init__(self, persona: Persona, llm=None, max_turns: Optional[int] = None):
        self.persona = persona
        self.llm = llm
        self.max_turns = max_turns if max_turns is not None else persona.max_turns
        self.history: list[dict] = []
        self.turns = 0
        self.closed = False
        # The reply prepared for the current agent turn, synthesized ahead of
        # time so the prospect can speak the instant the line frees.
        self._prepared_text: Optional[str] = None
        self._prepared_audio: Optional[bytes] = None
        self._preparing: Optional[asyncio.Task] = None

    # -- conversation state -------------------------------------------------

    def note_agent_said(self, text: str) -> None:
        if text:
            self.history.append({"role": "caller_says", "text": text})
            if self._is_closing(text):
                self.closed = True

    def _is_closing(self, agent_text: str) -> bool:
        low = agent_text.lower()
        return any(c in low for c in self.persona.closers)

    def note_prospect_said(self, text: str) -> None:
        self.history.append({"role": "you_said", "text": text})
        self.turns += 1
        self._prepared_text = None
        self._prepared_audio = None

    @property
    def should_stop(self) -> bool:
        stop = self.closed or self.turns >= self.max_turns
        if stop:
            logger.info("Far end stopping: closed=%s turns=%d/%d",
                        self.closed, self.turns, self.max_turns)
        return stop

    # -- reply generation ---------------------------------------------------

    def _prompt(self) -> str:
        lines = [
            REPLY_STYLE,
            "",
            f"You are {self.persona.description}",
            f"You are calling about: {self.persona.goal}",
            f"How you talk: {self.persona.style}",
            f"Your company is {self.persona.company}.",
            "",
        ]
        if self.history:
            lines.append("So far:")
            for turn in self.history[-8:]:
                who = "Caller" if turn["role"] == "caller_says" else "You"
                lines.append(f"{who}: {turn['text']}")
        return "\n".join(lines)

    async def generate(self) -> Optional[str]:
        """Produce the next line, or None if there is nothing left to say.

        Logged at info, not debug: a call whose far end goes quiet is otherwise
        indistinguishable from a normal end of call, and several failed runs
        looked clean because nothing here was visible.
        """
        if self.should_stop:
            return None
        if self.llm is None:
            return None
        try:
            text = await self.llm.complete(self._prompt())
        except Exception as exc:
            logger.warning("prospect generation failed: %s", exc)
            return None
        text = _clean(text)
        if not text:
            logger.info("Far end generated nothing for %r",
                        (self.history[-1]["text"][:40] if self.history else ""))
            return None
        logger.info("Far end says: %s", text)
        self._prepared_text = text
        return text

    # -- speculative preparation -------------------------------------------

    def prepare(self, tts) -> None:
        """Start generating and synthesizing the next reply in the background.

        Called while the agent is still speaking. Without this the prospect
        cannot answer until it has generated *and* synthesized, which is seconds
        of dead air after every agent turn — the single largest source of the
        gap in this call.
        """
        if self.should_stop or self._preparing is not None:
            return
        if not self.history:
            return   # nothing to answer yet

        async def run():
            try:
                text = await self.generate()
                if text and tts is not None:
                    audio = await tts.synthesize_pcm(text)
                    self._prepared_audio = audio
            except Exception as exc:
                # At warning, not debug. This runs in a detached task whose
                # failure is otherwise completely invisible, which is how a
                # far end that never once replied looked like a clean
                # end-of-call for several runs running.
                logger.warning("Far end could not prepare a reply: %s", exc)

        self._preparing = asyncio.create_task(run())

    async def take_prepared(self, tts, condition=None) -> Optional[tuple]:
        """The audio for this turn, and its text, waiting briefly for prep.

        Returns a (pcm, text) pair. The text matters as much as the audio: the
        transcript is built from it, and dropping it left the conversation
        unable to advance past its opening turn even though audio was playing.
        """
        if self._preparing is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._preparing), 3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.debug("reply preparation did not finish in time")
            except Exception as exc:
                logger.debug("reply preparation failed: %s", exc)
            self._preparing = None

        text = self._prepared_text
        audio = self._prepared_audio
        self._prepared_text = None
        self._prepared_audio = None
        if audio is not None and text:
            return (condition(audio) if condition else audio), text

        # Preparation did not produce usable audio. Fall back to generating
        # now, so the prospect still answers rather than going silent.
        if self.should_stop:
            return None
        text = await self.generate()
        if not text:
            return None
        pcm = await tts.synthesize_pcm(text)
        return (condition(pcm) if condition else pcm), text


def _clean(text: str) -> str:
    """Strip anything that is not something a person would say out loud."""
    if not text:
        return ""
    out = text.strip()
    # Models sometimes wrap a reply in quotes or add a label.
    for prefix in ("You:", "Caller:", "Prospect:", "Reply:"):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    out = out.strip('"').strip("'").strip()
    # One turn, not a paragraph. First one or two sentences.
    parts = [p.strip() for p in out.replace("\n", " ").split(".") if p.strip()]
    if len(parts) > 2:
        out = ". ".join(parts[:2]) + "."
    elif parts and not out.endswith((".", "?", "!")):
        out = out + "."
    return out.strip()