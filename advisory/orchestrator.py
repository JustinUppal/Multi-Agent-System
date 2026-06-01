"""The advisor<->client conversation loop.

The advisor and client get separate runners and sessions and this module passes
each one's reply to the other until the advisor ends the engagement. They're kept
separate because they're two independent personas, not a coordinator and a
helper (see the README for why that rules out ADK's agent transfer here). The
analyst isn't in this loop at all; the advisor reaches it as a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from . import config
from .agents import (
    RESOLVED_STATE_KEY,
    build_advisor_agent,
    build_analyst_agent,
    build_client_agent,
)
from .profiles import ClientProfile
from .tracing import start_trace

# Fixed opening line so the client agent has something to respond to on turn 0
# (saves a model call and keeps the start of the transcript stable).
ADVISOR_GREETING = (
    "Hello, I'm your financial advisor. Thanks for coming in today. "
    "What's on your mind, and how can I help with your finances?"
)


@dataclass
class Turn:
    """One line of the visible transcript, plus any tool activity behind it."""

    speaker: str  # "advisor" or "client"
    text: str
    # Human-readable traces of tools the speaker used this turn (analyst calls,
    # web searches). Surfaced so the inter-agent collaboration is observable.
    tool_trace: list[str] = field(default_factory=list)


@dataclass
class SessionResult:
    """Outcome of a full advisory session."""

    transcript: list[Turn]
    resolved: bool  # True if the advisor called end_conversation (vs. hitting the cap)
    turns_used: int


async def _run_turn(
    runner: Runner, session_id: str, message: str
) -> tuple[str, list[str]]:
    """Send one message to an agent and collect its final text + tool traces.

    One turn spans many ADK events (model output, function-call requests, tool
    responses, the final reply). We grab the final text here; the tool calls,
    including nested ones via AgentTool, are collected by the ContextVar tracer
    that ``start_trace`` resets for this turn.
    """
    tool_trace = start_trace()
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = ""

    async for event in runner.run_async(
        user_id=config.USER_ID, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(
                part.text for part in event.content.parts if part.text
            ).strip()

    return final_text, tool_trace


async def run_advisory_session(profile: ClientProfile) -> SessionResult:
    """Run a full advisor<->client conversation and return the transcript.

    The advisor and client each get an isolated runner/session. We alternate:
    advisor speaks, client responds, advisor responds (possibly consulting the
    analyst), and so on. The loop stops when the advisor sets the resolved flag in
    its session state, or when MAX_TURNS is reached as a safety net.
    """
    config.require_api_key()

    # Build the agent graph. The analyst is nested inside the advisor via AgentTool.
    analyst = build_analyst_agent()
    advisor = build_advisor_agent(analyst)
    client = build_client_agent(profile)

    # Independent session services keep the two personas' memories fully separate.
    session_service = InMemorySessionService()
    advisor_runner = Runner(
        agent=advisor, app_name=config.APP_NAME, session_service=session_service
    )
    client_runner = Runner(
        agent=client, app_name=config.APP_NAME, session_service=session_service
    )

    advisor_session = await session_service.create_session(
        app_name=config.APP_NAME, user_id=config.USER_ID
    )
    client_session = await session_service.create_session(
        app_name=config.APP_NAME, user_id=config.USER_ID
    )

    transcript: list[Turn] = [Turn(speaker="advisor", text=ADVISOR_GREETING)]
    advisor_message = ADVISOR_GREETING
    resolved = False
    turns_used = 0

    for _ in range(config.MAX_TURNS):
        turns_used += 1

        # Client responds to the advisor's latest message.
        client_reply, client_tools = await _run_turn(
            client_runner, client_session.id, advisor_message
        )
        transcript.append(
            Turn(speaker="client", text=client_reply, tool_trace=client_tools)
        )

        # Advisor responds, possibly consulting the analyst (and the web) on the way.
        advisor_message, advisor_tools = await _run_turn(
            advisor_runner, advisor_session.id, client_reply
        )
        transcript.append(
            Turn(speaker="advisor", text=advisor_message, tool_trace=advisor_tools)
        )

        # The end_conversation tool writes this flag; re-read fresh session state.
        state = (
            await session_service.get_session(
                app_name=config.APP_NAME,
                user_id=config.USER_ID,
                session_id=advisor_session.id,
            )
        ).state
        if state.get(RESOLVED_STATE_KEY):
            resolved = True
            break

    return SessionResult(
        transcript=transcript, resolved=resolved, turns_used=turns_used
    )


def format_transcript(result: SessionResult) -> str:
    """Pretty-print a session result as a labelled, readable transcript."""
    lines: list[str] = []
    labels = {"advisor": "ADVISOR", "client": "CLIENT "}
    for turn in result.transcript:
        for trace in turn.tool_trace:
            lines.append(f"      ↳ [{trace}]")
        lines.append(f"{labels.get(turn.speaker, turn.speaker.upper())}: {turn.text}")
        lines.append("")

    ending = (
        "Engagement resolved by the advisor."
        if result.resolved
        else f"Stopped at the {config.MAX_TURNS}-turn safety cap without a resolution."
    )
    lines.append(f"--- {ending} ({result.turns_used} round-trips) ---")
    return "\n".join(lines)
