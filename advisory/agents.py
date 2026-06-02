"""Builds the three agents and wires up their tools.

    client            advisor ───────────────► analyst
    (persona)   ◄────► (orchestrator)          (researcher)
                        tools:                  tools:
                          - analyst (AgentTool)   - lookup_investments (knowledge store)
                          - end_conversation      - web_search (AgentTool -> google_search)

Only the advisor holds a reference to the analyst, so it's the only agent that
can talk to both the client and the analyst.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext

from . import config
from .knowledge import lookup_investments
from .profiles import ClientProfile
from .tracing import record_tool_call

# Session-state key the advisor sets (via end_conversation) to signal the
# orchestration loop that the engagement is resolved. Centralised so the writer
# (the tool) and the reader (the loop) cannot drift apart.
RESOLVED_STATE_KEY = "engagement_resolved"


def end_conversation(summary: str, tool_context: ToolContext) -> dict:
    """End the advisory session once the client's needs are fully resolved.

    Call this only after you have given concrete, researched recommendations and
    the client has signalled they are satisfied. Provide a short closing summary
    of the advice for the record.

    Args:
        summary: A concise recap of the recommendation the client agreed to.

    Returns:
        An acknowledgement dict; the orchestrator reads the state flag this sets.
    """
    # The loop reads session state, not the return value (the return just goes
    # back to the model).
    tool_context.state[RESOLVED_STATE_KEY] = True
    return {"status": "conversation_ended", "summary": summary}


def build_client_agent(profile: ClientProfile) -> LlmAgent:
    """The client simulator: role-plays the given profile, has no tools.

    Persona comes in as data (``profile.to_prompt()``); behaviour is in the
    instruction. The client is told to accept good advice and say so, otherwise
    two LLMs will keep chatting forever.
    """
    return LlmAgent(
        name="client",
        model=config.make_model(),
        description="A prospective client seeking investment advice.",
        instruction=(
            "You are role-playing a real person talking to their financial advisor. "
            "Stay fully in character and never reveal that you are an AI.\n\n"
            "=== YOUR PROFILE ===\n"
            f"{profile.to_prompt()}\n"
            "====================\n\n"
            "Behaviour:\n"
            "- Open by raising your situation and what you want help with.\n"
            "- Answer the advisor's questions naturally from your profile. If asked "
            "something your profile does not cover, improvise something plausible and "
            "consistent.\n"
            "- Show normal client behaviour: ask a clarifying question or voice a "
            "concern (fees, risk, taxes) when it is natural.\n"
            "- You are NOT an expert; do not propose specific allocations yourself.\n"
            "- When the advisor has given clear, specific recommendations that address "
            "your goals and you feel satisfied, say so plainly (e.g. 'That makes sense, "
            "thank you, I'm comfortable with that plan.') and stop asking new questions.\n"
            "Keep replies conversational and a few sentences long."
        ),
    )


def build_analyst_agent() -> LlmAgent:
    """The research agent: a knowledge-store tool plus an isolated web-search tool.

    The web search is wrapped in its own sub-agent because ADK's built-in
    ``google_search`` can't share an agent with ordinary function tools. Isolating
    it behind ``AgentTool`` lets the analyst use it alongside ``lookup_investments``.
    """
    search_agent = LlmAgent(
        name="web_search",
        model=config.make_model(),
        description="Searches the public web and returns current, sourced facts.",
        instruction=(
            "You are a web research specialist. Use google_search to find current, "
            "factual information for the query you are given (market levels, rates, "
            "fund details, recent events). Answer concisely and cite what you found. "
            "If results are thin, say so rather than guessing."
        ),
        tools=[google_search],
    )

    return LlmAgent(
        name="analyst",
        model=config.make_model(),
        description=(
            "A financial analyst that researches markets and instruments using an "
            "in-house knowledge store and live web search."
        ),
        instruction=(
            "You are a financial analyst. The advisor delegates research tasks to "
            "you; you do NOT talk to the client.\n\n"
            "For each task:\n"
            "1. Start with `lookup_investments` for asset-class principles, expected "
            "risk/return, and model-portfolio allocations; it is the authoritative "
            "house view.\n"
            "2. Use `web_search` for anything time-sensitive or specific that the "
            "knowledge store does not cover (current rates, market levels, specific "
            "funds/tickers, recent news).\n"
            "3. Synthesise findings into a tight, decision-ready briefing for the "
            "advisor: lead with the answer, support it with figures, and flag any "
            "important caveats or uncertainty. Note when a figure came from the web "
            "versus the house view.\n"
            "Be objective and never fabricate numbers."
        ),
        tools=[lookup_investments, AgentTool(agent=search_agent)],
        before_tool_callback=record_tool_call,
    )


def build_advisor_agent(analyst: LlmAgent) -> LlmAgent:
    """The orchestrator: the only agent that talks to both client and analyst.

    The analyst is given to the advisor as a tool, so defining a task for it is
    just a tool call with a research brief. ``end_conversation`` lets the advisor
    end the session when it's resolved.
    """
    return LlmAgent(
        name="advisor",
        model=config.make_model(),
        description="A financial advisor coordinating research to advise a client.",
        instruction=(
            "You are a financial advisor. You speak directly with the client, and "
            "you are the ONLY one who may consult the analyst. The client never sees "
            "the analyst.\n\n"
            "Method:\n"
            "- Understand the client's situation, goals, risk tolerance, and horizon. "
            "Ask focused follow-up questions if something material is missing.\n"
            "- Before giving substantive recommendations, delegate the research to the "
            "`analyst` tool. Hand it a SPECIFIC brief (the client's relevant facts plus "
            "exactly what to find out), e.g. a suitable allocation for a moderate-risk "
            "38-year-old, how to unwind concentrated employer stock, current cash yields. "
            "Do not invent market data yourself; get it from the analyst.\n"
            "- Translate the analyst's findings into clear, personalised advice in plain "
            "language. Tie every recommendation back to the client's goals. Never expose "
            "the analyst's raw output or mention internal tooling.\n"
            "- When the client's questions are answered and they are satisfied, call "
            "`end_conversation` with a brief summary of the agreed plan. Do not keep the "
            "conversation going once it is resolved."
        ),
        tools=[AgentTool(agent=analyst), end_conversation],
        before_tool_callback=record_tool_call,
    )
