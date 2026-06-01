"""Tool-call tracing so the nested agent calls show up in the transcript.

When the advisor calls the analyst via ``AgentTool`` the analyst runs in its own
nested invocation, so only its final answer comes back on the advisor's event
stream. Its own tool calls (knowledge store, web search) stay hidden. To surface
them we put a ``before_tool_callback`` on each agent that appends a line to a
collector held in a ``ContextVar``, which the nested async work inherits.

``google_search`` is server-side and doesn't fire tool callbacks, so what we
actually trace is advisor->analyst and analyst->web_search/knowledge-store.
"""

from __future__ import annotations

from contextvars import ContextVar

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

# The active turn's trace list. None means "not collecting" (calls are ignored).
_active_trace: ContextVar[list[str] | None] = ContextVar("_active_trace", default=None)


def start_trace() -> list[str]:
    """Begin collecting tool traces for the current turn; returns the list."""
    collector: list[str] = []
    _active_trace.set(collector)
    return collector


def record_tool_call(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> None:
    """ADK ``before_tool_callback``: append a readable trace line, then proceed.

    Returning ``None`` tells ADK to run the tool normally; we only observe.
    """
    collector = _active_trace.get()
    if collector is not None:
        collector.append(_describe(tool.name, args or {}))
    return None


def _describe(name: str, args: dict) -> str:
    """Render a tool call as a short, human-readable trace line."""
    if name == "analyst":
        return f"advisor → analyst: {_truncate(_brief(args))}"
    if name == "web_search":
        return f"analyst → web search: {_truncate(_brief(args))}"
    if name == "lookup_investments":
        return f"analyst → knowledge store: lookup '{args.get('topic', '')}'"
    if name == "end_conversation":
        return "advisor → end_conversation (engagement resolved)"
    return f"{name}({args})"


def _brief(args: dict) -> str:
    """AgentTool passes the sub-agent prompt under 'request'; fall back to all args."""
    return str(args.get("request", args)).strip()


def _truncate(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
