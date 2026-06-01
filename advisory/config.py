"""Configuration, read once from the environment (a local .env in dev)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.models.google_llm import Gemini
from google.genai import types

load_dotenv()

# gemini-2.5-flash supports the google_search grounding the analyst needs. On a
# free-tier key the many calls per round-trip can hit rate limits, in which case
# ADVISORY_MODEL=gemini-2.0-flash has more headroom (make_model retries either way).
MODEL: str = os.getenv("ADVISORY_MODEL", "gemini-2.5-flash")

# Safety net only. The conversation should end when the advisor calls
# end_conversation; this just stops a model that never converges.
MAX_TURNS: int = int(os.getenv("ADVISORY_MAX_TURNS", "8"))

APP_NAME: str = "financial_advisory"
USER_ID: str = "demo_user"


def make_model() -> Gemini:
    """A Gemini wrapper that retries rate-limit/503 responses with backoff.

    The retry happens inside the GenAI client, so we never replay a partial turn
    and session history stays consistent. Fresh instance per agent.
    """
    return Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(
            attempts=6,
            initial_delay=8.0,
            max_delay=64.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[429, 503],
        ),
    )


def require_api_key() -> None:
    """Fail early with a clear message if the API key is missing."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit(
            "GOOGLE_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your key from "
            "https://aistudio.google.com/apikey"
        )
