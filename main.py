"""Entry point: run one financial-advisory session and print the transcript.

    uv run python main.py
"""

from __future__ import annotations

import asyncio

from advisory.orchestrator import format_transcript, run_advisory_session
from advisory.profiles import SAMPLE_CLIENT


async def _main() -> None:
    print("=" * 72)
    print("Multi-agent financial advisory session (Google ADK)")
    print(f"Client: {SAMPLE_CLIENT.name}")
    print("=" * 72 + "\n")

    result = await run_advisory_session(SAMPLE_CLIENT)
    print(format_transcript(result))


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
