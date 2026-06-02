# Multi-Agent Financial Advisor (Google ADK)
Welcome to my Multi Agent System!
Three agents that work together to advise a (simulated) client on their
investments. The advisor talks to the client, hands research tasks to an analyst,
and the conversation runs until the advisor is satisfied the client's questions
are answered.

```
        ┌─────────────┐   dialogue (turn by turn)   ┌──────────────┐
        │   CLIENT    │  ◄───────────────────────►  │   ADVISOR    │
        │ (simulator) │                             │(orchestrator)│
        └─────────────┘                             └──────┬───────┘
         role-plays a                                      │ delegates research
         generated profile                                 ▼  (AgentTool)
                                                    ┌──────────────┐
                                                    │   ANALYST    │
                                                    └──────┬───────┘
                                              ┌────────────┴────────────┐
                                              ▼                         ▼
                                     ┌──────────────────┐     ┌──────────────────┐
                                     │  knowledge store │     │   web search     │
                                     │ (function tool)  │     │ (google_search)  │
                                     └──────────────────┘     └──────────────────┘
```

- **Client** role-plays a generated profile (age, risk appetite, income, assets,
  current holdings, goals, time horizon) and says when it's satisfied.
- **Advisor** is the only agent that talks to both the client and the analyst. It
  figures out the client's situation, writes a research brief for the analyst,
  turns the findings into plain-language advice, and ends the session when done.
- **Analyst** does the research using an in-house knowledge store and live web
  search. It never speaks to the client directly.

## Running it

You need [uv](https://docs.astral.sh/uv/) and a Google AI Studio API key.

```bash
uv sync                       # install deps from pyproject / uv.lock

cp .env.example .env          # then paste your key into GOOGLE_API_KEY
                              # key from https://aistudio.google.com/apikey

uv run python main.py         # run one advisory session
```

It prints the conversation with each agent labelled, plus indented `↳ [...]`
lines showing the tool calls behind the scenes (advisor calling the analyst, the
analyst hitting the knowledge store / web search, and the final
`end_conversation`).

Optional settings in `.env`:

| Variable             | Default            | Purpose                                  |
| -------------------- | ------------------ | ---------------------------------------- |
| `GOOGLE_API_KEY`     | (required)         | Gemini API key                           |
| `ADVISORY_MODEL`     | `gemini-2.5-flash` | model for all agents                     |
| `ADVISORY_MAX_TURNS` | `8`                | cap on advisor↔client round-trips        |

### Layout

```
advisory/
  config.py        env loading, model factory, constants
  profiles.py      ClientProfile dataclass + sample client
  knowledge.py     the analyst's knowledge store + lookup tool
  agents.py        builds the three agents and wires up their tools
  tracing.py       surfaces nested tool calls so you can see the collaboration
  orchestrator.py  the advisor<->client loop and termination
main.py            run one session, print the transcript
```

## Notes on the five points of interest

I'll cover prompting, framework assumptions, first-principles
reasoning, Pythonic style, and the assignment details. My thoughts on each:

### The assignment

The assignment asked for a client simulator with a profile, an advisor that's the sole point of contact
for both the client and the analyst, and an analyst with internet access plus
some kind of knowledge store. The agents should collaborate and the conversation
should stop when things are resolved.

A couple of assumptions I made (the brief said I could): one client, one session;
the knowledge store is a small curated dataset standing in for what would be a
house-research DB in real life; "internet" means Google Search grounding rather
than arbitrary browsing. I also rely on the client prompt to make the client
actually converge, otherwise two LLMs will keep being polite at each other
forever.

I enforced "advisor is the only hub" by the setup of my code instead ofjust by asking
nicely in the prompt. Only the advisor holds a reference to the analyst (as a
tool), and only the advisor exchanges messages with the client. The analyst has
no path to the client at all.

### Prompting

A few choices worth mentioning:

- The client's *identity* is data (a dataclass rendered into the prompt) and its
  *behaviour* is the instruction. Swapping clients doesn't touch any prompt logic.
- I tell the advisor to give the analyst a *specific* brief (the relevant client
  facts plus exactly what to find out), not a vague "help me." Vague briefs are
  where the agents start rambling.
- For ending the conversation I gave the advisor an `end_conversation` tool
  instead of asking it to print a sentinel like `[DONE]` and regexing for it. A
  tool call is unambiguous, it can't accidentally fire because the client quoted
  the word "done," and it carries a summary with it.
- The analyst is told to reach for the knowledge store first and only use web
  search for time-sensitive specifics, and not to make up numbers. Left alone,
  models will happily invent a confident sounding 4.2%.

The general tradeoff: my prompts are prescriptive, which costs some
creative leeway but buys reliability. For something pretending to be a financial
advisor that felt like the safer side to err on.

### What ADK does under the hood

What it handles for you:

- **Tool schemas** are built from each function's type hints and docstrings, so
  `lookup_investments` and `end_conversation` need no manual JSON.
- **The tool-calling loop** (declare, call, feed result back, re-prompt) runs
  itself; you just iterate the event stream.
- **Sessions and state** persist per-agent history and a `state` dict; that's how
  `end_conversation` flags the loop to stop.
- **`AgentTool`** lets the advisor call the analyst like a function.

The gotcha: ADK's built-in `google_search` can't share an agent with regular
function tools, so an analyst holding both it and `lookup_investments` just fails.
The fix in `agents.py` is to isolate `google_search` in its own `web_search`
agent and give that to the analyst as an `AgentTool`, alongside the knowledge
store.

That nesting hides the analyst's own tool calls, though: only its final answer
reaches the advisor's event stream. `tracing.py` recovers them with a
`before_tool_callback` writing to a `ContextVar` the nested calls inherit, so the
transcript shows the analyst actually hitting the knowledge store and the web.

### Why it's built this way

- **Hand-written advisor<->client loop, not ADK's agent transfer.** Transfer is
  for delegation within one reasoning tree; advisor and client are two separate
  personas taking turns, each with their own memory. So each gets its own runner
  and session and I pass messages between them. The analyst *is* delegation, so it
  lives inside the advisor as a tool.
- **`AgentTool` for the analyst, not transfer.** The advisor needs the analyst's
  answer but must keep control to talk to the client. A tool call returns a value
  and leaves it in charge; transfer would hand the conversation away.
- **A tool for termination, not a string sentinel.** Ending is a control-flow
  decision, better carried on a structured signal than parsed out of prose.

### Python bits

- `ClientProfile` is a frozen dataclass: structured data, immutable
  for a session, with the one `to_prompt` method that knows how to render it.
- Agents are built by `build_*` functions rather than created at import time, so
  importing the package doesn't touch the network and the agents are easy to
  reconfigure or test.
- Secrets and config come from the environment via `python-dotenv`; the key never
  lives in code and `.env` is gitignored. `.env.example` documents what's needed.
- The resolved-state key is a single named constant shared by the tool that
  writes it and the loop that reads it, so they can't drift apart.
