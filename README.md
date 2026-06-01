# Multi-Agent Financial Advisor (Google ADK)

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

A client simulator with a profile, an advisor that's the sole point of contact
for both the client and the analyst, and an analyst with internet access plus
some kind of knowledge store. The agents should collaborate and the conversation
should stop when things are resolved.

A couple of assumptions I made (the brief said I could): one client, one session;
the knowledge store is a small curated dataset standing in for what would be a
house-research DB in real life; "internet" means Google Search grounding rather
than arbitrary browsing. I also lean on the client prompt to make the client
actually converge, otherwise two LLMs will keep being polite at each other
forever.

The advisor-is-the-only-hub rule is enforced by the wiring, not just by asking
nicely in the prompt: only the advisor holds a reference to the analyst (as a
tool), and only the advisor exchanges messages with the client. The analyst has
no path to the client at all.

### Prompting

A few choices worth mentioning:

- The client's *identity* is data (a dataclass rendered into the prompt) and its
  *behaviour* is the instruction. Swapping clients doesn't touch any prompt logic.
- I tell the advisor to give the analyst a *specific* brief (the relevant client
  facts plus exactly what to find out), not a vague "help me." Vague briefs are
  where these systems start rambling.
- For ending the conversation I gave the advisor an `end_conversation` tool
  instead of asking it to print a sentinel like `[DONE]` and regexing for it. A
  tool call is unambiguous, it can't accidentally fire because the client quoted
  the word "done," and it carries a summary with it.
- The analyst is told to reach for the knowledge store first and only use web
  search for time-sensitive specifics, and not to make up numbers. Left alone,
  models will happily invent a confident-sounding 4.2%.

The general tradeoff: my prompts are fairly prescriptive, which costs some
creative leeway but buys reliability. For something pretending to be a financial
advisor that felt like the safer side to err on.

### What ADK does under the hood

The convenient stuff:

- **Tool schemas come from the function signature.** `lookup_investments` and
  `end_conversation` are just Python functions; ADK reads their type hints and
  docstrings to build the JSON the model sees. So those hints and docstrings
  aren't decoration, the model literally reads them to decide how to call the
  tool. Worth writing them with that in mind.
- **The tool-calling loop is handled for you.** ADK sends the declarations, gets
  back a function-call request, runs your Python, feeds the result back, and
  re-prompts. You just iterate the event stream.
- **Sessions and state** persist each agent's history and a `state` dict across
  turns. `end_conversation` writes `state["engagement_resolved"]` and the loop
  reads it after the turn.
- **`AgentTool`** lets one agent call another like a function. That's how the
  advisor delegates to the analyst.

The thing that actually bit me: ADK's `google_search` is a built-in,
provider-side tool, and the Gemini API won't let it share an agent with regular
function tools. So an analyst with both `lookup_investments` and `google_search`
just fails. The workaround in `agents.py` is to put `google_search` in its own
tiny `web_search` agent and hand *that* to the analyst as an `AgentTool`. Now the
analyst sees web search as an ordinary tool and can use it alongside the
knowledge store.

A side effect of that workaround: because `AgentTool` runs the sub-agent in its
own nested call, the analyst's own tool calls don't show up in the advisor's
event stream, only the final answer does. If you just read function calls off the
top-level stream (the obvious thing to do) you see "advisor called analyst" and
nothing underneath. `tracing.py` fixes this with a `before_tool_callback` on each
agent that appends to a `ContextVar` the nested async calls inherit, so the
transcript actually shows the analyst hitting the knowledge store and the web.

### Why it's built this way

- **A hand-written advisor<->client loop instead of ADK's agent transfer.** ADK's
  sub-agent transfer is for delegation inside one reasoning tree (a coordinator
  handing off and getting control back). Advisor and client aren't that; they're
  two separate people taking turns, each with their own memory. Folding them into
  one tree would muddy both the personas and the "advisor is the only one who
  talks to the client" boundary. So they each get their own runner and session
  and I shuttle messages between them. The analyst genuinely *is* delegation, so
  it correctly lives inside the advisor as a tool.
- **`AgentTool` for the analyst, not transfer.** The advisor needs the analyst's
  *answer* and then has to keep control to talk to the client. A tool call returns
  a value and leaves the advisor in charge; transfer would hand the conversation
  away.
- **A tool for termination, not a string sentinel.** Ending the conversation is a
  control-flow decision, and I'd rather carry that on a structured signal than
  parse it out of prose the model might phrase ten different ways.

### Python bits

- `ClientProfile` is a frozen dataclass: structured data, no behaviour, immutable
  for a session, with the one `to_prompt` method that knows how to render it.
- Agents are built by `build_*` functions rather than created at import time, so
  importing the package doesn't touch the network and the agents are easy to
  reconfigure or test.
- Secrets and config come from the environment via `python-dotenv`; the key never
  lives in code and `.env` is gitignored. `.env.example` documents what's needed.
- The resolved-state key is a single named constant shared by the tool that
  writes it and the loop that reads it, so they can't drift apart.
