# Ops Agent: Docker & Kubernetes chat

Chat in plain English to inspect and operate Docker (and, later, Kubernetes). A local Ollama
model works out what you want; the backend runs only typed, validated tools, and **only after you
approve them**.

## How a message is handled

1. **Classify.** The LLM reports facts about the message (does it need the live system? which
   platform? read or change?). Code then routes it:
   - **question**: answered by the LLM with no tools and no system access
   - **unsupported**: refused, with a list of what *is* possible (no LLM call, no tools)
   - **action**: the LLM may propose tool calls
2. **Approve.** Every proposed call shows an approval card. Nothing runs until you click.
   - Read-only tools: **Approve**, **Always allow** (revocable under *Permissions*), or **Reject**
   - Change tools: must be approved every time, except **Allow for this session** on the
     build-and-fix tools (writing the Dockerfile/.dockerignore and building): one click lets them
     run on that project without a card until the chat ends. Revoke it from the bar above the
     message box; pre-checks still run; never offered for destructive actions
     (`[approvals] session_allow` in agent.toml)
   - Destructive tools: must be approved every time, and you type the resource name to confirm
   - Change and destructive tools run a read-only **pre-check** first: the card shows what will be
     affected (status, image, ports, who uses an image), and impossible actions ("already running",
     "container is running, needs force") are stopped before any card is shown
   - No answer within 5 minutes, or closing the tab, counts as a rejection
3. **Run and summarize.** The tool runs, you see the raw output, and the LLM summarizes it. Every
   proposal and its outcome goes into the audit log (`GET /api/audit`).

## Run it

Requirements: [uv](https://docs.astral.sh/uv/), Node 20+, [Ollama](https://ollama.com) with
`qwen3:8b` pulled, and Docker Desktop.

```bash
ollama pull qwen3:8b
```

```bash
cd backend && uv sync && uv run uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

Settings are `AGENT_*` env vars or `backend/.env`. See [backend/app/config.py](backend/app/config.py)
(e.g. `AGENT_OLLAMA_MODEL=qwen2.5-coder:7b`, `AGENT_OLLAMA_NUM_CTX=32768`, `AGENT_OLLAMA_THINK=true`,
`AGENT_MAX_ERRORS=10`).

Steps are not limited: the agent keeps going while it makes progress, and **Stop** (the button that
replaces Send while it works) finishes the current step and runs nothing else. Only failures are
capped: after `AGENT_MAX_ERRORS` errors in one request it stops and lists the last ones.
Thinking is off by default for speed. For models without a thinking mode (e.g. `llama3.1:8b`),
set `AGENT_OLLAMA_THINK=none` so the option isn't sent.

## Tests

```bash
cd backend && uv run pytest
```

Check against the real model (re-run after changing prompts or models). It scores routing
(action / question / unsupported) and tool calling (right tool and arguments, and whether the model
used the tool-calling API or wrote the call as text). Nothing is executed:

```bash
cd backend && uv run python -m scripts.eval_routing
```

## Adding a tool

Add a function in `backend/app/tools/` (see [docker_tools.py](backend/app/tools/docker_tools.py)):

```python
@registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Pause '{a.container}'", preview=_preview_pause)
def docker_pause_container(args: ContainerArgs) -> dict:
    """Pause a running container."""
    ...
```

The name prefix sets the platform (`docker_…`, `k8s_…`), the docstring is what the LLM sees, and
the argument model is validated before approval. Destructive tools need `confirm_field=...`.
Change and destructive tools should also pass `preview=...`: a read-only function that returns
`{"facts": {...}, "warnings": [...]}` for the approval card, or raises `ToolError` to stop early.

## Context management (feature flags)

Every model call re-sends the conversation, so long tasks fill the context window. These techniques
keep what the model sees small and current; each one is a switch in
[backend/agent.toml](backend/agent.toml), documented there, and can be turned off to compare or
revert (env vars override the file, e.g. `AGENT_CONTEXT__ATTEMPT_LOG=false`):

1. **Compact tool output**: logs → last N lines, tables → first rows (the UI still shows everything)
2. **Replace outdated outputs**: older reads/writes of the same file become a one-line note
3. **Attempt log**: earlier refused/failed attempts are listed in the instructions
4. *(rejected: fresh focused context per fix attempt; the reason is recorded in the file)*
5. **Relevant tools only**: Docker tools for Docker requests, project tools for code requests
6. **Measure and compact**: past 75% of the window, the oldest tool outputs are shortened first
7. **Build facts**: `project_context` states the detected package manager, start command and port

The full history is always kept; only the copy sent to the model is trimmed
([context.py](backend/app/context.py)).

## Choosing a model (local + free cloud)

Every chat starts on the **local model** (Ollama). The dropdown in the header switches the chat to
a free cloud model (Groq, Google Gemini, OpenRouter `:free`, NVIDIA); the conversation carries
over. Only genuinely free tiers are allowed: OpenRouter models must end in `:free`, Ollama must be
local, and Gemini is only free if its key's project has no billing linked.

1. Add the keys you have to `backend/.env` (yourself; a missing one shows as unavailable):
   `AGENT_GROQ_API_KEY`, `AGENT_GEMINI_API_KEY`, `AGENT_OPENROUTER_API_KEY`, `AGENT_NVIDIA_API_KEY`
2. Check the model ids against the providers (lists models only, uses no free requests):

```bash
cd backend && uv run python -m scripts.check_models
```

**Usage counter.** Each cloud model shows its free calls left today. Every request is reserved
before it's sent (requests in flight count), settled with the tokens the provider reports, kept in
SQLite per provider day (reset at the provider's midnight), and shared by all chats. Groq's own
daily count overrides ours whenever it reports it. Per-minute limits are waited out with a
countdown. When a model is down to **2 calls** (`[quota] reserve_calls`), the agent stops cleanly
before the next request and the model is greyed out until its reset. Models, limits and pools are
in `agent.toml` (`[[models]]`, `[pools.*]`); where sources disagreed, the lower limit is used.

## Versions, redeploy and rollback

Every successful build is a **version** with its own tag, `<project>:v<N>-<commit>` (plus `-dirty`
for uncommitted changes, and `:latest` for the newest), recorded in SQLite with whether it ever
passed a health check.

- **Redeploy** ("redeploy sample-project"): build, then run. If a version is already running, the
  new one starts next to it on a spare port and is health-checked there first; only then does it
  take over the real port. If the trial fails, the old version never stopped; if the real port
  fails, the old version is put back.
- **Rollback** ("roll back sample-project"): run with `version="previous"`, which code resolves to
  the newest healthy version older than the running one (or a specific one, e.g. "v2").
- **History** ("which versions do I have?") and **cleanup** ("clean up old images", destructive).

## Code projects

Register folders under **Projects** in the UI (only you can; the agent has no tool for it). The
agent can then read them through a sandbox ([workspace.py](backend/app/workspace.py)):

- paths can't leave the project folder (`../`, absolute paths and outside symlinks are refused)
- secret files (`.env*`, keys, credentials) are never read; only their variable names are shown
- secret-looking values in other files are masked ([redact.py](backend/app/redact.py))
- `.git/`, `node_modules/`, `.venv/`, build folders are skipped
- reading asks the first time; **Always allow reading &lt;project&gt;** covers that project only

| Tool | Risk | What it does |
|---|---|---|
| `project_list` | read | Registered projects |
| `project_context` | read | Overview **plus** key files (manifest, entry point, config, Docker files) in one call; lock files skipped |
| `project_overview` | read | Detected stack, Docker files, secret files (names only), git branch/commit, file tree |
| `project_list_files` | read | Folder tree |
| `project_read_file` | read | Up to 400 lines, secrets masked; lock files return a short note instead |
| `project_search` | read | Text/regex search with paths and line numbers |
| `project_write_file` | change | Writes `Dockerfile` / `.dockerignore` at the root, after rule checks; you approve the exact diff |
| `project_build_image` | change | Real `docker build` (BuildKit). Card shows base image, RUN commands, tags, what's sent; refuses if a secret file would be sent. Live output; Stop cancels it |
| `project_run_container` | change | Runs `<project>:latest` as a container: port from EXPOSE on 127.0.0.1 only, settings from the project's `.env` (names shown, values never), replaces only containers the agent started; waits for an HTTP answer and returns the URL, or the exit code + logs + explanation |
| `project_versions` | read | Every build of a project (v1, v2, …): built when, commit, size, health check result, which one is running |
| `project_prune_images` | destructive | Removes old images; keeps the newest build, anything a container uses and the last N healthy versions (default 3). Type the project name |

## Building images (build-and-fix loop)

`project_build_image` builds the project's Dockerfile and tags the image `<project>:<git-sha>`
(`-dirty` with uncommitted changes, a timestamp outside git) and `<project>:latest`. If the build
fails, the end of the log goes back to the model, which fixes the Dockerfile or .dockerignore
(new diff card) and builds again, until it works, you reject or Stop, or the error budget runs
out. Rebuilding exactly the files that just failed is refused (one retry with `no_cache` is
allowed, for network failures). You can follow the loop: every failed step is announced in the
chat, a "not found" error says whether the file is missing or excluded by `.dockerignore`, every
file change says why on its card, and builds show their attempt number. Settings: `[build]` in
[backend/agent.toml](backend/agent.toml). Why it shells out to `docker build` instead of using the
Docker SDK, and other trade-offs: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

The API only accepts browser requests from the app itself (`AGENT_ALLOWED_ORIGINS`, default
`http://localhost:5173`), so other websites open in your browser can't drive the agent.

## Models that write tool calls as text

Some models (e.g. `qwen2.5-coder:7b`; the default `qwen3:8b` uses native tool calls) reply with `{"name": ..., "arguments": ...}`
as text instead of using Ollama's tool-calling API. The agent recovers such calls
([text_tool_calls.py](backend/app/text_tool_calls.py)) only when `name` is a registered tool, hides
the raw JSON from the chat, and runs them through the same validation, pre-checks and approval as
any other call.

## Docker tools

| Tool | Risk | Notes |
|---|---|---|
| `docker_list_containers`, `docker_container_logs`, `docker_inspect_container`, `docker_list_images`, `docker_container_stats` | read | Can be always-allowed |
| `docker_start_container`, `docker_stop_container`, `docker_restart_container` | change | Pre-check: not already in that state |
| `docker_pull_image` | change | Pre-check: valid reference; shows whether a local copy exists |
| `docker_remove_container` | destructive | Type the name; running containers need `force` |
| `docker_remove_image` | destructive | Type the name; refused while running containers use it |

## Layout

```
backend/app/
  main.py         FastAPI app, WebSocket /ws/chat, REST endpoints
  agent.py        classify → answer / refuse / act loop, approval gate
  approvals.py    pending actions, decisions, timeouts
  prompts.py      classifier, action and question prompts
  llm.py          Ollama client (streaming + tool calls)
  store.py        SQLite: always-allow grants, audit log
  tools/          registry, Docker tools, project tools, build tool
frontend/src/
  App.tsx         chat UI
  useChat.ts      WebSocket protocol
  components/     approval card, tool output, permissions panel
```

## Roadmap

- [x] Phase 0: chat, three message types, approval gate, audit log
- [x] Phase 1: Docker read-only (containers, logs, inspect, images)
- [x] Phase 2: Docker actions (start, stop, restart, pull, stats, remove container/image)
- [x] Build & deploy, step 1: registered projects, read-only code access, secret protection
- [x] Build & deploy, step 2: write Dockerfile into the repo (diff card), Dockerfile checks
- [x] Agent robustness: error budget instead of a step limit, Stop button, project_context tool,
      lock files skipped, 32k context, context management flags (backend/agent.toml)
- [x] Build & deploy, step 3: build the image (BuildKit) with live output and a build-and-fix loop; build facts
- [x] Model dropdown: local model + free cloud models (Groq, Gemini, OpenRouter, NVIDIA) with a precise usage counter
- [x] Build & deploy, step 4: run the container + health check (fix loop covers running)
- [x] Build & deploy, step 5: versions, safe redeploy, rollback, cleanup
- [ ] Phase 3: Kubernetes read-only
- [ ] Phase 4: Kubernetes actions (scale, rollout restart, delete pod, apply with dry-run)
- [ ] Phase 5: multi-step diagnosis
- [ ] Phase 6: auth, multi-cluster, Helm
