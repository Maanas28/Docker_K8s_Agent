# Limitations and the workarounds they forced

Why some parts of the code look the way they do. Each entry: the limitation, what we did about
it, and where it lives. Add an entry whenever a limitation (of the model, a library, Docker, the
machine) changes the design, including what was rejected and why.

## The model (local 8B LLM via Ollama)

| Limitation | Workaround | Where |
|---|---|---|
| The machine has 18 GB RAM, shared with Docker, so only ~8B models fit. | `qwen3:8b` by default; models are swappable with `AGENT_OLLAMA_MODEL`. | `app/config.py` |
| `qwen2.5-coder:7b` never made native tool calls; it wrote them as JSON text. | Switched to `qwen3:8b` (2026-09-30). Kept a recovery step that turns JSON-text calls into real ones, only for registered tool names, still through validation and approval. | `app/text_tool_calls.py` |
| The model sometimes *describes* a tool call ("please call project_overview…") instead of making it. | One hidden nudge asking for the call itself; the retry isn't kept in the history. | `agent.py` (`_NUDGE`) |
| Thinking mode is slow (~5 min vs ~45 s for a Dockerfile). | Thinking off by default; `AGENT_OLLAMA_THINK=true` turns it on. | `app/config.py` |
| Ollama's default context is a few thousand tokens and silently drops the *oldest* text (the instructions and the request) when full. | 32k context, plus the context-management techniques (compact outputs, replace outdated ones, attempt log, relevant tools only, measure and compact). | `agent.toml`, `app/context.py` |
| Lock files are huge and useless to the model, and filled the window. | Lock files are never read; a short note points at the manifest. | `app/workspace.py` (`LOCK_FILES`) |
| Given every tool, a small model picks wrong ones more often. | Only the tools for the request's platform are sent. | `agent.toml` (5) |
| A stuck model repeats a refused call forever; a fixed step limit also cut off healthy long tasks. | Repeat guard for identical refused calls, and an error budget (10 per request) instead of a step limit; Stop button. | `agent.py` |
| The model invents container/image names. | "Grounded" arguments: the value must have appeared in user text or tool output. | `registry.py`, `agent.py` |
| Dockerfile *content* is often wrong (wrong install command, paths, port). | Rule checks before the diff card (secrets, missing files); since Step 3 a real build judges it and the model fixes what fails; build facts give it the package manager, start command and port. A per-mistake check was rejected (2026-09-30): generic fixes only. | `app/dockerfile_check.py`, `build_tools.py`, `agent.toml` (7) |
| The model offered to build/run/deploy before tools for it existed ("Fix 4"). | One generic prompt rule: the tools are the complete list of what it can do; say plainly when something isn't available yet. | `app/prompts.py` (`ACTION_PROMPT`) |

## Step 3: building images (2026-10-03)

| Limitation | Workaround | Where |
|---|---|---|
| **docker-py (the Docker SDK) only drives the legacy builder**, not BuildKit. The legacy builder is deprecated and fails on valid Dockerfiles (e.g. `RUN --mount=type=cache`, which uv's own Docker guide uses), so the fix loop would "fix" files that aren't broken. | Builds run the real `docker build` CLI (BuildKit, `--progress=plain`) as a subprocess. Cost: needs the `docker` CLI (looked up on PATH, then Docker Desktop / Homebrew / OrbStack paths) and output is plain text instead of JSON. | `app/tools/build_tools.py`, `agent.toml` `[build]` |
| Tools ran in worker threads, and a blocking read in a thread can't be interrupted, so a build couldn't be stopped or timed out cleanly. | Tools may now be `async def`: the build uses an asyncio subprocess and races its output against the Stop event and the timeout, then terminates the CLI (which cancels the BuildKit build). | `registry.py` (`RunContext`, `Tool.call`), `agent.py` |
| "Stop finishes the current step" would mean waiting minutes for a build. | Stop now also cancels a running long tool (a build). Nothing is half-done: BuildKit only tags an image when the build succeeds. A cancelled build doesn't count as an error. | `agent.py` (`request_stop`) |
| A build prints thousands of lines; one WebSocket message per line floods the browser. | Lines are batched (every 250 ms) into `tool_progress` events; the UI keeps the newest 2,000 while running, and the full log (up to 5,000 lines) comes with the result. | `agent.py` (`_Progress`), `useChat.ts` |
| The model sees only the last 60 lines of a failed build. An error printed earlier would be missed (rare: BuildKit ends with the failing step and a `failed to solve` summary). | The summary line is also sent as a separate `error` field. A successful build's log isn't sent to the model at all. | `app/context.py` |
| Tool output for the model is cut from the end when it's too long (`tool_output_llm_chars`), which would drop a log's error. | The build result puts `error` before `log`, so it survives the cut. | `build_tools.py` |
| Rebuilding unchanged files after a failure wastes minutes and fails the same way. | The pre-check refuses an identical rebuild (hash of Dockerfile + .dockerignore), unless `no_cache=true` (for network failures). Kept in memory only: a backend restart forgets it. | `build_tools.py` (`_last_failed`) |
| The secret check uses file-name patterns, so a package's public `cacert.pem` inside `.venv` looked like a secret. | Files inside dependency/editor folders aren't secret-checked; sending those folders at all is a separate warning. | `workspace.py` (`build_context`) |
| Our own `.dockerignore` matching could disagree with Docker's, and a disagreement could leak `.env` into an image. | Uses docker-py's `.dockerignore` matcher (`exclude_paths`), the same rules Docker applies. | `workspace.py` |
| Only the Dockerfile and .dockerignore are pinned between approval and build; source files can still change in between. | Accepted: the source is what you want built. The card's RUN lines and base image can't change unseen. | `build_tools.py` |
| A successful build doesn't prove the app starts (wrong start command or port still builds). | Not solved in Step 3; Step 4 runs the container with a health check. Build facts reduce the risk. | roadmap |
| Build facts are heuristics (port detection is a regex). | Labelled "detected" / "detected, check", with the file and line; behind the `build_facts` flag. | `code_tools.py`, `agent.toml` (7) |
| Image names must be lowercase alphanumerics with separators. | Project names are normalised for the image name (`Shop..API` → `shop-api`). | `build_tools.py` (`_repo_name`) |
| The `-dirty` tag counts *any* uncommitted change, including untracked files (unlike `git describe --dirty`), because an untracked Dockerfile does change the image. | Documented here. | `build_tools.py` (`image_tags`) |
| Builds on an Apple Silicon Mac produce arm64 images. | Not handled yet; matters for Kubernetes later (`--platform`). | roadmap |
| Each rebuild leaves untagged images behind. | Not handled yet; a prune tool later. | roadmap |

## Fixes after the first real build run (2026-10-03)

The first real run ended after 10 errors: the model wrote a .dockerignore that excluded `uv.lock`
and a Dockerfile that copies it, then never connected the build error to the .dockerignore.

| Limitation | Workaround | Where |
|---|---|---|
| Docker's error for a missing COPY source (`"/uv.lock": not found`) doesn't say *why*: missing from the project, or excluded by .dockerignore. The 8B model kept rewriting the Dockerfile and re-sending the same .dockerignore. | After a failed build, code adds "Why:" facts to the error: the file exists but is excluded by pattern `X` (Docker's own matcher, `!` re-includes respected), or the file doesn't exist. Generic for every missing COPY/ADD source. | `build_tools.py` (`diagnose`) |
| The repeat guard assumed "identical call = identical result", which is false once something changed in between (a fixed Dockerfile made the identical build call valid again, but it was refused). | Earlier refusals are forgotten whenever a change tool succeeds. | `agent.py` (`_run_call`) |
| The model used `no_cache=true` as a way around "nothing changed since the failed build", repeatedly. | `no_cache` may skip that guard once per failed set of files; after a failed no-cache build it's refused too. | `build_tools.py` (`_last_failed`) |
| The small model often goes straight to the next tool call without telling the user what failed or what it's changing, so the fix loop was hard to follow. | Code announces every failed step in the chat (`step_failed`: what failed, the reason, whether it will keep trying); `project_write_file` has a required `reason` shown above the diff ("Why: …"); builds show "attempt N"; the prompt asks for one sentence before each fix and a summary of the fixes after a success. | `agent.py`, `code_tools.py`, `build_tools.py`, `prompts.py`, `App.tsx`, `ToolCard.tsx` |

## Fixes after the second real run (2026-10-03)

The .dockerignore explanation worked (the model removed `uv.lock` from it). Then the loop got
stuck on `uv`: `python:3.12-slim` doesn't include it, and the model alternated between
`uv pip install uv` (using uv to install uv) and `apt-get install uv` (not an apt package).

| Limitation | Workaround | Where |
|---|---|---|
| The build only says `exit code: 127`; the model doesn't know that means "command not found", nor that apt's "Unable to locate package" means the package doesn't exist there. | "Why:" facts for both: which program isn't installed at that step and the base image, or which name apt doesn't know. | `build_tools.py` (`diagnose`) |
| The model retried the same broken fix three times; nothing stopped it until the error budget ran out. | Same-error brake: the same tool failing with the same error twice in a row (ids ignored) stops the loop; the agent lists the reasons of the changes it tried and asks the user. Switch: `[loop] same_error_brake`. | `agent.py`, `agent.toml` |
| After a failure the model explained its fix and then stopped without making the call, so the user had to type "ok, fix it". Partly caused by our own prompt ("write a sentence, then make the call"). | Prompt now says sentence *and* call in the same reply, no waiting for an OK; plus one hidden nudge when a failure is followed by a text-only reply (the text stays visible). Switch: `[loop] failure_nudge`. | `prompts.py`, `agent.py`, `agent.toml` |
| The model sometimes escapes line breaks twice in a tool call, so the file gets a literal `\n` (a whole RUN on one broken line). | The write pre-check refuses content with a literal `\n` before indentation or an instruction, and asks for real line breaks. Legitimate `\n` inside strings (e.g. `printf "a\nb"`) is allowed. | `code_tools.py` (`_ESCAPED_NEWLINE`) |
| The model doesn't know how uv (or Poetry, pnpm…) is installed in an image. | **Not fixed on our side on purpose**: it's model knowledge. Options under consideration: install facts per package manager (idea B), thinking on, or a bigger/cloud model (`kimi-k2.5:cloud` on Ollama Cloud). | — |

## Model dropdown and free cloud models (2026-10-05)

| Limitation | Workaround | Where |
|---|---|---|
| The local 8B model lacks the knowledge/reasoning for build debugging (e.g. how uv is installed). | A model dropdown: start locally, switch the chat to a free cloud model for harder steps. History is in our own format, so it carries over. | `models.py`, `llm_openai.py`, `ModelPicker.tsx` |
| Only free use is acceptable (not a penny). | Allow-list of providers with real free tiers; OpenRouter must be `:free`; Ollama must be local (no Ollama Cloud); no Mistral/Cerebras. Gemini can still bill if its project has billing linked: the user must use a project without billing (can't be checked from the API). | `config.py` (`_check_models`) |
| Providers don't tell us how much is left (except Groq's daily request count). | Our own counter: reserve before sending, settle with reported tokens, persist per provider day, share across chats; Groq's header overrides it ("confirmed by the provider"). | `quota.py` |
| Our counter can't see usage from outside this app (same key in another tool), except on Groq. | Shown as "counted by this app"; limits set to the lower published figure; the 2-call reserve; a daily 429 marks the model used up until reset anyway. | `quota.py`, `agent.toml` |
| Published free limits disagree (Gemini 3.8 Flash: 20 vs 1,500/day) and change. | The lower figure in `[pools.*]`, with a comment to raise it to what the console shows. Groq corrects its own. | `agent.toml` |
| Per-minute windows are in memory. | A restart forgets at most the last 60 s; a 429 is still waited out with `retry-after`. | `quota.py` |
| A provider that sends no token usage. | Tokens are estimated (4 chars ≈ 1 token) and shown separately as estimated. | `quota.py`, `ModelUsage.estimated_tokens` |
| The OpenAI-style API needs an id on every tool call; our history had none. | Ids are generated when converting (and real ones kept on new calls). Tool results whose call fell out of the window are sent as plain text. | `llm_openai.py` (`to_openai`) |
| Gemini 3 rejects tool calls in the history without a "thought signature" (e.g. calls made by the local model before switching). | Real signatures are kept with each call; others get Google's documented placeholder `skip_thought_signature_validator`. **To verify on the first real Gemini run.** | `llm_openai.py` |
| Not every provider/model supports the same JSON mode for the router. | Classification tries `json_schema`, then `json_object`, then plain instructions; each attempt is counted. | `llm_openai.py` (`classify`) |
| Some open models put `<think>…</think>` reasoning in their reply. | Stripped from the stream, even when split across chunks. | `llm_openai.py` (`_ThinkFilter`) |
| A call can be bigger than a provider's tokens-per-minute limit (Groq free: 8,000), and then can never succeed. | Refused up front with a clear message (start a new chat or pick another model). The context trimming keeps calls around 4-8k tokens. | `quota.py` (`check`) |
| Exact model ids (e.g. Groq's Qwen3.8 27B, OpenRouter's rotating free list) weren't verifiable without keys. | Marked "verify" in agent.toml; `scripts.check_models` checks them and lists free OpenRouter models with tool calling. | `scripts/check_models.py` |

## First real cloud run (2026-10-05)

| Limitation | Workaround | Where |
|---|---|---|
| A free endpoint (Qwen3.8 27B on OpenRouter) didn't split the model's XML-style tool call into arguments, so `</parameter reason=…`, `</think>`, `<tool_call>` ended up inside the Dockerfile and .dockerignore. Writing the file "from scratch" doesn't help: it already sends the whole file; the failure is in transport. | Every write is refused if its content contains tool-call/reasoning markup (with the line), and the model is told to send only the file in `content`. Prefer models with structured tool calls (local Ollama, Gemini) for writing files. Rejected for now: sending file content outside the tool call (option C), more complex and unreliable for small models. | `code_tools.py` (`_CALL_MARKUP`) |
| A line that isn't a Dockerfile instruction was only found by the build. | Instructions are checked at write time against Docker's list (heredoc bodies skipped). | `dockerfile_check.py` (`INSTRUCTIONS`) |
| The stop message hid `</parameter>` (Markdown drops HTML-looking text), repeated reasons and cut the error off. | `<` and backticks are escaped, reasons de-duplicated, the error shown as a wrapped quote. | `agent.py` (`_same_error_twice`, `_md_text`) |
| httpx timeouts have no text, so "Can't reach NVIDIA · GLM 5.3:" said nothing; 404s hid the provider's reason. | Timeouts say "didn't answer within N s"; errors include the provider's own message (JSON `error.message`, Gemini's list form too). | `llm_openai.py` (`_unreachable`, `_provider_message`) |
| Free endpoints are slow or briefly overloaded (GLM 5.3 timed out at 120 s; Gemini answered 503). | Timeout configurable, default 180 s (`[quota] request_timeout_s`); 502/503/504 retried twice (3 s, 8 s) with a countdown, each try counted to be safe. | `llm_openai.py`, `agent.toml` |
| Gemini 2.5 Flash is in the model list but the chat endpoint answered 404. | Unknown yet; the next 404 shows Google's own reason. | — |

## Step 4: running containers (2026-10-05)

| Limitation | Workaround | Where |
|---|---|---|
| A successful build doesn't prove the app starts (wrong start command, port, host). | The agent runs the image and waits for an HTTP answer (any status below 500 on /health, /healthz or /); failures go back into the fix loop. | `run_tools.py` |
| The model would make up ports, names and settings, and its arguments could leak secrets. | The model only names the project; code takes the port from EXPOSE, the name from the project, and the settings from `.env`. | `run_tools.py` |
| The app needs its `.env` settings, which are secret. | Values are read by code and handed to Docker only; the card shows names; exact values (and secret-looking text) are masked in any log; inspect already shows names only. A port setting that differs from EXPOSE is warned about without showing the value. | `workspace.py` (`env_values`), `run_tools.py` (`_Logs`) |
| A published port is reachable from the whole network by default. | Published on 127.0.0.1 only (`[run] bind_address`). | `agent.toml` |
| Replacing an old container is normally a destructive action. | Only containers carrying this agent's label for that project are replaced (shown on the card, checked again just before removal); any other container with that name stops the run. | `run_tools.py` (`LABEL`) |
| Docker's error for an app that won't come up is just an exit code, or silence. | "Why:" facts: exit codes 126/127/137/139, ModuleNotFoundError, "Address already in use", listening on 127.0.0.1, or on a port other than EXPOSE. | `run_tools.py` (`diagnose`) |
| Asking Docker for logs "since a time" works at whole seconds, which can repeat or skip lines. | The whole log is fetched each poll and only the new part shown (fine for a 30 s health check). | `run_tools.py` |
| A container that runs but never answers would keep the port busy for the next attempt. | It's stopped after a failed health check. A container left by Stop stays running (the message says so). | `run_tools.py` |

## Step 5: versions, redeploy, rollback, cleanup (2026-10-08)

| Limitation | Workaround | Where |
|---|---|---|
| Tags `<commit>-dirty` were reused by every rebuild of uncommitted work, so earlier builds lost their name and couldn't be rolled back to. | Every build gets a unique `v<N>-<commit>[-dirty]` tag and a row in `ImageVersion`. Images built before this show as "from before version tracking". | `build_tools.py` (`image_tags`), `store.py` |
| A published port belongs to one container, so two versions can't share it while the new one is checked. | Safe switch: the new version runs on a spare port and is checked; only then is the old one stopped (kept, renamed `-old`), the new one started on the real port and checked again; if that fails the old one is renamed back and restarted. Downtime is only between those two steps (seconds). A reverse proxy would remove it entirely; rejected as too heavy for local use (Kubernetes rollouts give that later). | `run_tools.py` |
| The model could name a version that doesn't exist, or the wrong "previous" one. | `version` is resolved by code against recorded versions: "previous" = newest healthy version older than the running one (read from the container's `ops-agent.version` label). Unknown or removed versions are refused with the list of real ones. | `run_tools.py` (`_resolve`) |
| Cleanup is destructive and could remove something still needed. | Keeps the newest build, anything any container uses, and the last N healthy versions; the card lists exactly what goes; the plan is recomputed at run time and only images that were on the approved card are removed. Type the project name to confirm; never session-allowed. | `deploy_tools.py` |
| Image sizes differ between `docker images` (e.g. 328 MB) and the API (e.g. 77 MB) with Docker Desktop's containerd image store (unpacked vs stored size). | The API's figure is shown; documented here. | `deploy_tools.py` |

## Health check fix (2026-10-09)

| Limitation | Workaround | Where |
|---|---|---|
| The health check accepted any answer below 500 on any path, so an app whose `/health` answered 500 passed because `/healthz` answered 404 (the route doesn't exist). v3 and v4 of sample-project went live with a broken `/health`. | 404/405 means "this path doesn't exist, try the next"; 5xx means "unhealthy" and never falls through; only the last path (`/`) may count a 404 as up. v3/v4 were corrected to "failed" in the history (checked with `docker cp`: exactly those two contain the bug). | `run_tools.py` (`_probe_paths`) |
| "Keep trying" had no visible limit. | Like Docker HEALTHCHECK / Kubernetes probes: one attempt per `health_interval_s` (1 s) for at most `health_timeout_s` (30 s, so 30 attempts); `health_failure_threshold` (5) 5xx answers in a row fail at once; shown on the card and in the result ("attempt 4", "attempts 3–7"). | `run_tools.py` (`_check`), `agent.toml` `[run]` |
| A Python traceback in the logs was only in the log tail. | The last exception line ("RuntimeError: broken on purpose") is added to the "Why:" facts. | `run_tools.py` (`diagnose`) |

## Phase 3: Kubernetes, read-only (2026-10-12)

| Limitation | Workaround | Where |
|---|---|---|
| A kubeconfig can point at real (work) clusters and holds their credentials. | Only contexts in `[kubernetes] allowed_contexts` (default `docker-desktop`) are used, chosen by code, never by the model; credentials never leave the SDK. | `k8s_tools.py` (`pick_context`) |
| The Python SDK has no "kubectl describe" (that logic lives in kubectl). | Our own `k8s_describe_pod`: pod + its events, plus "why" facts for CrashLoopBackOff, image pull errors, config errors, OOMKilled, unschedulable pods, failing probes. | `k8s_tools.py` (`explain_pod`) |
| Pod specs can contain secret values inline (env). | Env variables by name only; `envFrom` shows which Secret/ConfigMap, never its data; Secrets are never read; logs and event messages are masked. | `k8s_tools.py` |
| The SDK is synchronous. | Like the other sync tools it runs in a worker thread; every request has a timeout (`request_timeout_s`). | `registry.py` (`Tool.call`) |
| Docker Desktop's cluster in "kind" mode can't see images built on the Mac. | The example manifest uses `imagePullPolicy: Never` and says to use "kubeadm" mode; the "why" facts explain ErrImageNeverPull. | `deploy/sample-project.yaml` |
| Changes (scale, restart, apply) aren't possible yet. | Kubernetes change requests are classified as unsupported ("I can't make changes to Kubernetes yet"); the prompt forbids claiming changes. Phase 4 may use `kubectl` for apply/diff (logic that lives in kubectl). | `agent.py`, `prompts.py` |
| For "list my pods" the local model set `all_namespaces=true`, so Kubernetes' own system pods (kube-system) filled the answer. | The argument descriptions say when to use `all_namespaces`/`include_system` (the model reads them when choosing arguments); and as a safety net, system namespaces are hidden from an all-namespaces list unless `include_system=true`, with a note "+N Kubernetes system pods hidden". | `k8s_tools.py` (`ListPodsArgs`, `SYSTEM_NAMESPACES`) |
