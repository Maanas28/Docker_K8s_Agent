# Handoff: where the project stands (2026-10-03)

Read this first when starting a new chat. The README covers how things work; this file covers
where we are, what was decided, and what's next.

## Working agreement
- **Mentoring format** for every coding task: before code, explain the problem, why this approach
  (alternatives and trade-offs), which files change and the flow; then the code; then line-by-line,
  end-to-end flow, production notes (edge cases, performance, security, scalability) and interview
  questions with answers. Explain new concepts before using them.
- **Never start servers** (backend, frontend, Ollama). Give the commands; the user runs them.
  Running `uv run pytest` is fine.
- **New work waits for an explicit "go".** Reported bugs: propose the fix and ask.
- **Generic fixes over one rule per LLM mistake.**
- **Experimental behaviour goes behind switches in `backend/agent.toml`**, documented there,
  including rejected options and why.
- Don't permanently delete user files: look first, then move to Trash.

## Current state
- Backend: 266 tests pass (`cd backend && uv run pytest`). Frontend builds (`npm run build`).
- Model: `qwen3:8b`, thinking off, 32k context (`backend/app/config.py`).
- Done: chat with three message types; approval gate (reads can be always-allowed, per project for
  code; changes always ask; destructive needs the name typed); Docker read + actions with
  pre-checks; registered projects with sandboxed read access and secret protection;
  `project_context` (overview + key files in one call); writing `Dockerfile`/`.dockerignore` into
  the repo with rule checks and a diff card; error budget (10 per request) instead of a step
  limit; Stop button (cooperative); repeat guard; text tool-call recovery; context-management
  flags (compact output, replace outdated outputs, attempt log, relevant tools only, measure and
  compact).
- Step 3 done (2026-10-03, not yet tried against the real model and Docker): `project_build_image`
  runs the real `docker build` (BuildKit, not the SDK; see docs/LIMITATIONS.md) with live output,
  Stop and a timeout; the build-and-fix loop is the normal agent loop; identical rebuilds after a
  failure are refused; secret files in the build context are refused. Also done: Fix 4 (prompt
  rule: never offer actions without a tool) and idea A (build facts in `project_context`, flag
  `[context] build_facts`). Async tools + `RunContext` in the registry.
- First real run (2026-10-03) failed after 10 errors (uv.lock excluded by .dockerignore, never
  understood). Fixed the same day: "not found" diagnosis, repeat guard reset after a successful
  change, `no_cache` skips the unchanged-files guard only once, failures announced in the chat +
  required `reason` on file writes + build attempt numbers. Not yet re-tried against the model.
- Second run: stuck on installing uv (model knowledge). Fixed our side: exit 127 / apt
  explanations, same-error brake and failure nudge (`[loop]` in agent.toml), literal `\n` check,
  prompt fix. Open: whether to add install facts (idea B), try thinking on, or a cloud model
  (Ollama Cloud, e.g. `kimi-k2.5:cloud`). "Allow for this session" is done: one button
  covers writing Docker files AND building (group "build-loop"), for that one project, in memory
  until the chat ends, revocable from a bar above the input; switch `[approvals] session_allow`.
- Model dropdown (2026-10-05): local model (default) + free cloud models (Groq, Gemini, OpenRouter
  `:free`, NVIDIA) through one OpenAI-compatible client; precise usage counter (reserve/settle,
  SQLite per provider day, Groq headers win, 2-call reserve stops the chat and greys the model
  out, per-minute limits waited out). User decisions: free only (no Mistral, no Ollama Cloud), no
  privacy prompt, every chat starts local. Keys go in backend/.env (user adds them). NOT yet run
  against real providers: run `uv run python -m scripts.check_models` first; model ids checked
  2026-10-05 (Gemini 3.8/2.5 Flash, OpenRouter qwen3.8-27b:free + nemotron-3-super:free, NVIDIA
  z-ai/glm-5.3 + moonshotai/kimi-k3); Groq left out for now (login problem), its Qwen id unverified;
  Gemini placeholder signature still to verify.
- First cloud run (2026-10-05): OpenRouter Qwen3.8 leaked tool-call markup into both files
  (sample-fastapi-app Dockerfile line 25 and the end of .dockerignore still contain it; the agent
  should rewrite them). Fixed: markup check on writes, Dockerfile instruction check, stop message
  rendering, clearer cloud errors, 5xx retries, 180 s timeout. Gemini 2.5 Flash 404 unexplained.
- **Step 3 verified (2026-10-05):** the user built the sample-project image successfully through
  the agent after the fixes above. Next: Step 4 (run the container + health check).
- **Step 4 done and verified by the user (2026-10-06):** `project_run_container` (in the "build-loop"
  session group): image <project>:latest, port from EXPOSE on 127.0.0.1, `.env` passed (names
  only shown, values masked in logs), replaces only its own labelled container, HTTP health check
  with live logs and "Why:" facts. User chose: pass .env yes, 127.0.0.1 yes, auto-replace yes.
- **Step 5 done (2026-10-08, not yet run for real):** unique `v<N>-<commit>` tags + `ImageVersion`
  table; project_run_container gained `version` ("previous" = rollback) and the safe switch (trial
  port, then real port, old version restored on failure); new tools project_versions (read) and
  project_prune_images (destructive, keeps newest + in use + last 3 healthy). User chose all three
  recommendations.
- Health check fixed (2026-10-09): 5xx = unhealthy (no fall-through), 404 = next path, explicit
  interval/timeout/failure threshold. sample-project v3/v4 marked failed (they contain a test
  RuntimeError in /health); the junk lines were removed from its .dockerignore.
- **docs/LIMITATIONS.md** records every workaround forced by a limitation. Keep it updated.
- UI: each finished step is a small collapsed row; approval cards stay open until decided.

## Test project
`~/Projects/ops-agent-sandbox/sample-fastapi-app`, registered in the app as **sample-project**.
FastAPI, uv (`pyproject.toml` + `uv.lock`, no `requirements.txt`), starts with `python -m app`,
port from `APP_PORT` (default 8080), fake secrets in `.env`.

It currently contains an agent-written `Dockerfile` and `.dockerignore` (uncommitted). **That
Dockerfile won't build**: it runs `pip install -r requirements.txt`, and the file doesn't exist;
the .dockerignore also excludes `*.py` and `uv.lock` and doesn't exclude `.venv`. It's the test
case for the Step 3 fix loop.

## Known issues / open decisions
- Dockerfile *content* from the 8B model is often wrong (wrong install command, paths, port).
  Agreed approach: keep iterating, judged by a real build (Step 3), with context management
  keeping iterations affordable.
- Ideas raised but not chosen: (B) a reference example per package manager in the prompt; (C)
  thinking on only for file writes; (D) check that files referenced in a Dockerfile exist (mostly
  covered by the real build now).

## Next
- (Optional) Count fix-loop attempts with `build_facts` on and off, per model.
- Try Step 5 for real (redeploy, roll back, versions, clean up). Then Phase 3: Kubernetes read-only.
