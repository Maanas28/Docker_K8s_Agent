"""Check the real model: `uv run python -m scripts.eval_routing`.

Two parts:
1. Routing: is each message classified as action / question / unsupported correctly?
2. Tool calling: for action messages, does the model call the right tool with the right
   arguments, and does it use the tool-calling API ("native") or only write the call as text?

Re-run this after changing prompts or switching models. Nothing is executed: the check stops at
the model's proposed call, before approval.
"""

import asyncio
import sys

from app.agent import CLASSIFY_FALLBACK_REASON, Agent
from app.config import Settings
from app.llm import LLMUnavailable, OllamaLLM
from app.store import Store
from app.tools import load_all
from app.tools.registry import registry

load_all()
settings = Settings()
llm = OllamaLLM(settings.ollama_host, settings.ollama_model, settings.ollama_num_ctx, settings.ollama_think)

# A throwaway in-memory database with one project, so results don't depend on your real setup.
store = Store("sqlite://")
store.add_project("sample-project", "/tmp/ops-agent-eval/sample-fastapi-app")

ROUTING = [
    ("restart the nginx container", "action"), ("stop all containers", "action"),
    ("delete the old postgres container", "action"), ("pull the latest redis image", "action"),
    ("how much memory is the api container using?", "action"), ("remove the mongo image", "action"),
    ("scale the web deployment to 3 replicas", "unsupported"),
    ("list the pods in my default namespace", "action"), ("show my running containers", "action"),
    ("what is a pod?", "question"), ("show logs of the api container", "action"),
    ("is my postgres container healthy?", "action"), ("which containers exited with an error?", "action"),
    ("hi, what can you do?", "question"), ("delete my AWS account", "unsupported"),
    ("how do I write a Dockerfile for a node app?", "question"), ("what images do I have locally?", "action"),
    ("what does sample-project do?", "action"), ("which port does sample-project listen on?", "action"),
    ("what projects do I have?", "action"), ("tell me about my sample project", "action"),
    ("write a Dockerfile for sample-project", "action"),
    ("create a .dockerignore for my sample project", "action"),
    ("how should I structure a FastAPI project?", "question"),
    ("build the docker image for sample-project", "action"),
    ("run sample-project in a container", "action"),
    ("roll back sample-project", "action"),
    ("which versions of sample-project do I have?", "action"),
    ("why is my pod crashing in kubernetes?", "action"),
    ("restart the web deployment in kubernetes", "unsupported"),
]

# (message, expected tool (or a tuple of acceptable ones for vague requests), arguments that must
# match; other arguments may be anything)
TOOL_CALLS = [
    ("show my running containers", "docker_list_containers", {}),
    ("show the last 50 log lines of the api-gateway container", "docker_container_logs",
     {"container": "api-gateway", "tail": 50}),
    ("restart the nginx container", "docker_restart_container", {"container": "nginx"}),
    ("how much memory is the redis container using?", "docker_container_stats", {"container": "redis"}),
    ("what projects do I have?", "project_list", {}),
    ("give me an overview of sample-project", "project_overview", {"project": "sample-project"}),
    # Vague on purpose: tests that "my sample project" resolves to the registered name.
    ("tell me about my sample project", ("project_overview", "project_list_files"), {"project": "sample-project"}),
    ("read app/main.py in sample-project", "project_read_file", {"project": "sample-project", "path": "app/main.py"}),
    ("search sample-project for APP_PORT", "project_search", {"project": "sample-project", "pattern": "APP_PORT"}),
    # Writing: the first step should be reading the project (or writing, if it has enough to go on).
    ("write a Dockerfile for sample-project",
     ("project_context", "project_overview", "project_list_files", "project_read_file", "project_write_file"),
     {"project": "sample-project"}),
    ("what does sample-project need to run? check its dependencies and how it starts",
     ("project_context", "project_overview"), {"project": "sample-project"}),
    ("write a .dockerignore for sample-project that excludes .env, .git and .venv",
     ("project_write_file", "project_context", "project_overview", "project_list_files"), {"project": "sample-project"}),
    # Building: build straight away, or look at the project first.
    ("build the docker image for sample-project", ("project_build_image", "project_context", "project_overview"),
     {"project": "sample-project"}),
    ("roll back sample-project to the previous version", ("project_run_container", "project_versions"),
     {"project": "sample-project"}),
    ("what versions of sample-project do I have?", "project_versions", {"project": "sample-project"}),
    ("list the pods in the kube-system namespace", "k8s_list_pods", {"namespace": "kube-system"}),
    ("show warning events in kubernetes", "k8s_events", {}),
]


async def _quiet(event: dict) -> None:
    pass


def _agent() -> Agent:
    return Agent(llm=llm, registry=registry, store=store, broker=None, emit=_quiet, session_id="eval", settings=settings)


async def preflight() -> None:
    """Stop before scoring if the model can't be used; otherwise every case silently falls back."""
    status = await llm.status()
    if not status["ollama"]:
        sys.exit(f"Can't reach Ollama at {settings.ollama_host}. Start it with `ollama serve`.")
    if not status["model_available"]:
        sys.exit(f"The model {settings.ollama_model} isn't downloaded. Run `ollama pull {settings.ollama_model}`.")


async def routing() -> tuple[int, int]:
    print("== Routing ==")
    a = _agent()
    correct = 0
    for msg, want in ROUTING:
        a.history = [{"role": "user", "content": msg}]
        kind, reason = await a._classify()
        if reason == CLASSIFY_FALLBACK_REASON:
            # The model's answer couldn't be parsed; that's a failure, not a lucky "action".
            print(f"ERR {'-':12} (model returned an invalid answer) | {msg}")
            continue
        correct += kind == want
        print(f"{'OK ' if kind == want else 'BAD'} {kind:12} ({reason[:60]}) | {msg}")
    return correct, len(ROUTING)


def _matches(expected: dict, actual: dict) -> bool:
    return all(str(actual.get(k, "")).strip().lower() == str(v).lower() for k, v in expected.items())


async def tool_calls() -> tuple[int, int, int]:
    print("\n== Tool calling ==")
    a = _agent()
    system = a._action_system_prompt()
    native = recovered = 0
    for msg, tool, args in TOOL_CALLS:
        a.history = [{"role": "user", "content": msg}]
        text, calls = await a._llm_turn(system, a._window(), registry.schemas())
        call = calls[0] if calls else None
        wanted = tool if isinstance(tool, tuple) else (tool,)
        if call and call.name in wanted and _matches(args, call.arguments):
            how = "text  " if call.from_text else "native"
            recovered += call.from_text
            native += not call.from_text
            print(f"OK  {how} {call.name}({call.arguments}) | {msg}")
        elif call:
            print(f"BAD        {call.name}({call.arguments}), expected {' or '.join(wanted)}{args or ''} | {msg}")
        else:
            print(f"BAD        no tool call, replied: {text[:70]!r} | {msg}")
    return native, recovered, len(TOOL_CALLS)


async def main() -> None:
    await preflight()
    try:
        routed, total_routing = await routing()
        native, recovered, total_calls = await tool_calls()
    except LLMUnavailable as e:
        sys.exit(str(e))
    print(f"\n{settings.ollama_model}")
    print(f"  routing:      {routed}/{total_routing} correct")
    print(f"  tool calling: {native + recovered}/{total_calls} correct "
          f"({native} native, {recovered} recovered from text)")


asyncio.run(main())
