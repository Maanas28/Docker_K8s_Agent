from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import Agent
from app.approvals import ApprovalBroker
from app.config import get_settings
from app.llm import LLMUnavailable, OllamaLLM
from app.models import ModelHub
from app.store import Store
from app.tools import build_tools, code_tools, docker_tools, run_tools
from app.tools import load_all
from app.tools.registry import registry
from app.workspace import safe_project_path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_all()
settings = get_settings()
store = Store(settings.db_url)
llm = OllamaLLM(settings.ollama_host, settings.ollama_model, settings.ollama_num_ctx, settings.ollama_think)
hub = ModelHub(settings, store, local_llm=llm)
code_tools.configure(store, build_facts=settings.context.build_facts)
build_tools.configure(settings.build)
run_tools.configure(settings.run)

app = FastAPI(title="Docker/K8s Agent")


def _origin_allowed(origin: str | None) -> bool:
    return origin is None or origin in settings.allowed_origins


@app.middleware("http")
async def check_origin(request: Request, call_next):
    if not _origin_allowed(request.headers.get("origin")):
        return JSONResponse({"detail": "Requests from this website aren't allowed."}, status_code=403)
    return await call_next(request)


@app.get("/api/health")
async def health() -> dict:
    return {
        **await llm.status(),
        "model": settings.ollama_model,
        "docker": await asyncio.to_thread(docker_tools.ping),
    }


@app.get("/api/tools")
def list_tools() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "risk": t.risk.value,
         "always_allowed": store.is_always_allowed(t.name), "per_project": t.scope_field is not None}
        for t in registry.all()
    ]


@app.get("/api/models")
def list_models() -> list[dict]:
    return hub.statuses()


@app.get("/api/permissions")
def list_permissions() -> list[dict]:
    return [{"tool": p.tool, "created_at": p.created_at} for p in store.always_allowed()]


@app.delete("/api/permissions/{tool}")
def revoke_permission(tool: str) -> dict:
    if not store.revoke(tool):
        raise HTTPException(404, f"'{tool}' is not always-allowed")
    return {"revoked": tool}


# ---- projects (registered only through the UI; the agent has no tool to add one) ----------------


class NewProject(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    name: str | None = Field(None, max_length=64)


_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._")[:64]


def _project_json(p) -> dict:
    return {
        "name": p.name, "path": p.path, "source": p.source, "created_at": p.created_at,
        "exists": Path(p.path).is_dir(), "reads_always_allowed": store.project_reads_allowed(p.name),
    }


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return [_project_json(p) for p in store.projects()]


@app.post("/api/projects", status_code=201)
def add_project(body: NewProject) -> dict:
    try:
        path = safe_project_path(body.path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    name = _slug(body.name) if body.name else _slug(path.name)
    if not _NAME.match(name):
        raise HTTPException(400, "Use a name with letters, numbers, '.', '_' or '-'.")
    if store.get_project(name):
        raise HTTPException(409, f"A project called '{name}' already exists.")
    existing = store.project_by_path(str(path))
    if existing:
        raise HTTPException(409, f"That folder is already registered as '{existing.name}'.")
    return _project_json(store.add_project(name, str(path)))


@app.delete("/api/projects/{name}")
def remove_project(name: str) -> dict:
    if not store.remove_project(name):
        raise HTTPException(404, f"No project called '{name}'")
    return {"removed": name}


@app.delete("/api/projects/{name}/read-grant")
def revoke_project_reads(name: str) -> dict:
    if not store.revoke_project_reads(name):
        raise HTTPException(404, f"Reading '{name}' isn't always-allowed")
    return {"revoked": name}


@app.get("/api/audit")
def audit(limit: int = 50) -> list[dict]:
    return [e.model_dump() for e in store.recent_audit(min(limit, 500))]


@app.websocket("/ws/chat")
async def chat(ws: WebSocket) -> None:
    if not _origin_allowed(ws.headers.get("origin")):
        # Another website open in the browser must not be able to drive the agent.
        await ws.close(code=1008)
        return
    await ws.accept()
    send_lock = asyncio.Lock()

    async def emit(event: dict) -> None:
        async with send_lock:
            try:
                await ws.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                pass  # client went away; the finally block below cleans up

    broker = ApprovalBroker(settings.approval_timeout_s)
    agent = Agent(
        llm=llm, registry=registry, store=store, broker=broker, emit=emit,
        session_id=uuid4().hex, settings=settings, hub=hub,
    )
    await agent.emit_models()
    task: asyncio.Task | None = None

    async def run(text: str) -> None:
        try:
            await agent.handle(text)
        except LLMUnavailable as e:
            await emit({"type": "error", "message": str(e)})
        except Exception as e:
            log.exception("agent failed")
            await emit({"type": "error", "message": f"Something went wrong: {e}"})
        finally:
            await emit({"type": "done"})

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "user_message":
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue
                if task and not task.done():
                    await emit({"type": "error", "message": "Still working on the previous message."})
                    continue
                task = asyncio.create_task(run(text))
            elif msg.get("type") == "stop":
                # Cooperative: the agent finishes the current step and starts nothing new.
                if task and not task.done():
                    agent.request_stop()
            elif msg.get("type") == "set_model":
                if task and not task.done():
                    await emit({"type": "error", "message": "Can't switch models while I'm working. Wait or press Stop first."})
                    await agent.emit_models()
                    continue
                error = await agent.set_model(str(msg.get("id")))
                if error:
                    await emit({"type": "error", "message": error})
            elif msg.get("type") == "revoke_session_grant":
                await agent.revoke_session_grant(str(msg.get("group")), str(msg.get("scope")))
            elif msg.get("type") == "approval":
                error = broker.resolve(str(msg.get("id")), str(msg.get("decision")), msg.get("confirm_text"))
                if error:
                    await emit({"type": "approval_error", "id": msg.get("id"), "message": error})
    except WebSocketDisconnect:
        pass
    finally:
        broker.reject_all()  # a closed tab never counts as approval
        if task and not task.done():
            task.cancel()
