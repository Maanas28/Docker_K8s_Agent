from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import Classification, ToolCall
from app.store import Store
from app.tools.registry import Risk, ToolArgs, ToolError, ToolRegistry


class FakeLLM:
    """Scripted LLM: a fixed classification, then one scripted reply per `stream` call.

    `kind` is shorthand for a typical classification: "question" (no live system),
    "action" (a Docker read) or "change" (a Docker change).
    """

    def __init__(self, kind: str, turns: list[list[str | ToolCall]], **overrides) -> None:
        base = {
            "question": dict(live_system=False, platform="docker", operation="read"),
            "action": dict(live_system=True, platform="docker", operation="read"),
            "change": dict(live_system=True, platform="docker", operation="change"),
        }[kind]
        self.classification = Classification(reason="because", **{**base, **overrides})
        self.turns = list(turns)
        self.stream_calls: list[dict] = []

    async def classify(self, messages):
        return self.classification

    async def stream(self, messages, tools):
        self.stream_calls.append({"messages": messages, "tools": tools})
        # Out of scripted turns: an empty reply (e.g. the answer to a hidden nudge).
        for piece in self.turns.pop(0) if self.turns else []:
            yield piece


class NameArgs(ToolArgs):
    name: str


class ScaleArgs(ToolArgs):
    name: str
    replicas: int


@pytest.fixture
def executed() -> list[tuple[str, dict]]:
    return []


@pytest.fixture
def registry(executed) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(risk=Risk.READ, summary=lambda a: f"Read {a.name}", platform="docker")
    def read_thing(args: NameArgs) -> dict:
        """Read a thing."""
        executed.append(("read_thing", args.model_dump()))
        if args.name.startswith("missing"):
            raise ToolError(f"No thing named '{args.name}'")
        return {"kind": "json", "title": args.name, "data": {"ok": True}}

    @reg.tool(risk=Risk.CHANGE, summary=lambda a: f"Scale {a.name} to {a.replicas}", platform="docker")
    def scale_thing(args: ScaleArgs) -> dict:
        """Scale a thing."""
        executed.append(("scale_thing", args.model_dump()))
        return {"kind": "json", "title": args.name, "data": {"replicas": args.replicas}}

    @reg.tool(risk=Risk.DESTRUCTIVE, summary=lambda a: f"Delete {a.name}", confirm_field="name", platform="docker")
    def delete_thing(args: NameArgs) -> dict:
        """Delete a thing."""
        executed.append(("delete_thing", args.model_dump()))
        return {"kind": "json", "title": args.name, "data": {"deleted": True}}

    return reg


@pytest.fixture
def store() -> Store:
    return Store("sqlite://")


@pytest.fixture
def settings() -> Settings:
    return Settings(approval_timeout_s=2, max_errors=4)
