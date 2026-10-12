"""Tool registry: every operation the agent can perform is a typed, registered tool.

The LLM can only *propose* calls to tools listed here. Arguments are validated against
the tool's Pydantic model before anything else happens, and execution always goes
through the agent's approval gate.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict


class Risk(str, Enum):
    READ = "read"
    CHANGE = "change"
    DESTRUCTIVE = "destructive"


class ToolError(Exception):
    """An expected failure that is shown to the user (e.g. "container not found").

    `result` optionally carries a full result to show instead of just the message, e.g. the log
    of a failed build.
    """

    def __init__(self, message: str, result: dict | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class RunContext:
    """What a long-running tool gets while it runs (opt in with a `ctx` parameter).

    progress: send output lines to the user's step row as they happen.
    stopped:  set when the user presses Stop; the tool should cancel its work and raise ToolError.
    """

    progress: Callable[[str], Awaitable[None]]
    stopped: asyncio.Event = field(default_factory=asyncio.Event)


class ToolArgs(BaseModel):
    """Base class for tool arguments. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    args_model: type[ToolArgs]
    func: Callable[[Any], dict]
    summarize: Callable[[Any], str]
    platform: str
    # For destructive tools: the argument whose value the user must type to confirm.
    confirm_field: str | None = None
    # Optional read-only check that runs before the approval card is shown. It returns
    # {"facts": {label: value}, "warnings": [...]} describing what will be affected, or
    # raises ToolError if the action can't work (e.g. "already running"), in which case
    # no card is shown at all.
    preview: Callable[[Any], dict] | None = None
    # Arguments whose value must have appeared in something the user typed or a tool returned,
    # so the model can't act on a name it made up (e.g. an image name for a pull).
    grounded: tuple[str, ...] = ()
    # For read tools that act inside a project: the argument naming the project. "Always allow"
    # then applies to that project only, instead of to the tool everywhere.
    scope_field: str | None = None
    # Change tools that may be "allowed for this session" together, e.g. "build-loop" for writing the
    # Dockerfile and building it: one click covers every tool in the group, for one project
    # (`session_scope_field`), until the chat session ends. Never for destructive tools.
    session_group: str | None = None
    session_scope_field: str | None = None
    # True if `func` takes a second parameter: the preview result the user approved. Lets a tool
    # check at run time that nothing changed since the user saw the card (e.g. a file's diff).
    uses_preview: bool = False
    # True if `func` takes a `ctx` parameter (a RunContext), for live progress and Stop.
    uses_ctx: bool = False
    # True for `async def` tools: they run on the event loop instead of in a worker thread.
    is_async: bool = False

    def _kwargs(self, preview: dict | None, ctx: RunContext | None) -> dict:
        kwargs: dict[str, Any] = {}
        if self.uses_preview:
            kwargs["preview"] = preview
        if self.uses_ctx:
            kwargs["ctx"] = ctx
        return kwargs

    def run(self, args: ToolArgs, preview: dict | None, ctx: RunContext | None = None) -> dict:
        """Run a synchronous tool (blocking)."""
        return self.func(args, **self._kwargs(preview, ctx))

    async def call(self, args: ToolArgs, preview: dict | None, ctx: RunContext | None = None) -> dict:
        """Run any tool from async code: async tools are awaited, sync ones go to a thread so a
        slow Docker or file call never blocks the event loop (and with it every other chat)."""
        if self.is_async:
            return await self.func(args, **self._kwargs(preview, ctx))
        return await asyncio.to_thread(self.run, args, preview, ctx)

    def schema(self) -> dict:
        """Tool definition in the format Ollama's `tools` parameter expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args_model.model_json_schema()),
            },
        }

    def scope(self, args: ToolArgs) -> str | None:
        return str(getattr(args, self.scope_field)) if self.scope_field else None

    def session_key(self, args: ToolArgs) -> tuple[str, str] | None:
        """(group, scope) a session grant for this call would be stored under, or None if it can't be."""
        if self.session_group is None or self.session_scope_field is None:
            return None
        return self.session_group, str(getattr(args, self.session_scope_field))

    def confirm_text(self, args: ToolArgs) -> str | None:
        if self.confirm_field is None:
            return None
        return str(getattr(args, self.confirm_field))


def _strip_titles(schema: Any) -> Any:
    """Drop Pydantic's auto-generated "title" keys; they only add noise to the prompt."""
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v)
            for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        if tool.risk is Risk.DESTRUCTIVE and tool.confirm_field is None:
            raise ValueError(f"Destructive tool {tool.name!r} needs a confirm_field")
        if tool.session_group is not None and tool.risk is not Risk.CHANGE:
            raise ValueError(f"Only change tools can be allowed for a session, not {tool.name!r}")
        self._tools[tool.name] = tool

    def tool(
        self,
        *,
        risk: Risk,
        summary: Callable[[Any], str],
        confirm_field: str | None = None,
        platform: str | None = None,
        preview: Callable[[Any], dict] | None = None,
        grounded: tuple[str, ...] = (),
        scope_field: str | None = None,
        session_group: str | None = None,
        session_scope_field: str | None = None,
    ) -> Callable[[Callable], Callable]:
        """Register a function as a tool.

        The function name becomes the tool name, its docstring the description, and
        the type of its first parameter (a ToolArgs subclass) the argument schema. Optional
        extra parameters, by name: `preview` receives the approved preview result, and `ctx`
        a RunContext (live progress, Stop). The function may be `async def`.
        The platform defaults to the name's prefix ("docker_list_images" -> "docker").
        """

        def decorator(func: Callable) -> Callable:
            params = list(inspect.signature(func).parameters.values())
            extras = [p.name for p in params[1:]]
            if not params or any(e not in ("preview", "ctx") for e in extras) or ("preview" in extras and preview is None):
                raise ValueError(
                    f"Tool {func.__name__!r} must take (args), plus optionally `preview` (needs a preview "
                    "function) and/or `ctx`"
                )
            args_model = get_type_hints(func)[params[0].name]
            self.register(
                Tool(
                    name=func.__name__,
                    description=inspect.getdoc(func) or func.__name__,
                    risk=risk,
                    args_model=args_model,
                    func=func,
                    summarize=summary,
                    platform=platform or func.__name__.split("_", 1)[0],
                    confirm_field=confirm_field,
                    preview=preview,
                    grounded=grounded,
                    scope_field=scope_field,
                    session_group=session_group,
                    session_scope_field=session_scope_field,
                    uses_preview="preview" in extras,
                    uses_ctx="ctx" in extras,
                    is_async=inspect.iscoroutinefunction(func),
                )
            )
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def supports(self, platform: str, operation: str) -> bool:
        """Is there any tool for this platform that can read (operation="read") or change things?"""
        wanted = {Risk.READ} if operation == "read" else {Risk.CHANGE, Risk.DESTRUCTIVE}
        return any(t.platform == platform and t.risk in wanted for t in self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


registry = ToolRegistry()
