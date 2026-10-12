"""Thin wrapper around Ollama. The agent only depends on the `LLM` protocol, so tests can fake it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Protocol

import httpx
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, Field


class LLMUnavailable(Exception):
    """The model server can't be reached; retrying through another path won't help."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    # True when the model wrote the call as JSON text instead of using the tool-calling API.
    from_text: bool = False
    # Provider data that must be sent back with this call on the next turn (cloud models only):
    # the call's id, and e.g. Gemini's "thought signature".
    id: str | None = None
    extra_content: dict[str, Any] | None = None


@dataclass
class UsageReport:
    """Token usage of one model request, as reported by the provider (cloud models only).

    Yielded as the last item of a stream; the agent ignores it, the usage counter uses it.
    """

    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Classification(BaseModel):
    """What the model reports about a message. The agent's code turns this into a route."""

    reason: str = Field(description="One short sentence about what the user wants")
    live_system: bool = Field(
        description="True if doing this requires looking at or changing the user's actual "
        "containers, images, pods or clusters"
    )
    platform: Literal["docker", "kubernetes", "project", "other"] = Field(
        description="Which system the request is about"
    )
    operation: Literal["read", "change"] = Field(
        description='"read" = list/show/logs/inspect/check status. "change" = start/stop/restart/'
        "delete/create/scale/update/pull or anything else that modifies something"
    )


class LLM(Protocol):
    async def classify(self, messages: list[dict]) -> Classification: ...

    def stream(self, messages: list[dict], tools: list[dict] | None) -> AsyncIterator[str | ToolCall]:
        """Yield text chunks and ToolCalls as the model produces them."""
        ...


# Errors meaning the model can't be used at all: Ollama is down, or the model isn't pulled (404).
_UNAVAILABLE = (ConnectionError, httpx.TransportError, ResponseError)


def _is_other_response_error(e: Exception) -> bool:
    """A ResponseError that isn't "model not found" is a real bug, so it shouldn't be hidden."""
    return isinstance(e, ResponseError) and e.status_code != 404


def _pieces(chunk: Any) -> list[str | ToolCall]:
    out: list[str | ToolCall] = []
    if chunk.message.content:
        out.append(chunk.message.content)
    for call in chunk.message.tool_calls or []:
        args = call.function.arguments
        if isinstance(args, str):  # some models return a JSON string
            args = json.loads(args or "{}")
        out.append(ToolCall(call.function.name, dict(args or {})))
    return out


class OllamaLLM:
    def __init__(self, host: str, model: str, num_ctx: int = 16384, think: bool | None = None) -> None:
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        # Only sent when set: models without a thinking mode reject the option.
        self._think = {"think": think} if think is not None else {}
        self._client = AsyncClient(host=host)

    def _unavailable(self, error: Exception) -> LLMUnavailable:
        if isinstance(error, ResponseError) and error.status_code == 404:
            return LLMUnavailable(
                f"The model {self.model} isn't downloaded. Run `ollama pull {self.model}` and try again."
            )
        return LLMUnavailable(
            f"Can't reach Ollama at {self.host}. Start it with `ollama serve` (or open the Ollama app) and try again."
        )

    async def classify(self, messages: list[dict]) -> Classification:
        try:
            resp = await self._client.chat(
                model=self.model,
                messages=messages,
                format=Classification.model_json_schema(),
                options={"temperature": 0, "num_ctx": self.num_ctx},
                **self._think,
            )
        except _UNAVAILABLE as e:
            if _is_other_response_error(e):
                raise
            raise self._unavailable(e) from e
        return Classification.model_validate_json(resp.message.content or "")

    async def stream(self, messages: list[dict], tools: list[dict] | None) -> AsyncIterator[str | ToolCall]:
        try:
            chunks = await self._client.chat(
                model=self.model,
                messages=messages,
                tools=tools or None,
                stream=True,
                options={"temperature": 0.2, "num_ctx": self.num_ctx},
                **self._think,
            )
            async for chunk in chunks:
                for piece in _pieces(chunk):
                    yield piece
        except _UNAVAILABLE as e:
            if _is_other_response_error(e):
                raise
            raise self._unavailable(e) from e

    async def status(self) -> dict:
        """Whether Ollama is reachable and the configured model is pulled."""
        try:
            models = await self._client.list()
        except Exception:
            return {"ollama": False, "model_available": False}
        names = {m.model for m in models.models}
        available = self.model in names or f"{self.model}:latest" in names
        return {"ollama": True, "model_available": available}
