"""Recover tool calls that a model wrote as JSON text instead of using the tool-calling API.

Some models (qwen2.5-coder among them) reply with `{"name": "...", "arguments": {...}}` in
plain text. Such a call is only accepted if `name` is a registered tool; it then goes through
exactly the same validation, pre-checks and approval as a normal tool call.
"""

from __future__ import annotations

import json
from typing import Any

from app.llm import ToolCall
from app.tools.registry import ToolRegistry

# strict=False accepts raw line breaks inside strings, which models often emit when a tool
# argument is a whole file (e.g. a Dockerfile's content).
_DECODER = json.JSONDecoder(strict=False)


def _as_call(obj: Any, registry: ToolRegistry) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # {"function": {"name": ..., "arguments": ...}}
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}", strict=False)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or registry.get(name) is None or not isinstance(args, dict):
        return None
    return ToolCall(name, args, from_text=True)


def extract_text_tool_calls(text: str, registry: ToolRegistry) -> list[ToolCall]:
    """Tool calls written as JSON anywhere in `text` (bare, in ``` fences or <tool_call> tags)."""
    calls: list[ToolCall] = []
    i = 0
    while True:
        positions = [p for p in (text.find("{", i), text.find("[", i)) if p != -1]
        if not positions:
            return calls
        start = min(positions)
        try:
            obj, end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        candidates = [_as_call(item, registry) for item in (obj if isinstance(obj, list) else [obj])]
        found = [call for call in candidates if call is not None]
        if found:
            calls.extend(found)
            i = end
        else:
            i = start + 1
