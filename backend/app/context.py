"""Keep what the model sees small and current (techniques 1, 2 and 6 in agent.toml).

Every function here is pure: it returns new data and never modifies the conversation history it
is given. The agent keeps the full history and only trims the copy it sends to the model, so
turning a flag off brings everything back on the next call.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.config import ContextSettings

CHARS_PER_TOKEN = 4  # rough, model-independent estimate


# ---- 1. compact tool output ---------------------------------------------------------------------


def compact_result(result: dict, cfg: ContextSettings) -> dict:
    """The part of a tool result worth sending to the model. The UI always gets the full result."""
    kind = result.get("kind")
    if kind in ("logs", "build", "container"):
        field = "text" if kind == "logs" else "log"
        lines = result.get(field, "").splitlines()
        if kind in ("build", "container") and result.get("ok"):
            # A successful build's or run's log is noise for the model; the facts say what happened.
            return {k: v for k, v in result.items() if k != "log"}
        if len(lines) > cfg.compact_log_lines:
            kept = lines[-cfg.compact_log_lines:]  # errors are usually at the end of a log
            omitted = len(lines) - len(kept)
            return {**result, field: f"[{omitted} earlier lines omitted; the user sees the full log]\n" + "\n".join(kept)}
    if kind == "table":
        rows = result.get("rows", [])
        if len(rows) > cfg.compact_table_rows:
            return {
                **result,
                "rows": rows[: cfg.compact_table_rows],
                "note": f"Showing {cfg.compact_table_rows} of {len(rows)} rows; the user sees all of them.",
            }
    return result


# ---- 2. replace outdated outputs ----------------------------------------------------------------


def call_keys(name: str, args: dict[str, Any]) -> set[str]:
    """What a tool call is 'about'. A newer call sharing a key makes the older one outdated."""
    keys = {f"call:{name}:{json.dumps(args, sort_keys=True, default=str)}"}  # the exact same call
    if "project" in args and "path" in args:
        keys.add(f"file:{args['project']}:{args['path']}")  # any read or write of the same file
    if name.endswith("_build_image") and "project" in args:
        keys.add(f"build:{args['project']}")  # only the latest build of a project matters
    if name.endswith("_run_container") and "project" in args:
        keys.add(f"run:{args['project']}")  # and only its latest run
    return keys


def _pairs(messages: list[dict]) -> list[tuple[int, int, int | None]]:
    """(assistant message index, call index, matching tool message index) for every tool call.

    Tool results follow their assistant message in the same order as its tool_calls.
    """
    pairs = []
    for i, m in enumerate(messages):
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        for k in range(len(calls or [])):
            j = i + 1 + k
            ok = j < len(messages) and messages[j].get("role") == "tool"
            pairs.append((i, k, j if ok else None))
    return pairs


def replace_outdated(messages: list[dict]) -> tuple[list[dict], int]:
    """Copy of `messages` where older copies of the same file/call are one-line notes.

    Returns (messages, how many calls were replaced).
    """
    out = list(messages)
    pairs = _pairs(messages)
    seen: set[str] = set()
    replaced = 0
    for i, k, j in reversed(pairs):  # newest first, so `seen` holds keys of NEWER calls only
        fn = messages[i]["tool_calls"][k]["function"]
        name, args = fn["name"], fn.get("arguments") or {}
        keys = call_keys(name, args)
        if keys & seen:
            target = args.get("path") or args.get("project") or name
            note = f"[outdated: an earlier {name} for {target}; a newer result is further down]"
            if j is not None:
                out[j] = {**messages[j], "content": note}
            if "content" in args:  # e.g. the full text of an old file-write attempt
                if out[i] is messages[i]:
                    out[i] = copy.deepcopy(messages[i])
                out[i]["tool_calls"][k]["function"]["arguments"] = {
                    **args, "content": "[content of an earlier attempt, replaced by a newer one]"
                }
            replaced += 1
        seen |= keys
    return out, replaced


# ---- 6. measure and compact ---------------------------------------------------------------------


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return len(text) // CHARS_PER_TOKEN + 1


def fit_budget(system: str, messages: list[dict], tools: list[dict] | None,
               num_ctx: int, ratio: float) -> tuple[list[dict], int]:
    """Shorten the OLDEST tool outputs until everything fits in `ratio` of the context window.

    User and system messages are never touched, so the request and instructions survive.
    Returns (messages, how many outputs were shortened).
    """
    budget = int(num_ctx * ratio)
    fixed = estimate_tokens(system) + (estimate_tokens(tools) if tools else 0)
    sizes = [estimate_tokens(m) for m in messages]
    if fixed + sum(sizes) <= budget:
        return messages, 0
    out = list(messages)
    shortened = 0
    for idx, m in enumerate(messages):  # oldest first
        if fixed + sum(sizes) <= budget:
            break
        if m.get("role") == "tool" and len(m.get("content", "")) > 200:
            out[idx] = {**m, "content": f"[older {m.get('tool_name', 'tool')} output removed to save space]"}
            sizes[idx] = estimate_tokens(out[idx])
            shortened += 1
    return out, shortened
