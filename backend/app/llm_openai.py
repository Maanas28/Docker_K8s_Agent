"""Client for the free cloud providers, which all speak the OpenAI chat-completions API.

Groq, Google Gemini (its OpenAI-compatible endpoint), OpenRouter and NVIDIA differ only in their
address, key and model ids, so one client covers them. It implements the same `LLM` protocol as
the Ollama client, so the agent doesn't know which one it talks to.

Every HTTP request (including retries and fallbacks) goes through the Meter, so the usage counter
sees exactly what the provider sees.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import httpx

from app.context import estimate_tokens
from app.llm import Classification, LLMUnavailable, ToolCall, UsageReport
from app.quota import Meter, QuotaExhausted, Reservation

# Google's documented placeholder for tool calls in the history that Gemini didn't make itself
# (e.g. made by the local model before you switched): Gemini 3 requires a "thought signature" on
# every earlier tool call and otherwise rejects the request.
GEMINI_PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class ProviderError(LLMUnavailable):
    """The provider answered with an error. `counted`: it may have counted against the limit."""

    def __init__(self, message: str, status: int, counted: bool = True) -> None:
        super().__init__(message)
        self.status = status
        self.counted = counted


class RateLimited(Exception):
    def __init__(self, retry_after: float | None, daily: bool) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.daily = daily


class ServerBusy(Exception):
    """HTTP 502/503/504: the provider is overloaded or restarting. Usually gone within seconds."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Waits before retrying a busy server: a short one, then a longer one.
_BUSY_WAITS = (3, 8)


class _ThinkFilter:
    """Drops <think>…</think> reasoning from streamed text (some open models put it in the reply)."""

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buffer += text
        out = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find("</think>")
                if end < 0:
                    self.buffer = self.buffer[-8:]  # keep a possible partial "</think>"
                    break
                self.buffer = self.buffer[end + 8:].lstrip()
                self.inside = False
            else:
                start = self.buffer.find("<think>")
                if start < 0:
                    # Hold back a possible partial "<think>" at the end.
                    keep = next((k for k in range(min(6, len(self.buffer)), 0, -1)
                                 if "<think>".startswith(self.buffer[-k:])), 0)
                    out.append(self.buffer[: len(self.buffer) - keep])
                    self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + 7:]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest, self.buffer = ("" if self.inside else self.buffer), ""
        return rest


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _provider_message(body: str) -> str:
    """The provider's own explanation from an error body (JSON `error.message` or plain text)."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:300]
    if isinstance(data, list) and data and isinstance(data[0], dict):  # Gemini can wrap errors in a list
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        error = error.get("message") or error.get("detail") or json.dumps(error)
    return str(error).strip()[:300]


_DAILY_WORDS = re.compile(r"per[ _-]?day|daily|requests per day|\bRPD\b|\bTPD\b|tokens per day", re.I)


class OpenAICompatLLM:
    def __init__(self, *, label: str, base_url: str, api_key: str, model: str, meter: Meter | None,
                 http: httpx.AsyncClient, thought_signatures: bool = False, max_output_tokens: int = 4096,
                 max_wait_s: float = 90, timeout_s: float = 180) -> None:
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.meter = meter
        self.http = http
        self.thought_signatures = thought_signatures
        self.max_output_tokens = max_output_tokens
        self.max_wait_s = max_wait_s
        self.timeout_s = timeout_s
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # ---- message format ------------------------------------------------------------------------

    def to_openai(self, messages: list[dict]) -> list[dict]:
        """Our history (Ollama style) → OpenAI style: tool calls need ids that tool results refer to."""
        out: list[dict] = []
        pending: list[str] = []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                pending = []
                if m.get("tool_calls"):
                    calls = []
                    for k, tc in enumerate(m["tool_calls"]):
                        call_id = tc.get("id") or f"call_{len(out)}_{k}"
                        entry: dict[str, Any] = {
                            "id": call_id, "type": "function",
                            "function": {"name": tc["function"]["name"],
                                         "arguments": json.dumps(tc["function"].get("arguments") or {})},
                        }
                        if self.thought_signatures:
                            entry["extra_content"] = tc.get("extra_content") or {
                                "google": {"thought_signature": GEMINI_PLACEHOLDER_SIGNATURE}
                            }
                        calls.append(entry)
                        pending.append(call_id)
                    msg["tool_calls"] = calls
                    msg["content"] = msg["content"] or None
                out.append(msg)
            elif role == "tool":
                if pending:
                    out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m.get("content", "")})
                else:  # its call fell out of the window: keep the information as plain text
                    out.append({"role": "user", "content": f"(Earlier tool output)\n{m.get('content', '')}"})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    # ---- one metered HTTP request ---------------------------------------------------------------

    async def _acquire(self, payload: dict) -> Reservation | None:
        if self.meter is None:
            return None
        est = estimate_tokens(payload.get("messages")) + estimate_tokens(payload.get("tools") or []) + 500
        return await self.meter.acquire(est)

    async def _release(self, res: Reservation | None, *, counted: bool, tokens: int | None = None,
                       headers: httpx.Headers | None = None) -> None:
        if self.meter is not None and res is not None:
            await self.meter.release(res, counted=counted, tokens=tokens, headers=dict(headers or {}))

    async def _error(self, response: httpx.Response) -> Exception:
        body = (await response.aread()).decode("utf-8", errors="replace")
        detail = _provider_message(body)
        status = response.status_code
        if status == 429:
            return RateLimited(_retry_after(response), daily=bool(_DAILY_WORDS.search(body)))
        if status in (502, 503, 504):
            return ServerBusy(status, detail)
        said = f" {self.label} said: {detail}" if detail else ""
        if status in (401, 403):
            return ProviderError(f"{self.label} rejected the API key (HTTP {status}). Check it in backend/.env.{said}", status)
        if status == 404:
            return ProviderError(f"{self.label} can't find the model '{self.model}' (HTTP 404).{said}", status)
        if status == 400:
            return ProviderError(f"{self.label} rejected the request (HTTP 400).{said}", status)
        return ProviderError(f"{self.label} had a problem (HTTP {status}).{said} Try again, or pick another model.", status)

    def _unreachable(self, error: httpx.HTTPError, answered: bool) -> LLMUnavailable:
        """httpx errors often have no text at all, so say what kind of failure it was."""
        if isinstance(error, httpx.TimeoutException):
            return LLMUnavailable(
                f"{self.label} didn't answer within {self.timeout_s:.0f} s (free endpoints can be busy, and large "
                "models are slow to start). Try again later, or pick another model."
            )
        what = str(error) or type(error).__name__
        if answered:
            return LLMUnavailable(f"Lost the connection to {self.label} while it was answering ({what}).")
        return LLMUnavailable(f"Can't reach {self.label} ({what}). Check your internet connection.")

    async def _retry(self, error: Exception, attempt: int) -> None:
        """Wait before another attempt (with a countdown), or give up by raising."""
        if isinstance(error, RateLimited):
            await self._rate_limited(error, attempt)
            return
        assert isinstance(error, ServerBusy)
        if attempt >= len(_BUSY_WAITS) or self.meter is None:
            raise ProviderError(
                f"{self.label} is overloaded or unavailable (HTTP {error.status}, {len(_BUSY_WAITS) + 1} tries)."
                + (f" It said: {error.detail}" if error.detail else "") + " Try again later, or pick another model.",
                error.status,
            )
        await self.meter.wait(_BUSY_WAITS[attempt], f"{self.label} being busy (HTTP {error.status})")

    async def _rate_limited(self, error: RateLimited, attempt: int) -> None:
        """Wait out a per-minute 429 (with a countdown), or stop on a daily one."""
        if error.daily or (error.retry_after or 0) > self.max_wait_s:
            if self.meter is not None:
                raise await self.meter.exhausted_for_today()
            raise QuotaExhausted(f"{self.label} refused more requests today.")
        if attempt == 2 or self.meter is None:
            raise LLMUnavailable(f"{self.label} keeps refusing requests (rate limited). Wait a minute or pick another model.")
        await self.meter.wait(error.retry_after or 10, f"{self.label}'s rate limit")

    async def _with_retries(self, send):
        for attempt in range(3):
            try:
                return await send()
            except (RateLimited, ServerBusy) as e:
                await self._retry(e, attempt)
        raise AssertionError("unreachable")

    # ---- LLM protocol ------------------------------------------------------------------------

    async def classify(self, messages: list[dict]) -> Classification:
        schema = Classification.model_json_schema()
        formats = [
            {"type": "json_schema", "json_schema": {"name": "classification", "schema": schema}},
            {"type": "json_object"},
            None,
        ]
        hint = {
            "role": "system",
            "content": "Reply with only a JSON object with the keys reason (string), live_system "
                       "(boolean), platform (docker|kubernetes|project|other) and operation (read|change).",
        }
        last: Exception | None = None
        for response_format in formats:  # not every provider/model supports every JSON mode
            payload: dict[str, Any] = {
                "model": self.model, "messages": self.to_openai([*messages, hint]),
                "temperature": 0, "max_tokens": 512,
            }
            if response_format:
                payload["response_format"] = response_format
            try:
                text = await self._with_retries(lambda: self._complete(payload))
            except ProviderError as e:
                if e.status != 400:
                    raise
                last = e
                continue
            match = re.search(r"\{.*\}", _THINK.sub("", text), re.S)
            try:
                return Classification.model_validate_json(match.group(0) if match else text)
            except ValueError as e:
                last = e
        raise last or ValueError("classification failed")

    async def _complete(self, payload: dict) -> str:
        res = await self._acquire(payload)
        try:
            response = await self.http.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers)
        except httpx.HTTPError as e:
            await self._release(res, counted=False)
            raise self._unreachable(e, answered=False) from e
        if response.status_code != 200:
            error = await self._error(response)
            # A 429 isn't counted by the provider; any other answered error might be, so we count it.
            await self._release(res, counted=not isinstance(error, RateLimited), headers=response.headers)
            raise error
        data = response.json()
        usage = data.get("usage") or {}
        await self._release(res, counted=True, tokens=usage.get("total_tokens"), headers=response.headers)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""

    async def stream(self, messages: list[dict], tools: list[dict] | None) -> AsyncIterator[str | ToolCall | UsageReport]:
        """Text is passed on as it streams; tool calls and the usage report come at the end."""
        payload: dict[str, Any] = {
            "model": self.model, "messages": self.to_openai(messages), "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.2,
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
        for attempt in range(3):
            res = await self._acquire(payload)
            answered = False  # did the provider start answering? Then it counts the request.
            try:
                async with self.http.stream("POST", f"{self.base_url}/chat/completions", json=payload,
                                            headers=self._headers) as response:
                    if response.status_code != 200:
                        error = await self._error(response)
                        await self._release(res, counted=not isinstance(error, RateLimited), headers=response.headers)
                        res = None
                        if isinstance(error, (RateLimited, ServerBusy)):
                            await self._retry(error, attempt)
                            continue
                        raise error
                    answered = True
                    usage: dict = {}
                    async for piece in self._parse(response, usage):
                        yield piece
                    total = usage.get("total_tokens")
                    await self._release(res, counted=True, tokens=total, headers=response.headers)
                    res = None
                    if total is not None:
                        yield UsageReport(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    return
            except httpx.HTTPError as e:
                if res is not None:
                    await self._release(res, counted=answered)
                    res = None
                raise self._unreachable(e, answered) from e
            finally:
                if res is not None:
                    # The reader stopped early (the user pressed Stop): the request was sent and
                    # the provider counts it, so we do too (with the estimated size).
                    await self._release(res, counted=True)

    async def _parse(self, response: httpx.Response, usage: dict) -> AsyncIterator[str | ToolCall]:
        think = _ThinkFilter()
        calls: dict[int, dict[str, Any]] = {}
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage.update(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text = think.feed(delta["content"])
                    if text:
                        yield text
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", len(calls)),
                                            {"id": None, "name": "", "arguments": "", "extra": None})
                    slot["id"] = slot["id"] or tc.get("id")
                    fn = tc.get("function") or {}
                    slot["name"] = slot["name"] or fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                    slot["extra"] = tc.get("extra_content") or slot["extra"]
        rest = think.flush()
        if rest:
            yield rest
        for _, slot in sorted(calls.items()):
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except ValueError:
                # Passed on so validation explains the problem to the model instead of crashing.
                arguments = {"invalid_json_arguments": slot["arguments"][:200]}
            yield ToolCall(slot["name"], arguments if isinstance(arguments, dict) else {},
                           id=slot["id"], extra_content=slot["extra"])
