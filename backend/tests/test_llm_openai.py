"""The cloud client (llm_openai.py) against a fake HTTP transport, with the real usage counter."""

import json

import httpx
import pytest

from app.config import PoolSpec, Settings
from app.llm import ToolCall, UsageReport
from app.llm_openai import GEMINI_PLACEHOLDER_SIGNATURE, OpenAICompatLLM, ProviderError
from app.quota import HeaderSpec, Meter, QuotaExhausted, QuotaTracker
from app.store import Store


def sse(*chunks, usage=None):
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def text(t):
    return {"choices": [{"delta": {"content": t}}]}


def tool_delta(index, *, id=None, name=None, args="", extra=None):
    tc = {"index": index, "function": {"arguments": args}}
    if id:
        tc["id"] = id
    if name:
        tc["function"]["name"] = name
    if extra:
        tc["extra_content"] = extra
    return {"choices": [{"delta": {"tool_calls": [tc]}}]}


class Fake:
    """Answers requests in order with the given responses; records what was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def handler(self, request):
        self.requests.append(json.loads(request.content))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def setup():
    store = Store("sqlite://")
    tracker = QuotaTracker(store, {"p": PoolSpec(requests_per_day=100, tokens_per_day=1_000_000)}, reserve_calls=2)
    events = []

    async def notify(e):
        events.append(e)

    def make(fake, *, signatures=False, headers=None, max_wait_s=90):
        meter = Meter(tracker, "p", "Groq · Test", notify, lambda: False, headers)
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        return OpenAICompatLLM(label="Groq · Test", base_url="https://x.test/v1", api_key="k", model="m",
                               meter=meter, http=http, thought_signatures=signatures, max_wait_s=max_wait_s)

    return make, tracker, events


async def collect(client, messages=None, tools=None):
    return [p async for p in client.stream(messages or [{"role": "user", "content": "hi"}], tools)]


async def test_streams_text_tool_calls_and_counts_reported_usage(setup):
    make, tracker, events = setup
    body = sse(
        text("Building "), text("now."),
        tool_delta(0, id="call_1", name="project_build_image", args='{"proj'),
        tool_delta(0, args='ect": "shop"}'),
        usage={"prompt_tokens": 1200, "completion_tokens": 34, "total_tokens": 1234},
    )
    fake = Fake(httpx.Response(200, content=body))
    pieces = await collect(make(fake))
    assert pieces[:2] == ["Building ", "now."]
    assert pieces[2] == ToolCall("project_build_image", {"project": "shop"}, id="call_1")
    assert pieces[3] == UsageReport(1200, 34)
    snap = tracker.snapshot("p")
    assert (snap["requests_used"], snap["tokens_used"], snap["tokens_estimated"]) == (1, 1234, 0)
    assert {"type": "usage_changed"} in events
    assert fake.requests[0]["stream_options"] == {"include_usage": True}


async def test_think_blocks_are_removed_even_across_chunks(setup):
    make, _, _ = setup
    fake = Fake(httpx.Response(200, content=sse(text("<thi"), text("nk>secret plan</th"), text("ink>Hello"))))
    assert "".join(p for p in await collect(make(fake)) if isinstance(p, str)) == "Hello"


async def test_a_per_minute_429_is_waited_out_and_not_counted(setup):
    make, tracker, events = setup
    fake = Fake(
        httpx.Response(429, headers={"retry-after": "0.05"}, content=b'{"error": "rate limit: tokens per minute"}'),
        httpx.Response(200, content=sse(text("ok"), usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11})),
    )
    pieces = await collect(make(fake))
    assert "ok" in pieces
    assert tracker.snapshot("p")["requests_used"] == 1  # only the accepted request
    assert any(e["type"] == "rate_wait" for e in events)


async def test_a_daily_429_stops_and_blocks_the_model_for_today(setup):
    make, tracker, _ = setup
    fake = Fake(httpx.Response(429, content=b'{"error": "Rate limit reached: requests per day (RPD)"}'))
    with pytest.raises(QuotaExhausted, match="refused more requests today"):
        await collect(make(fake))
    assert tracker.snapshot("p")["blocked"] and not tracker.usable("p")[0]


async def test_errors_are_explained_and_counted_conservatively(setup):
    make, tracker, _ = setup
    with pytest.raises(ProviderError, match="rejected the API key"):
        await collect(make(Fake(httpx.Response(401, content=b"bad key"))))
    gemini_404 = b'[{"error": {"code": 404, "message": "models/m is not found for API version v1beta"}}]'
    with pytest.raises(ProviderError, match="can't find the model 'm' .* said: models/m is not found for API version v1beta"):
        await collect(make(Fake(httpx.Response(404, content=gemini_404))))
    assert tracker.snapshot("p")["requests_used"] == 2  # an answered request may count at the provider


async def test_a_request_that_never_reached_the_provider_is_not_counted(setup):
    make, tracker, _ = setup
    with pytest.raises(Exception, match="Can't reach"):
        await collect(make(Fake(httpx.ConnectError("down"))))
    assert tracker.snapshot("p")["requests_used"] == 0


async def test_stopping_mid_stream_still_counts_the_request(setup):
    make, tracker, _ = setup
    fake = Fake(httpx.Response(200, content=sse(text("one "), text("two "), text("three"))))
    stream = make(fake).stream([{"role": "user", "content": "hi"}], None)
    assert await stream.__anext__() == "one "
    await stream.aclose()  # what Stop does
    snap = tracker.snapshot("p")
    assert snap["requests_used"] == 1 and snap["tokens_estimated"] > 0


async def test_groq_headers_sync_the_daily_count(setup):
    make, tracker, _ = setup
    headers = {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "60"}
    fake = Fake(httpx.Response(200, headers=headers, content=sse(text("ok"))))
    await collect(make(fake, headers=HeaderSpec("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests")))
    assert tracker.snapshot("p")["requests_used"] == 40


def test_history_conversion_pairs_tool_results_with_call_ids(setup):
    make, _, _ = setup
    client = make(Fake(), signatures=True)
    history = [
        {"role": "user", "content": "build"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "a", "arguments": {"x": 1}}},  # made by the local model: no id
            {"function": {"name": "b", "arguments": {}}, "id": "g1",
             "extra_content": {"google": {"thought_signature": "real-sig"}}},
        ]},
        {"role": "tool", "tool_name": "a", "content": "A"},
        {"role": "tool", "tool_name": "b", "content": "B"},
        {"role": "tool", "tool_name": "c", "content": "orphan"},
    ]
    out = client.to_openai(history)
    calls = out[1]["tool_calls"]
    assert out[1]["content"] is None and calls[0]["function"]["arguments"] == '{"x": 1}'
    assert [out[2]["tool_call_id"], out[3]["tool_call_id"]] == [calls[0]["id"], "g1"]
    assert calls[0]["extra_content"]["google"]["thought_signature"] == GEMINI_PLACEHOLDER_SIGNATURE
    assert calls[1]["extra_content"]["google"]["thought_signature"] == "real-sig"
    assert out[4] == {"role": "user", "content": "(Earlier tool output)\norphan"}


async def test_classify_falls_back_to_simpler_json_modes_and_counts_each_try(setup):
    make, tracker, _ = setup
    answer = {"reason": "Build", "live_system": True, "platform": "project", "operation": "change"}
    fake = Fake(
        httpx.Response(400, content=b'{"error": "response_format json_schema not supported"}'),
        httpx.Response(200, json={"choices": [{"message": {"content": "<think>hm</think>" + json.dumps(answer)}}],
                                  "usage": {"total_tokens": 300}}),
    )
    result = await make(fake).classify([{"role": "user", "content": "build it"}])
    assert result.platform == "project" and result.operation == "change"
    assert fake.requests[1]["response_format"] == {"type": "json_object"}
    assert tracker.snapshot("p")["requests_used"] == 2


def test_config_allows_only_free_providers():
    base = dict(pools={"p": PoolSpec()})
    with pytest.raises(ValueError, match="isn't a free"):
        Settings(models=[{"id": "local", "label": "L", "provider": "ollama"},
                         {"id": "or", "label": "x", "provider": "openrouter", "model": "openai/gpt-5", "pool": "p"}], **base)
    with pytest.raises(ValueError, match="must be this machine"):
        Settings(ollama_host="https://ollama.com", models=[{"id": "local", "label": "L", "provider": "ollama"}])
    with pytest.raises(ValueError, match="needs a pool"):
        Settings(models=[{"id": "local", "label": "L", "provider": "ollama"},
                         {"id": "g", "label": "x", "provider": "groq", "model": "m"}])
    with pytest.raises(ValueError):
        Settings(models=[{"id": "local", "label": "L", "provider": "mistral", "model": "m"}])



async def test_timeouts_say_what_happened(setup):
    make, tracker, _ = setup
    with pytest.raises(Exception, match="didn't answer within 180 s"):
        await collect(make(Fake(httpx.ReadTimeout(""))))  # httpx gives no text here


async def test_a_busy_server_is_retried_and_each_try_is_counted(setup):
    make, tracker, events = setup
    busy = httpx.Response(503, content=b'{"error": {"message": "The model is overloaded."}}')
    ok = httpx.Response(200, content=sse(text("fine"), usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}))
    import app.llm_openai as mod
    mod._BUSY_WAITS = (0.01, 0.01)
    try:
        assert "fine" in await collect(make(Fake(busy, ok)))
        assert tracker.snapshot("p")["requests_used"] == 2  # the 503 may count at the provider
        assert any(e["type"] == "rate_wait" and "busy (HTTP 503)" in e["reason"] for e in events)
        with pytest.raises(ProviderError, match="overloaded or unavailable .* It said: The model is overloaded."):
            await collect(make(Fake(busy, busy, busy)))
    finally:
        mod._BUSY_WAITS = (3, 8)
