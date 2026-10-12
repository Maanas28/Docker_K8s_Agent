"""The model dropdown end to end: agent + ModelHub + cloud client (fake HTTP) + usage counter."""

import json

import httpx
import pytest

from app.agent import Agent
from app.approvals import ApprovalBroker
from app.config import Settings
from app.models import ModelHub
from app.store import Store
from tests.conftest import FakeLLM

CLASSIFY = {"reason": "Read a thing", "live_system": True, "platform": "docker", "operation": "read"}


def settings(**pool):
    return Settings(
        approval_timeout_s=2, max_errors=4,
        models=[
            {"id": "local", "label": "Local model", "provider": "ollama"},
            {"id": "groq-x", "label": "X 27B", "provider": "groq", "model": "x-27b", "pool": "groq-x"},
            {"id": "or-y", "label": "Y", "provider": "openrouter", "model": "y:free", "pool": "or"},
        ],
        pools={"groq-x": {"requests_per_day": 100, **pool}, "or": {"requests_per_day": 50}},
        groq_api_key="test-key", openrouter_api_key=None,
    )


class Provider:
    """Fake Groq: answers classification and chat requests, records them."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        assert request.headers["authorization"] == "Bearer test-key"
        if not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(CLASSIFY)}}],
                                             "usage": {"total_tokens": 200}})
        text = self.replies.pop(0)
        chunks = [{"choices": [{"delta": {"content": text}}]}, {"choices": [], "usage": {"total_tokens": 900}}]
        return httpx.Response(200, content="".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n")


@pytest.fixture
def make(registry):
    def _make(cfg, provider):
        store = Store("sqlite://")
        http = httpx.AsyncClient(transport=httpx.MockTransport(provider.handler))
        hub = ModelHub(cfg, store, local_llm=FakeLLM("action", [["local reply"]]), http=http)
        events = []

        async def emit(e):
            events.append(e)

        agent = Agent(llm=None, registry=registry, store=store, broker=ApprovalBroker(2), emit=emit,
                      session_id="s", settings=cfg, hub=hub)
        return agent, hub, events
    return _make


def last_models(events):
    return {m["id"]: m for m in [e for e in events if e["type"] == "models"][-1]["models"]}


async def test_chats_start_on_the_local_model_and_list_every_model(make):
    agent, _, events = make(settings(), Provider([]))
    await agent.emit_models()
    models = last_models(events)
    assert models["local"]["selected"] and models["local"]["local"] and models["local"]["usage"] is None
    assert models["groq-x"]["usable"] and models["groq-x"]["usage"]["calls_left"] == 100
    assert not models["or-y"]["usable"] and models["or-y"]["reason"] == "add AGENT_OPENROUTER_API_KEY to backend/.env"


async def test_switching_to_a_cloud_model_counts_every_request(make):
    provider = Provider(["Hello from Groq."])
    agent, hub, events = make(settings(), provider)
    assert await agent.set_model("groq-x") is None
    await agent.handle("hi")
    assert len(provider.requests) == 2  # classification + reply
    assert provider.requests[1]["model"] == "x-27b"
    usage = last_models(events)["groq-x"]["usage"]
    assert (usage["requests_used"], usage["tokens_used"], usage["calls_left"]) == (2, 1100, 98)
    assert last_models(events)["groq-x"]["selected"]
    assert "Hello from Groq." in "".join(e.get("text", "") for e in events if e["type"] == "token")


async def test_near_the_limit_the_chat_stops_cleanly_and_the_model_is_greyed_out(make):
    provider = Provider(["never sent"])
    agent, hub, events = make(settings(requests_per_day=5), provider)
    await agent.set_model("groq-x")
    for _ in range(2):  # 2 used elsewhere today; 3 left, reserve 2
        hub.tracker.release(hub.tracker.reserve("groq-x", 100), counted=True, tokens=100)
    await agent.handle("hi")  # classification uses the 3rd; the reply would go below the reserve
    assert len(provider.requests) == 1
    said = "".join(e.get("text", "") for e in events if e["type"] == "token")
    assert "down to about 2 free calls" in said and "Pick another model" in said
    groq = last_models(events)["groq-x"]
    assert not groq["usable"] and groq["reason"].startswith("2 calls left, kept in reserve")
    assert await agent.set_model("groq-x") is not None  # can't be picked again today
    assert await agent.set_model("local") is None  # the local model is always there


async def test_a_model_without_its_key_cant_be_chosen(make):
    agent, _, _ = make(settings(), Provider([]))
    assert "add AGENT_OPENROUTER_API_KEY" in await agent.set_model("or-y")
    assert agent.model_id == "local"


async def test_the_conversation_carries_over_when_switching(make):
    provider = Provider(["Second answer."])
    agent, _, _ = make(settings(), provider)
    await agent.handle("first question")  # local model
    await agent.set_model("groq-x")
    await agent.handle("second question")
    sent = provider.requests[-1]["messages"]
    assert [m["content"] for m in sent if m["role"] == "user"][:2] == ["first question", "second question"]
