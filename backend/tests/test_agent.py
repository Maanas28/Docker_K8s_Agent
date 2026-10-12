import json
import asyncio

import pytest

from app.agent import Agent
from app.approvals import ApprovalBroker
from app.llm import ToolCall
from tests.conftest import FakeLLM


class Harness:
    def __init__(self, llm, registry, store, settings):
        self.events: list[dict] = []
        self.broker = ApprovalBroker(settings.approval_timeout_s)
        self.agent = Agent(
            llm=llm, registry=registry, store=store, broker=self.broker,
            emit=self._emit, session_id="s1", settings=settings,
        )
        self.decisions: list[tuple[str, str | None]] = []  # answered in order, per approval request

    async def _emit(self, event):
        self.events.append(event)
        if event["type"] == "approval_request" and self.decisions:
            decision, text = self.decisions.pop(0)
            asyncio.get_running_loop().call_soon(self.broker.resolve, event["id"], decision, text)

    def types(self):
        return [e["type"] for e in self.events]

    def text(self):
        return "".join(e["text"] for e in self.events if e["type"] == "token")


@pytest.fixture
def make(registry, store, settings):
    def _make(llm):
        return Harness(llm, registry, store, settings)
    return _make


async def test_question_is_answered_without_tools(make, executed):
    llm = FakeLLM("question", [["A pod is ", "the smallest unit."]])
    h = make(llm)
    await h.agent.handle("what is a pod?")
    assert h.text() == "A pod is the smallest unit."
    assert llm.stream_calls[0]["tools"] is None
    assert executed == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(platform="kubernetes"), "I can't read from Kubernetes yet."),
        (dict(platform="other"), "I can only work with Docker, Kubernetes and your registered code projects."),
    ],
)
async def test_unsupported_is_refused_without_llm_or_tools(make, executed, overrides, reason):
    llm = FakeLLM("action", [], **overrides)
    h = make(llm)
    await h.agent.handle("list my pods")
    assert h.events[0] == {"type": "classification", "kind": "unsupported", "reason": reason}
    assert f"I can't do that. {reason}" in h.text()
    assert llm.stream_calls == [] and executed == []


async def test_change_without_change_tools_is_unsupported(make, executed, registry):
    registry._tools = {k: v for k, v in registry._tools.items() if k == "read_thing"}
    llm = FakeLLM("change", [])
    h = make(llm)
    await h.agent.handle("restart nginx")
    assert h.events[0]["reason"] == "I can't make changes to Docker yet."
    assert executed == []


async def test_read_needs_approval_the_first_time(make, executed, store):
    llm = FakeLLM("action", [[ToolCall("read_thing", {"name": "web"})], ["All good."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("check web")
    assert h.types()[:3] == ["classification", "approval_request", "approval_decision"]
    assert executed == [("read_thing", {"name": "web"})]
    assert h.text() == "All good."
    assert store.recent_audit()[0].decision == "approve"


async def test_always_allow_skips_future_approvals(make, executed, store):
    llm = FakeLLM("action", [
        [ToolCall("read_thing", {"name": "a"})], ["done"],
        [ToolCall("read_thing", {"name": "b"})], ["done"],
    ])
    h = make(llm)
    h.decisions = [("always_allow", None)]
    await h.agent.handle("check a")
    await h.agent.handle("check b")
    assert store.is_always_allowed("read_thing")
    assert h.types().count("approval_request") == 1
    assert "tool_started" in h.types()
    assert [name for name, _ in executed] == ["read_thing", "read_thing"]


async def test_rejection_runs_nothing_and_stops(make, executed, store):
    llm = FakeLLM("change", [[ToolCall("scale_thing", {"name": "web", "replicas": 3})]])
    h = make(llm)
    h.decisions = [("reject", None)]
    await h.agent.handle("scale web to 3")
    assert executed == []
    assert "didn't run it" in h.text()
    assert len(llm.stream_calls) == 1  # no retry
    assert store.recent_audit()[0].outcome == "not_run"


async def test_change_tools_always_ask_even_if_always_allow_was_stored(make, executed, store):
    store.allow_always("scale_thing")  # shouldn't be possible, but must not bypass the gate
    llm = FakeLLM("change", [[ToolCall("scale_thing", {"name": "web", "replicas": 2})], ["ok"]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("scale web")
    assert "approval_request" in h.types()


async def test_no_answer_expires_and_runs_nothing(make, executed, settings):
    settings.approval_timeout_s = 0.05
    llm = FakeLLM("action", [[ToolCall("read_thing", {"name": "web"})]])
    h = make(llm)
    await h.agent.handle("check web")
    assert executed == []
    assert {"type": "approval_decision", "id": h.events[1]["id"], "decision": "expired"} in h.events


async def test_destructive_requires_typed_name(make, executed):
    llm = FakeLLM("change", [[ToolCall("delete_thing", {"name": "db"})], ["deleted"]])
    h = make(llm)
    h.decisions = [("approve", "db")]
    await h.agent.handle("delete db")
    request = next(e for e in h.events if e["type"] == "approval_request")
    assert request["confirm_text"] == "db" and request["can_always_allow"] is False
    assert executed == [("delete_thing", {"name": "db"})]


async def test_unknown_tool_is_refused(make, executed, store):
    llm = FakeLLM("action", [[ToolCall("rm_rf", {"path": "/"})], ["That's not supported."]])
    h = make(llm)
    await h.agent.handle("wipe the disk")
    assert "tool_refused" in h.types() and executed == []
    assert "no tool named 'rm_rf'" in llm.stream_calls[1]["messages"][-1]["content"]
    assert store.recent_audit()[0].decision == "unknown_tool"


async def test_invalid_args_go_back_to_llm_without_asking(make, executed):
    llm = FakeLLM("change", [
        [ToolCall("scale_thing", {"name": "web", "replicas": "lots"})],
        ["How many replicas?"],
    ])
    h = make(llm)
    await h.agent.handle("scale web a lot")
    assert "approval_request" not in h.types() and executed == []
    assert "invalid arguments" in llm.stream_calls[1]["messages"][-1]["content"]


async def test_tool_errors_are_reported(make, executed):
    llm = FakeLLM("action", [[ToolCall("read_thing", {"name": "missing"})], ["Not found."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("check missing")
    result = next(e for e in h.events if e["type"] == "tool_result")
    assert result["ok"] is False and "No thing named" in result["result"]["message"]


async def test_preview_is_shown_on_the_approval_card(make, registry, executed):
    from app.tools.registry import Risk
    from tests.conftest import NameArgs

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Stop {a.name}", platform="docker",
                   preview=lambda a: {"facts": {"Status": "running"}, "warnings": ["Downtime"]})
    def stop_thing(args: NameArgs) -> dict:
        """Stop a thing."""
        executed.append(("stop_thing", args.model_dump()))
        return {"kind": "facts", "title": "Stopped", "facts": {}}

    llm = FakeLLM("change", [[ToolCall("stop_thing", {"name": "web"})], ["Stopped."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("stop web")
    request = next(e for e in h.events if e["type"] == "approval_request")
    assert request["preview"] == {"facts": {"Status": "running"}, "warnings": ["Downtime"]}
    assert executed == [("stop_thing", {"name": "web"})]


async def test_failed_precheck_never_asks_or_runs(make, registry, executed, store):
    from app.tools.registry import Risk, ToolError
    from tests.conftest import NameArgs

    def already_running(args):
        raise ToolError("'web' is already running.")

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Start {a.name}", platform="docker", preview=already_running)
    def start_thing(args: NameArgs) -> dict:
        """Start a thing."""
        executed.append(("start_thing", args.model_dump()))
        return {}

    llm = FakeLLM("change", [[ToolCall("start_thing", {"name": "web"})], ["It's already running."]])
    h = make(llm)
    await h.agent.handle("start web")
    assert "approval_request" not in h.types() and executed == []
    refused = next(e for e in h.events if e["type"] == "tool_refused")
    assert "already running" in refused["message"]
    assert "pre-check failed" in llm.stream_calls[1]["messages"][-1]["content"]
    assert store.recent_audit()[0].decision == "precheck_failed"


def _grounded_tool(registry, executed):
    from app.tools.registry import Risk, ToolArgs

    class PullArgs(ToolArgs):
        image: str

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Pull {a.image}", platform="docker", grounded=("image",))
    def pull_thing(args: PullArgs) -> dict:
        """Pull a thing."""
        executed.append(("pull_thing", args.model_dump()))
        return {"kind": "facts", "title": "Pulled", "facts": {}}


async def test_made_up_names_are_refused(make, registry, executed, store):
    _grounded_tool(registry, executed)
    llm = FakeLLM("change", [[ToolCall("pull_thing", {"image": "shop_project"})], ["Which image do you mean?"]])
    h = make(llm)
    await h.agent.handle("now pull the image")
    assert "approval_request" not in h.types() and executed == []
    assert "wasn't mentioned by you" in next(e for e in h.events if e["type"] == "tool_refused")["message"]
    assert store.recent_audit()[0].decision == "ungrounded"


@pytest.mark.parametrize(
    "user_text, image",
    [
        ("pull redis", "redis"),
        ("pull the latest redis image", "redis:latest"),
        ("please pull nginx:1.27.", "nginx:1.27"),
        ("pull ghcr.io/org/app:v1", "ghcr.io/org/app:v1"),
    ],
)
async def test_names_the_user_typed_are_allowed(make, registry, executed, user_text, image):
    _grounded_tool(registry, executed)
    llm = FakeLLM("change", [[ToolCall("pull_thing", {"image": image})], ["Pulled."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle(user_text)
    assert executed == [("pull_thing", {"image": image})]


async def test_names_from_tool_output_are_allowed(make, registry, executed):
    _grounded_tool(registry, executed)
    llm = FakeLLM("change", [
        [ToolCall("read_thing", {"name": "x"})],
        [ToolCall("pull_thing", {"image": "shop-api:latest"})],
        ["Pulled."],
    ])
    h = make(llm)
    h.agent.history.append({"role": "tool", "tool_name": "list", "content": '{"image": "shop-api:latest"}'})
    h.decisions = [("approve", None), ("approve", None)]
    await h.agent.handle("update the image of the api container")
    assert ("pull_thing", {"image": "shop-api:latest"}) in executed


async def test_substrings_are_not_enough(make, registry, executed):
    _grounded_tool(registry, executed)
    llm = FakeLLM("change", [[ToolCall("pull_thing", {"image": "shop"})], ["Which one?"]])
    h = make(llm)
    h.agent.history.append({"role": "tool", "tool_name": "list", "content": '{"image": "shop-api:latest"}'})
    await h.agent.handle("pull it")
    assert executed == []


async def test_missing_model_is_not_hidden_by_the_fallback(make):
    from app.llm import LLMUnavailable

    class NoModel(FakeLLM):
        async def classify(self, messages):
            raise LLMUnavailable("The model x isn't downloaded. Run `ollama pull x` and try again.")

    h = make(NoModel("action", []))
    with pytest.raises(LLMUnavailable, match="ollama pull x"):
        await h.agent.handle("show my containers")
    assert h.types() == []  # no fake "action" classification was shown


async def test_always_allow_for_a_project_covers_only_that_project(make, registry, executed, store):
    from app.tools.registry import Risk, ToolArgs

    class Args(ToolArgs):
        project: str

    @registry.tool(risk=Risk.READ, summary=lambda a: f"Read {a.project}", platform="project", scope_field="project")
    def read_code(args: Args) -> dict:
        """Read code."""
        executed.append(("read_code", args.project))
        return {"kind": "facts", "title": "ok", "facts": {}}

    llm = FakeLLM("action", [
        [ToolCall("read_code", {"project": "a"})], ["ok"],
        [ToolCall("read_code", {"project": "a"})], ["ok"],
        [ToolCall("read_code", {"project": "b"})], ["ok"],
    ], platform="project")
    h = make(llm)
    h.decisions = [("always_allow", None), ("approve", None)]
    await h.agent.handle("read project a")
    first = next(e for e in h.events if e["type"] == "approval_request")
    assert first["always_allow_label"] == "Always allow reading a"
    await h.agent.handle("read project a again")
    await h.agent.handle("read project b")
    assert store.project_reads_allowed("a") and not store.project_reads_allowed("b")
    assert not store.is_always_allowed("read_code")  # the grant is per project, not for the tool
    assert h.types().count("approval_request") == 2  # a (first time) and b; not a again
    assert executed == [("read_code", "a"), ("read_code", "a"), ("read_code", "b")]


async def test_calls_written_as_text_are_run_through_the_normal_gate(make, executed, store):
    text_call = 'Let me look.\n```json\n{"name": "read_thing", "arguments": {"name": "web"}}\n```'
    llm = FakeLLM("action", [[text_call], ["All good."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("check web")
    rewrite = next(e for e in h.events if e["type"] == "assistant_rewrite")
    assert rewrite == {"type": "assistant_rewrite", "from": text_call, "to": ""}
    assert "approval_request" in h.types()  # still needs approval
    assert executed == [("read_thing", {"name": "web"})]
    assert h.agent.history[1]["tool_calls"] == [{"function": {"name": "read_thing", "arguments": {"name": "web"}}}]
    assert h.agent.history[1]["content"] == ""


async def test_text_calls_are_not_recovered_for_general_questions(make, executed):
    llm = FakeLLM("question", [['Example: {"name": "read_thing", "arguments": {"name": "web"}}']])
    h = make(llm)
    await h.agent.handle("how do tool calls look?")
    assert executed == [] and "assistant_rewrite" not in h.types()


async def test_registered_projects_are_in_the_instructions(make, store, tmp_path):
    store.add_project("sample-project", str(tmp_path / "sample-fastapi-app"))
    llm = FakeLLM("action", [["Which file?"]])
    h = make(llm)
    await h.agent.handle("tell me about my sample project")
    system = llm.stream_calls[0]["messages"][0]["content"]
    assert "- sample-project (folder: sample-fastapi-app)" in system


def test_projects_are_given_to_the_classifier():
    from app.prompts import build_classify_messages

    msgs = build_classify_messages([{"role": "user", "content": "about sample-project?"}], ["sample-project"])
    assert "registered code projects" in msgs[0]["content"] and "sample-project" in msgs[0]["content"]


@pytest.mark.parametrize("decision, written", [("approve", True), ("reject", False)])
async def test_writing_a_file_needs_approval_of_the_diff(tmp_path, settings, decision, written):
    from app.store import Store
    from app.tools import code_tools, load_all
    from app.tools.registry import registry as real_registry

    load_all()
    root = tmp_path / "shop"
    root.mkdir()
    store = Store("sqlite://")
    store.add_project("shop", str(root))
    code_tools.configure(store)
    llm = FakeLLM("change", [
        [ToolCall("project_write_file", {"project": "shop", "path": ".dockerignore", "content": ".env\n.git\n", "reason": "Keep secrets out."})],
        ["Done."],
    ], platform="project")
    h = Harness(llm, real_registry, store, settings)
    h.decisions = [(decision, None)]
    await h.agent.handle("write a dockerignore for shop")
    card = next(e for e in h.events if e["type"] == "approval_request")
    assert card["preview"]["file"]["diff"].startswith("--- /dev/null") and card["can_always_allow"] is False
    assert (root / ".dockerignore").exists() is written


async def test_described_but_not_made_calls_get_one_hidden_nudge(make, executed):
    llm = FakeLLM("action", [
        ["Please call `read_thing` with the name web."],             # describes, doesn't call
        [ToolCall("read_thing", {"name": "web"})],                  # after the hidden nudge
        ["All good."],
    ])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("check web")
    assert executed == [("read_thing", {"name": "web"})]
    nudge = llm.stream_calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "make the tool call yourself" in nudge["content"]
    assert {"type": "assistant_rewrite", "from": "Please call `read_thing` with the name web.", "to": ""} in h.events
    assert not any("make the tool call yourself" in (m.get("content") or "") for m in h.agent.history)


async def test_no_nudge_for_plain_answers(make):
    llm = FakeLLM("action", [["Which container do you mean?"]])
    h = make(llm)
    await h.agent.handle("check it")
    assert len(llm.stream_calls) == 1



def _failing_preview_tool(registry, executed):
    from app.tools.registry import Risk, ToolError
    from tests.conftest import NameArgs

    def always_fails(args):
        raise ToolError("the first line must be FROM.")

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Write {a.name}", platform="docker", preview=always_fails)
    def write_thing(args: NameArgs) -> dict:
        """Write a thing."""
        executed.append(("write_thing", args.model_dump()))
        return {}


async def test_identical_refused_call_is_not_retried_and_then_stops(make, registry, executed, store):
    _failing_preview_tool(registry, executed)
    same = ToolCall("write_thing", {"name": "Dockerfile"})
    llm = FakeLLM("change", [[same], [same], [same], [same], ["unreachable"]])
    h = make(llm)
    await h.agent.handle("write it")
    refused = [e["message"] for e in h.events if e["type"] == "tool_refused"]
    assert len(refused) == 2 and "same as the attempt that was just refused" in refused[1]
    assert "identical to your previous attempt" in llm.stream_calls[2]["messages"][-1]["content"]
    assert "I got stuck" in h.text() and "the first line must be FROM." in h.text()
    assert len(llm.stream_calls) == 3  # stopped instead of looping to the step limit
    assert executed == []
    decisions = [e.decision for e in store.recent_audit()]
    assert decisions.count("precheck_failed") == 1 and decisions.count("repeat_refused") == 1


async def test_a_changed_attempt_is_checked_normally(make, registry, executed):
    _failing_preview_tool(registry, executed)
    llm = FakeLLM("change", [
        [ToolCall("write_thing", {"name": "v1"})],
        [ToolCall("write_thing", {"name": "v2"})],
        ["Couldn't do it."],
    ])
    h = make(llm)
    await h.agent.handle("write it")
    refused = [e["message"] for e in h.events if e["type"] == "tool_refused"]
    assert len(refused) == 2 and all("first line must be FROM" in m for m in refused)


async def test_refusals_reset_for_each_new_request(make, registry, executed):
    _failing_preview_tool(registry, executed)
    same = ToolCall("write_thing", {"name": "x"})
    llm = FakeLLM("change", [[same], ["no"], [same], ["no"]])
    h = make(llm)
    await h.agent.handle("write it")
    await h.agent.handle("try again")
    refused = [e["message"] for e in h.events if e["type"] == "tool_refused"]
    assert len(refused) == 2 and all("first line must be FROM" in m for m in refused)


# ---- error budget and Stop ----------------------------------------------------------------------


async def test_successful_steps_are_not_limited(make, executed, store):
    store.allow_always("read_thing")
    turns = [[ToolCall("read_thing", {"name": f"f{i}"})] for i in range(12)] + [["Done reading."]]
    h = make(FakeLLM("action", turns))
    await h.agent.handle("read everything")
    assert len(executed) == 12 and h.text().endswith("Done reading.")  # max_errors=4 in tests, no step cap


async def test_errors_stop_at_the_budget(make, executed, store):
    store.allow_always("read_thing")
    # Different arguments each time, so the repeat guard doesn't kick in; every call fails.
    turns = [[ToolCall("read_thing", {"name": "missing", "extra": i})] for i in range(10)]
    h = make(FakeLLM("action", turns))
    await h.agent.handle("read it")
    assert "I stopped after 4 errors in this request" in h.text()
    assert "read_thing: invalid arguments" in h.text()  # the last errors are listed


async def test_failed_tools_count_as_errors(make, executed, store):
    store.allow_always("read_thing")
    turns = [[ToolCall("read_thing", {"name": f"missing-{i}"})] for i in range(4)]
    h = make(FakeLLM("action", turns))
    # Different errors each time (the same error twice stops earlier, see below); a tool that ran
    # and failed is an ordinary error, counted by the budget.
    await h.agent.handle("read missing")
    assert "I stopped after 4 errors" in h.text() and "No thing named 'missing-3'" in h.text()
    assert len(executed) == 4


async def test_stop_while_a_card_is_open_runs_nothing(make, executed):
    llm = FakeLLM("change", [[ToolCall("scale_thing", {"name": "web", "replicas": 2})], ["unreachable"]])
    h = make(llm)

    async def emit(event):
        h.events.append(event)
        if event["type"] == "approval_request":
            h.agent.request_stop()  # the user presses Stop while the card is open

    h.agent.emit = emit
    await h.agent.handle("scale web")
    assert executed == [] and h.text().endswith("Stopped by you. Nothing else will run.")
    assert len(llm.stream_calls) == 1


async def test_stop_mid_reply_closes_the_model_stream(make):
    closed = []

    class Streaming(FakeLLM):
        async def stream(self, messages, tools):
            try:
                for word in ["one ", "two ", "three ", "four "]:
                    yield word
            finally:
                closed.append(True)  # runs when the consumer closes the generator

    h = make(Streaming("question", []))

    async def emit(event):
        h.events.append(event)
        if event["type"] == "token" and event["text"] == "two ":
            h.agent.request_stop()

    h.agent.emit = emit
    await h.agent.handle("count")
    tokens = [e["text"] for e in h.events if e["type"] == "token"]
    assert tokens[:2] == ["one ", "two "] and "three " not in tokens
    assert closed == [True] and "Stopped by you." in h.text()


async def test_stop_flag_resets_for_the_next_request(make, executed, store):
    store.allow_always("read_thing")
    h = make(FakeLLM("action", [[ToolCall("read_thing", {"name": "a"})], ["ok"]]))
    h.agent.request_stop()  # pressed while idle
    await h.agent.handle("read a")
    assert executed == [("read_thing", {"name": "a"})]


# ---- context techniques inside the agent ----------------------------------------------------------


async def test_only_tools_for_the_request_platform_are_sent(make, registry, executed):
    from app.tools.registry import Risk, ToolArgs

    class P(ToolArgs):
        project: str

    @registry.tool(risk=Risk.READ, summary=lambda a: "x", platform="project", scope_field="project")
    def project_peek(args: P) -> dict:
        """Peek."""
        return {}

    llm = FakeLLM("action", [["ok"]])  # classified as docker
    h = make(llm)
    await h.agent.handle("list containers")
    sent = {t["function"]["name"] for t in llm.stream_calls[0]["tools"]}
    assert "project_peek" not in sent and "read_thing" in sent


async def test_all_tools_are_sent_when_the_flag_is_off(make, registry, settings):
    settings.context.relevant_tools_only = False
    llm = FakeLLM("action", [["ok"]])
    h = make(llm)
    await h.agent.handle("list containers")
    assert len(llm.stream_calls[0]["tools"]) == len(registry.all())


async def test_attempt_log_reaches_the_instructions_of_later_requests(make, registry, executed):
    _failing_preview_tool(registry, executed)
    llm = FakeLLM("change", [[ToolCall("write_thing", {"name": "v1"})], ["sorry"], ["ok"]])
    h = make(llm)
    await h.agent.handle("write it")
    await h.agent.handle("try again")
    system = llm.stream_calls[2]["messages"][0]["content"]
    assert "Earlier attempts in this conversation" in system and "first line must be FROM" in system


async def test_attempt_log_flag_off(make, registry, executed, settings):
    settings.context.attempt_log = False
    _failing_preview_tool(registry, executed)
    llm = FakeLLM("change", [[ToolCall("write_thing", {"name": "v1"})], ["sorry"]])
    h = make(llm)
    await h.agent.handle("write it")
    assert "Earlier attempts" not in llm.stream_calls[1]["messages"][0]["content"]


async def test_history_keeps_full_results_while_the_model_gets_trimmed_ones(make, executed, store):
    store.allow_always("read_thing")
    same = ToolCall("read_thing", {"name": "web"})
    llm = FakeLLM("action", [[same], [same], ["done"]])
    h = make(llm)
    await h.agent.handle("read web twice")
    tool_msgs = [m for m in h.agent.history if m["role"] == "tool"]
    assert all(not m["content"].startswith("[outdated") for m in tool_msgs)  # history: full
    sent = [m for m in llm.stream_calls[2]["messages"] if m.get("role") == "tool"]
    assert sent[0]["content"].startswith("[outdated") and not sent[1]["content"].startswith("[outdated")


# ---- long-running tools (Step 3: builds) ----------------------------------------------------------


@pytest.fixture
def slow_tool(registry, executed):
    """An async tool that streams lines and runs until it's stopped (like a build)."""
    from app.tools.registry import Risk, RunContext, ToolError
    from tests.conftest import NameArgs

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Build {a.name}", platform="docker")
    async def build_thing(args: NameArgs, ctx: RunContext) -> dict:
        """Build a thing."""
        executed.append(("build_thing", args.model_dump()))
        for i in range(3):
            await ctx.progress(f"step {i}")
        if args.name == "broken":
            raise ToolError("The build failed: exit code 1",
                            result={"kind": "build", "ok": False, "error": "exit code 1", "log": "step 2\nboom"})
        if args.name == "forever":
            await ctx.stopped.wait()
            raise ToolError("Build cancelled because you pressed Stop.")
        return {"kind": "build", "ok": True, "title": "Built", "facts": {}, "log": "step 0"}


async def test_progress_lines_reach_the_ui_before_the_result(make, slow_tool, settings):
    settings.build.progress_interval_ms = 10_000  # only the final flush sends them
    h = make(FakeLLM("change", [[ToolCall("build_thing", {"name": "web"})], ["Built."]]))
    h.decisions = [("approve", None)]
    await h.agent.handle("build web")
    progress = [e for e in h.events if e["type"] == "tool_progress"]
    assert [e["lines"] for e in progress] == [["step 0", "step 1", "step 2"]]  # batched into one event
    assert h.types().index("tool_progress") < h.types().index("tool_result")


async def test_a_failed_build_shows_its_log_and_the_model_gets_it(make, slow_tool, store):
    llm = FakeLLM("change", [[ToolCall("build_thing", {"name": "broken"})], ["It failed."]])
    h = make(llm)
    h.decisions = [("approve", None)]
    await h.agent.handle("build broken")
    result = next(e for e in h.events if e["type"] == "tool_result")
    assert not result["ok"] and result["result"]["log"] == "step 2\nboom"
    assert "boom" in llm.stream_calls[1]["messages"][-1]["content"]
    assert store.recent_audit()[0].detail == "The build failed: exit code 1"


async def test_stop_cancels_a_running_tool_without_counting_an_error(make, slow_tool, executed):
    llm = FakeLLM("change", [[ToolCall("build_thing", {"name": "forever"})], ["unreachable"]])
    h = make(llm)
    h.decisions = [("approve", None)]

    async def emit(event):
        h.events.append(event)
        if event["type"] == "approval_request":
            asyncio.get_running_loop().call_soon(h.broker.resolve, event["id"], "approve", None)
        if event["type"] == "tool_progress":
            h.agent.request_stop()  # the user presses Stop while the build is running

    h.agent.emit = emit
    h.agent.settings.build.progress_interval_ms = 50
    await asyncio.wait_for(h.agent.handle("build forever"), 5)
    assert executed == [("build_thing", {"name": "forever"})]
    assert "pressed Stop" in next(e for e in h.events if e["type"] == "tool_result")["result"]["message"]
    assert h.agent._errors == [] and len(llm.stream_calls) == 1


def test_tools_may_only_take_known_extra_parameters(registry):
    from app.tools.registry import Risk
    from tests.conftest import NameArgs

    with pytest.raises(ValueError, match="optionally `preview`"):
        @registry.tool(risk=Risk.READ, summary=lambda a: "x")
        def docker_odd(args: NameArgs, extra: dict) -> dict:
            """Odd."""
            return {}


async def test_failures_are_announced_to_the_user(make, slow_tool):
    h = make(FakeLLM("change", [[ToolCall("build_thing", {"name": "broken"})], ["Fixing it."]]))
    h.decisions = [("approve", None)]
    await h.agent.handle("build broken")
    failed = next(e for e in h.events if e["type"] == "step_failed")
    assert failed["summary"] == "Build broken" and failed["message"] == "The build failed: exit code 1"
    assert failed["retrying"] is True
    assert h.types().index("tool_result") < h.types().index("step_failed")


async def test_a_successful_change_resets_the_repeat_guard(make, registry, executed):
    from app.tools.registry import Risk, ToolError
    from tests.conftest import NameArgs
    state = {"fixed": False}

    def _check(args):
        if not state["fixed"]:
            raise ToolError("nothing changed since the failure")
        return {"facts": {}, "warnings": []}

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: "Build", platform="docker", preview=_check)
    def rebuild_thing(args: NameArgs) -> dict:
        """Rebuild."""
        executed.append(("rebuild_thing", {}))
        return {"kind": "json", "title": "ok", "data": {}}

    @registry.tool(risk=Risk.CHANGE, summary=lambda a: "Fix", platform="docker")
    def fix_thing(args: NameArgs) -> dict:
        """Fix."""
        state["fixed"] = True
        return {"kind": "json", "title": "fixed", "data": {}}

    build = ToolCall("rebuild_thing", {"name": "web"})
    llm = FakeLLM("change", [[build], [ToolCall("fix_thing", {"name": "web"})], [build], ["Built."]])
    h = make(llm)
    h.decisions = [("approve", None), ("approve", None)]
    await h.agent.handle("build web")
    assert ("rebuild_thing", {}) in executed  # the identical call was checked again after the fix
    assert "same as the attempt that was just refused" not in json.dumps(h.events)


async def test_the_same_error_twice_in_a_row_stops_the_loop(make, executed, store):
    store.allow_always("read_thing")
    call = ToolCall("read_thing", {"name": "missing"})
    llm = FakeLLM("action", [[call], [call], ["unreachable"]])
    h = make(llm)
    await h.agent.handle("read missing")
    assert len(executed) == 2 and len(llm.stream_calls) == 2
    assert "same error came back twice in a row" in h.text() and "No thing named 'missing'" in h.text()


def test_error_ids_are_ignored_when_comparing():
    from app.agent import _normalize_error
    a = 'failed to calculate checksum of ref hllauwtomw50c6vqhhs87tvev::6xbva2ottjx7vnjm3lttlza9i: "/uv.lock": not found'
    b = 'failed to calculate checksum of ref hllauwtomw50c6vqhhs87tvev::786n8b53xlbpu1qg74ijojd6m: "/uv.lock": not found'
    assert _normalize_error(a) == _normalize_error(b)
    assert _normalize_error("exit code: 127") != _normalize_error("exit code: 1")


async def test_after_a_failure_a_talk_only_reply_gets_one_nudge(make, executed, store):
    store.allow_always("read_thing")
    llm = FakeLLM("action", [
        [ToolCall("read_thing", {"name": "missing"})],
        ["It failed. I'll read the other one instead."],  # talks, no call
        [ToolCall("read_thing", {"name": "web"})],  # the hidden nudge's answer
        ["Done."],
    ])
    h = make(llm)
    await h.agent.handle("read missing")
    assert [n for n, _ in executed] == ["read_thing", "read_thing"]
    assert "_FAILURE_NUDGE" not in h.text() and "I'll read the other one instead." in h.text()  # explanation kept
    assert "don't wait for their OK" in llm.stream_calls[2]["messages"][-1]["content"]
    assert not any("don't wait for their OK" in (m.get("content") or "") for m in h.agent.history)


async def test_no_failure_nudge_without_a_failure(make, executed, store):
    store.allow_always("read_thing")
    llm = FakeLLM("action", [[ToolCall("read_thing", {"name": "web"})], ["All good, I'll keep watching it."]])
    h = make(llm)
    await h.agent.handle("read web")
    assert len(llm.stream_calls) == 2


async def test_brake_and_nudge_can_be_turned_off(make, executed, store, settings):
    settings.loop.same_error_brake = False
    settings.loop.failure_nudge = False
    store.allow_always("read_thing")
    call = ToolCall("read_thing", {"name": "missing"})
    llm = FakeLLM("action", [[call], [call], ["I'll try something else."]])
    h = make(llm)
    await h.agent.handle("read missing")
    assert len(executed) == 2 and len(llm.stream_calls) == 3  # no brake, and no nudge call
    assert "same error" not in h.text()


# ---- "Allow for this session" ---------------------------------------------------------------------


@pytest.fixture
def loop_tools(registry, executed):
    """Two change tools in one session group, scoped by project (like write file + build)."""
    from app.tools.registry import Risk, ToolArgs

    class ProjectArgs(ToolArgs):
        project: str

    def make_tool(name):
        def run(args: ProjectArgs) -> dict:
            executed.append((name, args.model_dump()))
            return {"kind": "json", "title": "ok", "data": {}}
        return run

    for name in ("write_thing", "build_thing2"):
        run = make_tool(name)
        run.__name__ = name
        run.__doc__ = f"{name}."
        registry.tool(risk=Risk.CHANGE, summary=lambda a, n=name: f"{n} {a.project}", platform="docker",
                      session_group="build-loop", session_scope_field="project")(run)


async def test_allow_for_this_session_covers_the_group_on_that_project(make, loop_tools, executed, store):
    llm = FakeLLM("change", [
        [ToolCall("write_thing", {"project": "shop"})],
        [ToolCall("build_thing2", {"project": "shop"})],  # same group, same project: no card
        [ToolCall("write_thing", {"project": "shop"})],
        [ToolCall("write_thing", {"project": "billing"})],  # another project: asks again
        ["done"],
    ])
    h = make(llm)
    h.decisions = [("session_allow", None), ("approve", None)]
    await h.agent.handle("build shop")
    requests = [e for e in h.events if e["type"] == "approval_request"]
    assert [r["args"]["project"] for r in requests] == ["shop", "billing"]
    assert requests[0]["can_session_allow"] and "build the image and run it in 'shop'" in requests[0]["session_allow_label"]
    started = [e for e in h.events if e["type"] == "tool_started"]
    assert [e["allowed_by"] for e in started] == ["session", "session"]
    assert len(executed) == 4
    grants = [e for e in h.events if e["type"] == "session_grants"][-1]["grants"]
    assert grants == [{"group": "build-loop", "scope": "shop", "label": "write the Dockerfile/.dockerignore, build the image and run it in 'shop'"}]
    assert [a.decision for a in reversed(store.recent_audit())] == ["session_allow", "session", "session", "approve"]


async def test_revoked_grants_ask_again(make, loop_tools, executed):
    llm = FakeLLM("change", [[ToolCall("write_thing", {"project": "shop"})], ["ok"],
                             [ToolCall("write_thing", {"project": "shop"})], ["ok"]])
    h = make(llm)
    h.decisions = [("session_allow", None), ("approve", None)]
    await h.agent.handle("write")
    await h.agent.revoke_session_grant("build-loop", "shop")
    await h.agent.handle("write again")
    assert h.types().count("approval_request") == 2
    assert [e for e in h.events if e["type"] == "session_grants"][-1]["grants"] == []


async def test_session_allow_is_not_offered_for_other_change_tools_or_when_off(make, loop_tools, settings):
    h = make(FakeLLM("change", [[ToolCall("scale_thing", {"name": "web", "replicas": 2})], ["ok"]]))
    h.decisions = [("approve", None)]
    await h.agent.handle("scale")
    assert next(e for e in h.events if e["type"] == "approval_request")["can_session_allow"] is False

    settings.approvals.session_allow = False
    h2 = make(FakeLLM("change", [[ToolCall("write_thing", {"project": "shop"})], ["ok"]]))
    h2.decisions = [("approve", None)]
    await h2.agent.handle("write")
    assert next(e for e in h2.events if e["type"] == "approval_request")["can_session_allow"] is False


def test_only_change_tools_can_join_a_session_group(registry):
    from app.tools.registry import Risk
    from tests.conftest import NameArgs

    with pytest.raises(ValueError, match="Only change tools"):
        @registry.tool(risk=Risk.DESTRUCTIVE, summary=lambda a: "x", confirm_field="name",
                       session_group="build-loop", session_scope_field="name")
        def docker_wipe(args: NameArgs) -> dict:
            """Wipe."""
            return {}


def test_the_real_build_loop_tools_share_one_group():
    from app.tools import load_all
    from app.tools.registry import registry as real
    load_all()
    groups = {t.name: t.session_group for t in real.all() if t.session_group}
    assert groups == {"project_write_file": "build-loop", "project_build_image": "build-loop",
                      "project_run_container": "build-loop"}



def test_the_stop_message_keeps_angle_brackets_and_drops_repeats(make):
    h = make(FakeLLM("action", []))
    h.agent._fixes = ["Fix the stray </parameter> tag", "Fix the stray </parameter> tag", "Use `uv sync`"]
    text = h.agent._same_error_twice("ERROR: unknown instruction: </parameter\nsecond line")
    assert text.count("Fix the stray") == 1
    assert "\\</parameter\\>" not in text and "\\</parameter>" in text  # escaped, so Markdown shows it
    assert "> ERROR: unknown instruction: \\</parameter" in text and "> second line" in text
    assert "\\`uv sync\\`" in text
