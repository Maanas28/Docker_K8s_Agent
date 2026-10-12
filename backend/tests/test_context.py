"""The context techniques as pure functions (no agent, no model)."""

import copy

from app.config import ContextSettings, Settings
from app.context import call_keys, compact_result, estimate_tokens, fit_budget, replace_outdated

CFG = ContextSettings(compact_log_lines=5, compact_table_rows=5)


def write_call(path, content):
    return {"function": {"name": "project_write_file", "arguments": {"project": "p", "path": path, "content": content}}}


def read_call(path):
    return {"function": {"name": "project_read_file", "arguments": {"project": "p", "path": path}}}


def turn(call, result):
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_name": call["function"]["name"], "content": result}]


# ---- 1. compact tool output ----


def test_logs_keep_the_tail():
    logs = {"kind": "logs", "title": "x", "text": "\n".join(f"line {i}" for i in range(1, 21))}
    out = compact_result(logs, CFG)
    assert out["text"].splitlines()[0] == "[15 earlier lines omitted; the user sees the full log]"
    assert out["text"].splitlines()[-1] == "line 20" and logs["text"].count("\n") == 19  # input untouched


def test_tables_keep_the_first_rows():
    table = {"kind": "table", "columns": ["a"], "rows": [{"a": i} for i in range(12)]}
    out = compact_result(table, CFG)
    assert len(out["rows"]) == 5 and "5 of 12" in out["note"] and len(table["rows"]) == 12


def test_short_results_are_unchanged():
    small = {"kind": "logs", "title": "x", "text": "a\nb"}
    assert compact_result(small, CFG) is small


# ---- 2. replace outdated outputs ----


def test_newer_write_replaces_older_read_and_attempt():
    history = [
        {"role": "user", "content": "write it"},
        *turn(read_call("Dockerfile"), "FROM old"),
        *turn(write_call("Dockerfile", "FROM v1"), "refused: problems"),
        *turn(write_call("Dockerfile", "FROM v2"), "Wrote Dockerfile"),
    ]
    original = copy.deepcopy(history)
    out, replaced = replace_outdated(history)
    assert replaced == 2
    assert out[2]["content"].startswith("[outdated: an earlier project_read_file for Dockerfile")
    assert out[3]["tool_calls"][0]["function"]["arguments"]["content"].startswith("[content of an earlier attempt")
    assert out[4]["content"].startswith("[outdated")
    assert out[5]["tool_calls"][0]["function"]["arguments"]["content"] == "FROM v2"  # newest kept
    assert out[6]["content"] == "Wrote Dockerfile"
    assert history == original  # the real history is never modified


def test_unrelated_and_single_calls_are_kept():
    history = [*turn(read_call("app/main.py"), "main"), *turn(read_call("app/config.py"), "config")]
    out, replaced = replace_outdated(history)
    assert replaced == 0 and out == history


def test_repeating_the_exact_same_call_replaces_the_older_one():
    call = {"function": {"name": "docker_list_containers", "arguments": {"all": False}}}
    out, replaced = replace_outdated([*turn(call, "old list"), *turn(call, "new list")])
    assert replaced == 1 and out[1]["content"].startswith("[outdated") and out[3]["content"] == "new list"


def test_call_keys():
    assert "file:p:Dockerfile" in call_keys("project_read_file", {"project": "p", "path": "Dockerfile"})
    assert not any(k.startswith("file:") for k in call_keys("docker_list_containers", {}))


# ---- 6. measure and compact ----


def test_fit_budget_shortens_oldest_tool_outputs_first():
    big = "x" * 4000  # ~1000 tokens each
    history = [{"role": "user", "content": "the request"},
               *turn(read_call("a"), big), *turn(read_call("b"), big), *turn(read_call("c"), big)]
    # budget = 2000 * 0.75 = 1500 tokens: only one ~1000-token output can stay
    out, shortened = fit_budget("system", history, None, num_ctx=2000, ratio=0.75)
    assert shortened == 2
    assert out[2]["content"].startswith("[older") and out[4]["content"].startswith("[older")
    assert out[6]["content"] == big  # newest kept
    assert out[0] == {"role": "user", "content": "the request"}  # request never touched


def test_fit_budget_leaves_small_conversations_alone():
    history = [{"role": "user", "content": "hi"}]
    assert fit_budget("system", history, None, 32768, 0.75) == (history, 0)
    assert estimate_tokens("abcd" * 10) == 11


# ---- the config file ----


def test_flags_load_from_a_toml_file(tmp_path, monkeypatch):
    toml = tmp_path / "agent.toml"
    toml.write_text("[context]\nattempt_log = false\ncompact_log_lines = 99\n")

    class FromFile(Settings):
        model_config = {**Settings.model_config, "toml_file": toml}

    s = FromFile()
    assert s.context.attempt_log is False and s.context.compact_log_lines == 99
    assert s.context.replace_outdated_outputs is True  # not in the file → default
    monkeypatch.setenv("AGENT_CONTEXT__COMPACT_LOG_LINES", "7")
    assert FromFile().context.compact_log_lines == 7  # env var beats the file


def test_failed_builds_keep_the_end_of_the_log():
    build = {"kind": "build", "ok": False, "error": "failed to solve", "log": "\n".join(f"#5 {i}" for i in range(20))}
    out = compact_result(build, CFG)
    assert out["log"].splitlines()[-1] == "#5 19" and len(out["log"].splitlines()) == 6
    assert out["error"] == "failed to solve"


def test_successful_builds_drop_the_log():
    build = {"kind": "build", "ok": True, "title": "Built x", "facts": {"Image": "x:1"}, "log": "lots"}
    assert compact_result(build, CFG) == {"kind": "build", "ok": True, "title": "Built x", "facts": {"Image": "x:1"}}


def test_only_the_latest_build_of_a_project_is_kept():
    first = {"function": {"name": "project_build_image", "arguments": {"project": "p"}}}
    second = {"function": {"name": "project_build_image", "arguments": {"project": "p", "no_cache": True}}}
    messages = turn(first, "old failure log") + turn(second, "new failure log")
    out, replaced = replace_outdated(messages)
    assert replaced == 1 and out[1]["content"].startswith("[outdated") and out[3]["content"] == "new failure log"
