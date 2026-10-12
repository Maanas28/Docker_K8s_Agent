import pytest

from app.text_tool_calls import extract_text_tool_calls


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "read_thing", "arguments": {"name": "web"}}',
        'Sure, let me check.\n```json\n{"name": "read_thing", "arguments": {"name": "web"}}\n```',
        '<tool_call>\n{"name": "read_thing", "arguments": {"name": "web"}}\n</tool_call>',
        '{"function": {"name": "read_thing", "arguments": "{\\"name\\": \\"web\\"}"}}',
        'Please call:\n{\n  "name": "read_thing",\n  "parameters": {"name": "web"}\n}',
    ],
)
def test_recovers_calls_in_common_formats(registry, text):
    calls = extract_text_tool_calls(text, registry)
    assert [(c.name, c.arguments, c.from_text) for c in calls] == [("read_thing", {"name": "web"}, True)]


def test_several_calls_in_a_list(registry):
    text = '[{"name": "read_thing", "arguments": {"name": "a"}}, {"name": "read_thing", "arguments": {"name": "b"}}]'
    assert [c.arguments["name"] for c in extract_text_tool_calls(text, registry)] == ["a", "b"]


@pytest.mark.parametrize(
    "text",
    [
        "The container is running fine.",
        '{"name": "my-app", "version": "1.0.0", "scripts": {"start": "node index.js"}}',  # a package.json
        '{"name": "rm_rf", "arguments": {"path": "/"}}',  # not a registered tool
        '{"name": "read_thing", "arguments": "not json"}',
        '{"name": "read_thing", "arguments": {"name": "web"',  # cut off
    ],
)
def test_ignores_text_that_is_not_a_call(registry, text):
    assert extract_text_tool_calls(text, registry) == []


def test_raw_line_breaks_inside_file_content_are_accepted(registry):
    text = '{"name": "read_thing", "arguments": {"name": "FROM python:3.12\nCMD [\\"x\\"]\n"}}'
    calls = extract_text_tool_calls(text, registry)
    assert calls and calls[0].arguments["name"] == 'FROM python:3.12\nCMD ["x"]\n'
