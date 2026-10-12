import asyncio
import json
import subprocess
import sys
import time

import pytest

from app.config import BuildSettings
from app.store import Store
from app.tools import build_tools, code_tools
from app.tools.registry import RunContext, ToolError, registry

# Stands in for the `docker` CLI: records its arguments and behaves according to a "mode" file.
FAKE_DOCKER = f"""#!{sys.executable}
import json, pathlib, sys, time
here = pathlib.Path(__file__).parent
with (here / "calls.jsonl").open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["image", "inspect"]:
    print("sha256:built123", 123_400_000)
    sys.exit(0)
mode = (here / "mode").read_text().strip()
if mode == "ok":
    print("#1 [1/2] FROM docker.io/library/python:3.12-slim")
    print("#2 \\x1b[32mDONE 0.1s\\x1b[0m")
elif mode == "fail":
    for i in range(100):
        print(f"#5 0.{{i}} collecting line {{i}}")
    print("#5 0.9 ERROR: Could not open requirements file: requirements.txt")
    print('ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully: exit code: 1', file=sys.stderr)
    sys.exit(1)
elif mode == "notfound":
    print('ERROR: failed to solve: failed to compute cache key: failed to calculate checksum of ref abc::def: "/uv.lock": not found', file=sys.stderr)
    print('ERROR: failed to solve: failed to compute cache key: failed to calculate checksum of ref abc::ghi: "/requirements.txt": not found', file=sys.stderr)
    sys.exit(1)
elif mode == "nocommand":
    print("#7 0.212 /bin/sh: 1: uv: not found")
    print("#8 1.402 E: Unable to locate package uv")
    print('ERROR: failed to solve: process "/bin/sh -c uv pip install -U uv" did not complete successfully: exit code: 127', file=sys.stderr)
    sys.exit(1)
elif mode == "secret":
    print("#3 0.1 token=" + "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789")  # split: not a real token
elif mode == "hang":
    print("#1 started", flush=True)
    time.sleep(30)
"""

DOCKERFILE = "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nRUN pip install uv && uv sync --frozen\nCMD [\"python\", \"-m\", \"app\"]\n"


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "shop-api"
    (root / "app").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "shop-api"\n')
    (root / "app" / "__main__.py").write_text("print('hi')\n")
    (root / ".env").write_text("APP_API_KEY=fake-test-key-123\n")
    (root / ".dockerignore").write_text(".env\n.git\n.venv\n")
    (root / "Dockerfile").write_text(DOCKERFILE)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    store = Store("sqlite://")
    store.add_project("shop-api", str(root.resolve()))
    code_tools.configure(store)
    build_tools.configure(BuildSettings())
    build_tools._last_failed.clear()
    build_tools._failures.clear()
    return root


@pytest.fixture
def docker(tmp_path, monkeypatch):
    """The fake docker CLI. Set its behaviour with docker.mode("ok" | "fail" | "hang" | "secret")."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "docker"
    script.write_text(FAKE_DOCKER)
    script.chmod(0o755)
    monkeypatch.setattr(build_tools, "docker_bin", lambda: str(script))

    class Fake:
        def mode(self, value):
            (bin_dir / "mode").write_text(value)

        def calls(self):
            path = bin_dir / "calls.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    fake = Fake()
    fake.mode("ok")
    return fake


def preview(**kwargs):
    tool = registry.get("project_build_image")
    return tool.preview(tool.args_model.model_validate({"project": "shop-api", **kwargs}))


async def build(prev=None, ctx=None, **kwargs):
    tool = registry.get("project_build_image")
    args = tool.args_model.model_validate({"project": "shop-api", **kwargs})
    return await tool.call(args, prev or tool.preview(args), ctx)


def collecting_ctx():
    lines: list[str] = []

    async def progress(line):
        lines.append(line)

    return RunContext(progress=progress), lines


# ---- pre-check (approval card) -----------------------------------------------------------------


def test_card_shows_tags_base_image_and_run_commands(project):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project, capture_output=True, text=True).stdout.strip()
    p = preview()
    assert p["facts"]["Image"] == f"shop-api:v1-{commit}"
    assert p["facts"]["Also tagged"] == "shop-api:latest"
    assert p["facts"]["Base image"] == "python:3.12-slim"
    assert p["commands"] == ["RUN pip install uv && uv sync --frozen"]
    assert p["warnings"] == []


def test_uncommitted_changes_make_a_dirty_tag(project):
    (project / "Dockerfile").write_text(DOCKERFILE + "EXPOSE 8080\n")
    p = preview()
    assert p["facts"]["Image"].endswith("-dirty")
    assert any("uncommitted changes" in w for w in p["warnings"])


def test_outside_git_the_tag_is_a_timestamp(project):
    import shutil
    shutil.rmtree(project / ".git")
    assert preview()["facts"]["Image"] == "shop-api:v1"


def test_no_dockerfile_is_refused(project):
    (project / "Dockerfile").unlink()
    with pytest.raises(ToolError, match="no Dockerfile"):
        preview()


def test_secret_files_in_the_build_context_are_refused(project):
    (project / ".dockerignore").write_text(".git\n")
    with pytest.raises(ToolError, match=r"secret files would be sent to the build: \.env"):
        preview()


def test_dependency_folders_are_a_warning_not_a_secret(project):
    pkg = project / ".venv" / "lib" / "certifi"
    pkg.mkdir(parents=True)
    (pkg / "cacert.pem").write_text("public CA bundle")
    (project / ".dockerignore").write_text(".env\n.git\n")
    p = preview()  # not refused: a package's .pem isn't the project's secret
    assert any(".venv" in w for w in p["warnings"])


def test_image_names_are_made_valid():
    assert build_tools._repo_name("Shop..API_v2-") == "shop-api-v2"


# ---- running the build -------------------------------------------------------------------------


async def test_successful_build_streams_output_and_tags_the_image(project, docker):
    ctx, lines = collecting_ctx()
    result = await build(ctx=ctx)
    assert result["kind"] == "build" and result["ok"]
    assert result["facts"]["Size"] == "123.4 MB"
    assert lines == ["#1 [1/2] FROM docker.io/library/python:3.12-slim", "#2 DONE 0.1s"]  # colour codes removed
    cmd = docker.calls()[0]
    assert cmd[:2] == ["build", "--progress=plain"]
    assert cmd.count("--tag") == 2 and "shop-api:latest" in cmd
    assert any(a.startswith("org.opencontainers.image.revision=") for a in cmd)
    assert cmd[-1] == str(project.resolve())
    assert "--no-cache" not in cmd


async def test_failed_build_returns_the_log_and_blocks_an_identical_retry(project, docker):
    docker.mode("fail")
    with pytest.raises(ToolError) as err:
        await build()
    assert "failed to solve" in str(err.value)
    result = err.value.result
    assert result["kind"] == "build" and not result["ok"]
    assert "Could not open requirements file" in result["log"]
    assert list(result)[:4] == ["kind", "ok", "title", "error"]  # error before the long log

    with pytest.raises(ToolError, match="exactly the same as in the build that just failed"):
        preview()
    assert preview(no_cache=True)  # a network failure may be retried without the cache
    (project / "Dockerfile").write_text(DOCKERFILE.replace("uv sync --frozen", "uv sync"))
    assert preview()  # changed files may be built again


async def test_a_success_clears_the_failure_memory(project, docker):
    docker.mode("fail")
    with pytest.raises(ToolError):
        await build()
    docker.mode("ok")
    await build(no_cache=True)
    assert "--no-cache" in docker.calls()[-2]  # [-1] is the image size lookup
    assert preview()  # no longer blocked


async def test_files_changed_after_approval_are_not_built(project, docker):
    approved = preview()
    (project / "Dockerfile").write_text(DOCKERFILE + "# edited\n")
    with pytest.raises(ToolError, match="changed after you approved"):
        await build(prev=approved)
    assert docker.calls() == []


async def test_stop_cancels_a_running_build(project, docker):
    docker.mode("hang")
    ctx, lines = collecting_ctx()

    async def press_stop():
        while not lines:
            await asyncio.sleep(0.02)
        ctx.stopped.set()

    started = time.monotonic()
    stopper = asyncio.create_task(press_stop())
    with pytest.raises(ToolError, match="pressed Stop") as err:
        await build(ctx=ctx)
    await stopper
    assert time.monotonic() - started < 10
    assert err.value.result["title"].endswith("cancelled")
    assert preview()  # a stop says nothing about the files, so rebuilding isn't blocked


async def test_builds_time_out(project, docker):
    docker.mode("hang")
    build_tools.configure(BuildSettings.model_construct(**{**BuildSettings().model_dump(), "timeout_s": 0.5}))
    with pytest.raises(ToolError, match="timed out after 0.5 s"):
        await build()


async def test_secret_values_in_build_output_are_masked(project, docker):
    docker.mode("secret")
    ctx, lines = collecting_ctx()
    result = await build(ctx=ctx)
    assert "ghp_" not in result["log"] and "ghp_" not in "".join(lines)


# ---- explaining failures (fix 1), no_cache once (fix 3), attempts (fix 4) ------------------------


async def test_not_found_errors_say_whether_the_file_is_excluded_or_missing(project, docker):
    (project / "uv.lock").write_text("")
    (project / ".dockerignore").write_text(".env\n.git\n.venv\nuv.lock\n")
    docker.mode("notfound")
    with pytest.raises(ToolError) as err:
        await build()
    message = str(err.value)
    assert "'uv.lock' exists in the project, but .dockerignore excludes it (pattern `uv.lock`)" in message
    assert "'requirements.txt' doesn't exist in the project" in message
    assert len(err.value.result["diagnosis"]) == 2


def test_negated_patterns_re_include_a_file():
    assert build_tools._excluding_pattern("uv.lock", ["*.lock", "!uv.lock"]) is None
    assert build_tools._excluding_pattern("uv.lock", ["!uv.lock", "*.lock"]) == "*.lock"


async def test_no_cache_may_skip_the_unchanged_guard_only_once(project, docker):
    docker.mode("fail")
    with pytest.raises(ToolError):
        await build(no_cache=True)
    with pytest.raises(ToolError, match="already retried without the cache"):
        preview(no_cache=True)


async def test_attempts_are_counted_until_a_success(project, docker):
    tool = registry.get("project_build_image")
    args = tool.args_model.model_validate({"project": "shop-api"})
    assert tool.summarize(args) == "Build a Docker image for 'shop-api'"
    docker.mode("fail")
    with pytest.raises(ToolError):
        await build()
    (project / "Dockerfile").write_text(DOCKERFILE + "# fixed\n")
    assert tool.summarize(args) == "Build a Docker image for 'shop-api' (attempt 2)"
    assert preview()["facts"]["Attempt"] == "2 (1 failed so far)"
    docker.mode("ok")
    await build()
    assert tool.summarize(args) == "Build a Docker image for 'shop-api'"


async def test_missing_programs_and_apt_packages_are_explained(project, docker):
    docker.mode("nocommand")
    with pytest.raises(ToolError) as err:
        await build()
    message = str(err.value)
    assert "The program `uv` isn't installed in the image at that step (exit code 127" in message
    assert "base image (python:3.12-slim)" in message
    assert "apt-get has no package called `uv`" in message


async def test_each_successful_build_is_recorded_as_the_next_version(project, docker):
    await build()
    store = code_tools.get_store()
    first = store.versions("shop-api")[0]
    assert (first.number, first.image_id, first.size_bytes) == (1, "sha256:built123", 123_400_000)
    assert first.tag.startswith("shop-api:v1-") and first.healthy is None
    assert preview()["facts"]["Image"].startswith("shop-api:v2-")  # earlier builds keep their own tag
    docker.mode("fail")
    with pytest.raises(ToolError):
        await build(no_cache=True)
    assert len(store.versions("shop-api")) == 1  # failed builds aren't versions
