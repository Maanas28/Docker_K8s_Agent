"""Build a Docker image from a registered project (Step 3).

The build runs the real `docker build` CLI (BuildKit) as a subprocess, so "it builds here" means
it builds in the user's terminal and in CI too. docker-py was not used: it only drives the legacy
builder (see docs/LIMITATIONS.md). Output is streamed live to the user's step row; Stop and a
timeout kill the build. A failed build's log goes back to the model, which fixes the Dockerfile
with project_write_file and builds again (the build-and-fix loop is the normal agent loop).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

from docker.utils.build import PatternMatcher
from pydantic import Field

from app.config import BuildSettings
from app.dockerfile_check import parse
from app.redact import redact
from app.store import ImageVersion
from app.tools.code_tools import _workspace, get_store
from app.tools.registry import Risk, RunContext, ToolArgs, ToolError, registry
from app.workspace import Workspace

_settings = BuildSettings()

# Each project's last failed build: (hash of Dockerfile + .dockerignore, whether it ran without the
# cache). Building the same inputs again would fail the same way, so the pre-check refuses it; one
# retry with no_cache=true is allowed for temporary failures. In memory: a restart forgets it.
_last_failed: dict[str, tuple[str, bool]] = {}
# Failed builds per project since its last successful one, shown as "attempt N" to the user.
_failures: dict[str, int] = {}

# Where Docker Desktop, Homebrew and OrbStack put the CLI, for when PATH doesn't include it.
_DOCKER_PATHS = ("/usr/local/bin/docker", "/opt/homebrew/bin/docker", str(Path.home() / ".orbstack/bin/docker"))
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def configure(settings: BuildSettings) -> None:
    global _settings
    _settings = settings


def docker_bin() -> str:
    found = shutil.which("docker") or next((p for p in _DOCKER_PATHS if os.access(p, os.X_OK)), None)
    if not found:
        raise ToolError("Can't find the docker command. Is Docker Desktop installed?")
    return found


class BuildArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    no_cache: bool = Field(False, description="Rebuild every step from scratch. Only if the user asks, or to retry a failure caused by the network")


def _inputs(ws: Workspace) -> tuple[str | None, str]:
    """The Dockerfile's text and a hash of everything the build depends on that we control."""
    dockerfile = ws.read_raw("Dockerfile")
    ignore = ws.read_raw(".dockerignore")
    digest = hashlib.sha256(f"{dockerfile}\0{ignore}".encode()).hexdigest()
    return dockerfile, digest


def _repo_name(project: str) -> str:
    """A valid image name: lowercase letters and digits separated by single dashes."""
    return re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-") or "project"


def image_tags(ws: Workspace, number: int) -> tuple[list[str], str | None]:
    """(tags, git commit). <name>:v<number>-<commit>, plus -dirty if there are uncommitted changes
    (the image is then not exactly that commit), or <name>:v<number> outside git; and <name>:latest.

    The build number makes every tag unique, so earlier builds keep their names and can be rolled
    back to (with only <commit>-dirty, every rebuild of uncommitted work would take over the tag).
    """
    repo = _repo_name(ws.name)
    git = ws.git_info()
    commit = git["commit"] if git and not git["commit"].startswith("(") else None
    version = f"v{number}"
    if commit:
        version += f"-{commit}" + ("" if git["uncommitted changes"] == "none" else "-dirty")
    return [f"{repo}:{version}", f"{repo}:latest"], commit


def _attempt(project: str) -> int:
    return _failures.get(project, 0) + 1


def _build_summary(args: BuildArgs) -> str:
    attempt = _attempt(args.project)
    return (
        f"Build a Docker image for '{args.project}'"
        + (f" (attempt {attempt})" if attempt > 1 else "")
        + (" (no cache)" if args.no_cache else "")
    )


# BuildKit's message for a COPY/ADD source that isn't in the build context: `"/uv.lock": not found`.
_NOT_FOUND = re.compile(r'"/?([^"\n]+)": not found')
# The shell's message for a program that isn't installed: `/bin/sh: 1: uv: not found` (exit code 127).
_NO_COMMAND = re.compile(r"/bin/(?:ba)?sh: (?:\d+: )?([^\s:]+): (?:command )?not found")
# apt-get's message for a package name it doesn't know.
_NO_APT_PACKAGE = re.compile(r"E: Unable to locate package (\S+)")


def diagnose(ws: Workspace, log: str) -> list[str]:
    """Facts that explain a failed build, in the user's terms.

    Docker only says a file was "not found"; it doesn't say whether the file is missing from the
    project or was excluded by .dockerignore. Without that, a small model keeps rewriting the
    Dockerfile instead of fixing the .dockerignore.
    """
    patterns = [
        line.strip() for line in (ws.read_raw(".dockerignore") or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    notes: list[str] = []
    for command in dict.fromkeys(_NO_COMMAND.findall(log)):
        base = _base_images(ws)
        notes.append(
            f"The program `{command}` isn't installed in the image at that step (exit code 127 means "
            f"\"command not found\"). A RUN step can only use programs that the base image"
            f"{f' ({base})' if base else ''} already has, or that an earlier step installed."
        )
    for package in dict.fromkeys(_NO_APT_PACKAGE.findall(log)):
        notes.append(
            f"apt-get has no package called `{package}` in this base image's package lists, so it "
            "can't be installed with apt-get. It has to be installed another way."
        )
    for path in dict.fromkeys(_NOT_FOUND.findall(log)):  # unique, in order
        try:
            target = ws.resolve(path)
        except ToolError:
            continue  # not a path inside the project
        if not target.exists():
            notes.append(f"'{path}' doesn't exist in the project, so it can't be copied. Copy only files that exist.")
            continue
        culprit = _excluding_pattern(path, patterns)
        if culprit:
            notes.append(
                f"'{path}' exists in the project, but .dockerignore excludes it (pattern `{culprit}`), so "
                f"the build can't see it. Remove that pattern from .dockerignore, or don't copy '{path}'."
            )
        else:
            notes.append(f"'{path}' exists and isn't excluded by .dockerignore; check the COPY path and the build stage.")
    return notes


def _base_images(ws: Workspace) -> str:
    dockerfile = ws.read_raw("Dockerfile") or ""
    return ", ".join(s.args.split()[0] for s in parse(dockerfile) if s.command == "FROM" and s.args)


def _excluding_pattern(path: str, patterns: list[str]) -> str | None:
    """The .dockerignore pattern that excludes `path` (the last match wins, `!` re-includes)."""
    culprit = None
    for pattern in patterns:
        negated = pattern.startswith("!")
        if PatternMatcher([pattern.lstrip("!")]).matches(path):
            culprit = None if negated else pattern
    return culprit


def _mb(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB"


def _preview_build(args: BuildArgs) -> dict:
    ws = _workspace(args.project)
    dockerfile, digest = _inputs(ws)
    if dockerfile is None:
        raise ToolError(f"Project '{ws.name}' has no Dockerfile at its root. Write one first with project_write_file.")
    failed_digest, failed_without_cache = _last_failed.get(ws.name, (None, False))
    if failed_digest == digest and (not args.no_cache or failed_without_cache):
        retry = (
            "" if failed_without_cache
            else " Only if the failure was temporary (e.g. a network error) retry once with no_cache=true."
        )
        raise ToolError(
            "The Dockerfile and .dockerignore are exactly the same as in the build that just failed"
            + (" (already retried without the cache)" if failed_without_cache else "")
            + ", so it would fail the same way. Fix what the build error says with project_write_file "
            "first." + retry
        )
    context = ws.build_context()
    if context["secrets"]:
        raise ToolError(
            f"These secret files would be sent to the build: {', '.join(context['secrets'])}. "
            "Exclude them in .dockerignore (project_write_file) before building."
        )

    steps = parse(dockerfile)
    number = get_store().next_version(ws.name)
    tags, commit = image_tags(ws, number)
    facts = {
        "Project": ws.name,
        **({"Attempt": f"{_attempt(ws.name)} ({_attempt(ws.name) - 1} failed so far)"} if _failures.get(ws.name) else {}),
        "Image": tags[0],
        "Also tagged": tags[1],
        "Base image": ", ".join(s.args.split()[0] for s in steps if s.command == "FROM" and s.args) or "none",
        "Sent to the build": f"{context['files']} files, {_mb(context['bytes'])}",
    }
    if args.no_cache:
        facts["Cache"] = "off (every step is rebuilt)"
    warnings = []
    if tags[0].endswith("-dirty"):
        warnings.append("The project has uncommitted changes, so the image isn't exactly commit " f"{commit}.")
    if context["dependency_dirs"]:
        warnings.append(
            f"These folders would be sent to the build: {', '.join(context['dependency_dirs'])}. They "
            "are usually excluded in .dockerignore: dependencies are installed inside the image, and "
            "editor or build folders don't belong in it."
        )
    if not context["has_dockerignore"]:
        warnings.append("There's no .dockerignore, so everything in the folder is sent to the build.")
    if context["bytes"] > _settings.warn_context_mb * 1_000_000:
        warnings.append(
            f"{_mb(context['bytes'])} would be sent to the build. A dependency folder (.venv, "
            "node_modules) is probably missing from .dockerignore."
        )
    return {
        "facts": facts,
        "warnings": warnings,
        # Shown on the card: the commands the build will run on this machine.
        "commands": [s.raw for s in steps if s.command == "RUN"],
        "build": {"tags": tags, "commit": commit, "hash": digest, "number": number},
    }


class _Log:
    """Build output: cleaned, secret values masked, bounded in size, and streamed to the user."""

    def __init__(self, ctx: RunContext | None) -> None:
        self.ctx = ctx
        self.lines: list[str] = []
        self.dropped = 0

    async def add(self, raw: bytes) -> None:
        line, _ = redact(_ANSI.sub("", raw.decode("utf-8", errors="replace")).rstrip())
        self.lines.append(line)
        if len(self.lines) > _settings.max_log_lines:
            del self.lines[0]
            self.dropped += 1
        if self.ctx:
            await self.ctx.progress(line)

    def text(self) -> str:
        head = f"[{self.dropped} earlier lines dropped]\n" if self.dropped else ""
        return head + "\n".join(self.lines)

    def error_line(self) -> str | None:
        """BuildKit's summary of what failed, e.g. 'ERROR: failed to solve: process … exit code: 1'."""
        for line in reversed(self.lines):
            if "failed to solve" in line or line.lstrip().startswith("ERROR"):
                return line.strip()
        return None


async def _run(cmd: list[str], log: _Log, ctx: RunContext | None, timeout: float) -> str:
    """Run `cmd`, streaming its output into `log`. Returns "ok", "failed", "stopped" or "timeout"."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # BuildKit writes its progress to stderr
        stdin=asyncio.subprocess.DEVNULL,
        limit=1 << 20,  # a single long line (e.g. a pip resolver dump) must not break reading
        env={**os.environ, "DOCKER_BUILDKIT": "1", "BUILDKIT_PROGRESS": "plain", "NO_COLOR": "1"},
    )

    async def read() -> None:
        assert proc.stdout is not None
        while line := await proc.stdout.readline():
            await log.add(line)
        await proc.wait()

    reader = asyncio.create_task(read())
    stopper = asyncio.create_task(ctx.stopped.wait()) if ctx else None
    try:
        done, _ = await asyncio.wait({reader, *([stopper] if stopper else [])}, timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if reader in done:
            reader.result()  # re-raise a reading error, if any
            return "ok" if proc.returncode == 0 else "failed"
        return "stopped" if stopper in done else "timeout"
    finally:
        if stopper:
            stopper.cancel()
        if proc.returncode is None:
            # Ending the CLI closes its session with BuildKit, which cancels the build.
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if not reader.done():
            reader.cancel()


async def _image_info(tag: str) -> tuple[str, int] | None:
    """(image id, size in bytes) of a freshly built tag."""
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin(), "image", "inspect", "--format", "{{.Id}} {{.Size}}", tag,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 10)
        image_id, size = out.decode().split()
        return (image_id, int(size)) if proc.returncode == 0 else None
    except (OSError, ValueError, asyncio.TimeoutError, ToolError):
        return None


@registry.tool(
    risk=Risk.CHANGE,
    summary=_build_summary,
    platform="project",
    preview=_preview_build,
    session_group="build-loop",
    session_scope_field="project",
)
async def project_build_image(args: BuildArgs, preview: dict, ctx: RunContext | None = None) -> dict:
    """Build a Docker image from a project's Dockerfile (tagged with a version number and its git commit, e.g. v4-e54f34f, plus latest; every build is kept for rollback). Output streams to the user. If the build fails you get the end of the log: fix the Dockerfile or .dockerignore with project_write_file, then build again."""
    ws = _workspace(args.project)
    _, digest = _inputs(ws)
    if digest != preview["build"]["hash"]:
        raise ToolError("The Dockerfile or .dockerignore changed after you approved the build, so I didn't build. Ask again to see the new card.")

    tags, commit = preview["build"]["tags"], preview["build"]["commit"]
    cmd = [docker_bin(), "build", "--progress=plain", "--file", str(ws.root / "Dockerfile")]
    for tag in tags:
        cmd += ["--tag", tag]
    cmd += ["--label", f"org.opencontainers.image.title={ws.name}"]
    if commit:
        cmd += ["--label", f"org.opencontainers.image.revision={commit}"]
    if args.no_cache:
        cmd.append("--no-cache")
    cmd.append(str(ws.root))

    log = _Log(ctx)
    started = time.monotonic()
    outcome = await _run(cmd, log, ctx, _settings.timeout_s)
    took = f"{time.monotonic() - started:.0f} s"

    if outcome == "ok":
        _last_failed.pop(ws.name, None)
        _failures.pop(ws.name, None)
        facts = {"Image": tags[0], "Also tagged": tags[1], "Took": took}
        info = await _image_info(tags[0])
        if info:
            facts["Size"] = _mb(info[1])
            # The version history (project_versions, rollback, cleanup) starts here.
            get_store().add_version(ImageVersion(
                project=ws.name, number=preview["build"]["number"], tag=tags[0], image_id=info[0],
                commit=commit, dirty=tags[0].endswith("-dirty"), size_bytes=info[1],
            ))
        return {"kind": "build", "ok": True, "title": f"Built {tags[0]}", "facts": facts, "log": log.text()}

    reason = {
        "stopped": "Build cancelled because you pressed Stop. No image was tagged.",
        "timeout": f"Build timed out after {_settings.timeout_s} s and was cancelled. No image was tagged.",
    }.get(outcome)
    diagnosis: list[str] = []
    if reason is None:
        # Only real failures: a stop or timeout says nothing about the files.
        _last_failed[ws.name] = (digest, args.no_cache)
        _failures[ws.name] = _failures.get(ws.name, 0) + 1
        diagnosis = diagnose(ws, log.text())
        reason = f"The build failed: {log.error_line() or 'docker build exited with an error'}"
        if diagnosis:
            reason += "\nWhy: " + " ".join(diagnosis)
    # "error" comes before "log" so it survives even if the text sent to the model is cut short.
    raise ToolError(reason, result={
        "kind": "build", "ok": False,
        "title": f"Build of {tags[0]} {({'stopped': 'cancelled', 'timeout': 'timed out'}).get(outcome, 'failed')}",
        "error": reason, "diagnosis": diagnosis, "facts": {"Took": took}, "log": log.text(),
    })
