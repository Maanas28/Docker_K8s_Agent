"""Run a project's image as a container and check that the app answers (Step 4).

The model only names the project. Code decides everything else: the image (<project>:latest from
project_build_image), the container name, the port (from the image's EXPOSE, published on
127.0.0.1 only) and the settings (the project's .env, whose values only Docker sees). A container
this agent started earlier for the project is replaced; any other container is never touched.

After starting, the tool waits for an HTTP answer while streaming the container's logs. If the app
exits or doesn't answer, the model gets the exit code, the end of the logs and facts explaining
them, then fixes the Dockerfile, rebuilds and runs again (the normal agent loop).
"""

from __future__ import annotations

import asyncio
import re
import socket
import time
from dataclasses import dataclass
from typing import Any

import httpx
from docker.errors import NotFound
from pydantic import Field

from app.config import RunSettings
from app.redact import MASK, redact
from app.tools import docker_tools
from app.tools.build_tools import _repo_name
from app.store import ImageVersion
from app.tools.code_tools import _workspace, get_store
from app.tools.registry import Risk, RunContext, ToolArgs, ToolError, registry
from app.workspace import Workspace, env_values

_settings = RunSettings()

# Marks containers this agent created: (label -> project). Only those are ever replaced.
LABEL = "ops-agent.project"
# The version (image tag) a container runs, so rollback knows what "previous" means.
VERSION_LABEL = "ops-agent.version"
LOG_TAIL = 60


def configure(settings: RunSettings) -> None:
    global _settings
    _settings = settings


class RunArgs(ToolArgs):
    project: str = Field(description="Name of a registered project whose image was built with project_build_image")
    version: str | None = Field(
        None,
        description='Which build to run. Leave empty for the newest build. "previous" = roll back to the '
                    'newest version that passed a health check and is older than the one running. Or a '
                    'version from project_versions, e.g. "v3".',
    )


def _env(ws: Workspace) -> dict[str, str]:
    path = ws.root / ".env"
    return env_values(path) if _settings.pass_env_file and path.is_file() else {}


def port_free(port: int, address: str) -> bool:
    """Nothing listens on it and we could bind it (Docker Desktop's port forwarders count as taken)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        if probe.connect_ex((address, port)) == 0:
            return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((address, port))
        return True
    except OSError:
        return False


def _existing(client, name: str):
    found = client.containers.list(all=True, filters={"name": f"^/{name}$"})
    return found[0] if found else None


def _version_number(tag: str | None) -> int | None:
    match = re.search(r":v(\d+)\b", tag or "")
    return int(match.group(1)) if match else None


def _running_version(container, versions: list[ImageVersion]) -> ImageVersion | None:
    """Which recorded version a container runs: from its label, or (older containers) its image id."""
    if container is None:
        return None
    number = _version_number(container.labels.get(VERSION_LABEL))
    by_label = next((v for v in versions if v.number == number), None)
    image_id = (container.attrs.get("Image") or "") if hasattr(container, "attrs") else ""
    return by_label or next((v for v in versions if image_id and v.image_id == image_id), None)


def _resolve(ws: Workspace, client, version: str | None, existing) -> tuple[str, ImageVersion | None, Any]:
    """(image reference, its version record, the image) for what the model asked to run.

    Resolved by code against the recorded versions, so a made-up version can't be run.
    """
    repo = _repo_name(ws.name)
    versions = get_store().versions(ws.name)
    wanted = (version or "").strip().lower()

    def image(ref: str):
        try:
            return client.images.get(ref)
        except NotFound:
            return None

    if wanted in ("", "latest", "newest"):
        img = image(f"{repo}:latest")
        if img is None:
            raise ToolError(f"There's no image {repo}:latest yet. Build it first with project_build_image.")
        record = next((v for v in versions if v.image_id == img.id), None)
        return (record.tag if record else f"{repo}:latest"), record, img

    if wanted in ("previous", "prev", "last", "rollback"):
        current = _running_version(existing, versions)
        for v in versions:  # newest first
            if v.healthy and (current is None or v.number < current.number) and (img := image(v.tag)):
                return v.tag, v, img
        below = f" older than {current.tag.split(':')[-1]}" if current else ""
        raise ToolError(f"There's no version{below} that passed a health check and still exists, so there's "
                        "nothing to roll back to. Use project_versions to see what's there.")

    match = re.search(r"(?:^|:)v?(\d+)\b", wanted)  # "v3", "3" or a full tag "sample-project:v3-ab12cd3"
    number = int(match.group(1)) if match else None
    record = next((v for v in versions if v.number == number), None)
    if record is None:
        known = ", ".join(f"v{v.number}" for v in versions) or "none yet"
        raise ToolError(f"There's no version '{version}' of {ws.name}. Versions: {known}.")
    img = image(record.tag)
    if img is None:
        raise ToolError(f"v{record.number} was removed (e.g. by a cleanup), so it can't be run any more.")
    return record.tag, record, img


def _free_port(start: int, skip: set[int]) -> int | None:
    return next((p for p in range(start, start + 50) if p not in skip and port_free(p, _settings.bind_address)), None)


def _preview_run(args: RunArgs) -> dict:
    ws = _workspace(args.project)
    repo = _repo_name(ws.name)
    with docker_tools._docker_errors():
        client = docker_tools.get_client()
        existing = _existing(client, repo)
        if existing is not None and existing.labels.get(LABEL) != ws.name:
            raise ToolError(
                f"A container called '{repo}' already exists and I didn't create it, so I won't touch it. "
                "Remove or rename it yourself, then ask again."
            )
        ref, record, image = _resolve(ws, client, args.version, existing)
        # Leftovers of an interrupted switch (only ever our own).
        stale = [c for c in (_existing(client, f"{repo}-next"), _existing(client, f"{repo}-old"))
                 if c is not None and c.labels.get(LABEL) == ws.name]
    exposed = sorted(int(p.split("/")[0]) for p in (image.attrs.get("Config") or {}).get("ExposedPorts") or {}
                     if p.endswith("/tcp"))
    if not exposed:
        raise ToolError(
            f"The image {ref} doesn't EXPOSE a port, so I don't know where the app listens. Add "
            "EXPOSE <port> to the Dockerfile (the port the app really uses), rebuild, then run it."
        )
    container_port = exposed[0]
    current = _running_version(existing, get_store().versions(ws.name))
    swap = existing is not None and existing.status == "running"
    host_port = _published_port(existing) if existing is not None else None
    if host_port is None:
        host_port = _free_port(container_port, set())
    side_port = _free_port(container_port + 1, {host_port}) if swap else None
    if host_port is None or (swap and side_port is None):
        raise ToolError(f"Ports {container_port}-{container_port + 50} are all in use on this Mac.")

    env = _env(ws)
    label = ref.split(":", 1)[-1]
    facts = {
        "Project": ws.name,
        "Version": f"{ref} ({image.short_id.removeprefix('sha256:')})",
        "Container": repo,
        "Port": f"http://localhost:{host_port} → container port {container_port} (this Mac only)",
        "Settings from .env": ", ".join(sorted(env)) + " (values not shown)" if env else "none",
        "Health check": _health_rule(),
    }
    warnings = []
    if swap:
        now = current.tag.split(":", 1)[-1] if current else "the running version"
        facts["Switch"] = (
            f"{now} → {label}: {label} starts next to it on port {side_port} and is checked there first; "
            f"only if it answers does it move to port {host_port}. If it fails, {now} keeps running."
        )
    elif existing is not None:
        facts["Replaces"] = f"the container '{repo}' I started earlier ({existing.status})"
    if record is not None and record.healthy is not True:
        warnings.append(f"{label} has never passed a health check" + (" (it failed before)." if record.healthy is False else "."))
    if stale:
        warnings.append(f"Leftovers of an interrupted switch will be removed: {', '.join(c.name for c in stale)}.")
    if not swap and host_port != container_port and existing is None:
        warnings.append(f"Port {container_port} is in use on this Mac, so the app will be at port {host_port}.")
    mismatched = [name for name, value in env.items()
                  if name.upper().endswith("PORT") and value.isdigit() and int(value) != container_port]
    if mismatched:
        # Said without the value: it comes from a secret file.
        warnings.append(
            f"{', '.join(mismatched)} in .env is set to a different port than the image EXPOSEs "
            f"({container_port}), so the app may listen somewhere the health check doesn't look."
        )
    return {
        "facts": facts,
        "warnings": warnings,
        "run": {
            "image": ref, "image_id": image.id, "number": record.number if record else None,
            "name": repo, "container_port": container_port, "host_port": host_port,
            "swap": swap, "side_port": side_port,
            "replace_id": existing.id if existing is not None else None,
            "stale_ids": [c.id for c in stale],
        },
    }


def _published_port(container) -> int | None:
    if container is None:
        return None
    for bindings in ((container.attrs.get("HostConfig") or {}).get("PortBindings") or {}).values():
        for b in bindings or []:
            if str(b.get("HostPort", "")).isdigit():
                return int(b["HostPort"])
    return None


class _Logs:
    """The container's output: secrets masked (including the exact .env values), streamed to the user."""

    def __init__(self, ctx: RunContext | None, secret_values: list[str]) -> None:
        self.ctx = ctx
        self.lines: list[str] = []
        # Longest first, so a value that contains another is masked whole.
        self.secrets = sorted((v for v in secret_values if len(v) >= 4), key=len, reverse=True)

    def clean(self, line: str) -> str:
        for value in self.secrets:
            line = line.replace(value, MASK)
        return redact(line)[0]

    async def add(self, raw: str) -> None:
        for line in raw.splitlines():
            line = self.clean(line.rstrip())
            self.lines.append(line)
            if self.ctx:
                await self.ctx.progress(line)

    def tail(self, n: int = LOG_TAIL) -> str:
        return "\n".join(self.lines[-n:]) or "(no output)"


async def _probe(url: str) -> int | None:
    """HTTP status of a GET, or None if nothing answered."""
    try:
        async with httpx.AsyncClient(timeout=2) as http:
            return (await http.get(url)).status_code
    except httpx.HTTPError:
        return None


# Exit codes worth explaining, and what they mean.
_EXIT_CODES = {
    126: "the start command exists but can't be executed (permissions, or not a program)",
    127: "the start command wasn't found in the image (CMD/ENTRYPOINT names a program that isn't installed)",
    137: "the container was killed (out of memory, or stopped from outside)",
    139: "the program crashed (segmentation fault)",
}
_LISTEN = re.compile(r"(?:https?://)?(0\.0\.0\.0|127\.0\.0\.1|localhost|\[::\]|::):(\d{2,5})")
# The last line of a Python traceback, e.g. "RuntimeError: broken on purpose".
_EXCEPTION = re.compile(r"^(?:[\w.]+\.)?\w*(?:Error|Exception): .+$", re.M)
_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def diagnose(logs: str, exit_code: int | None, container_port: int, answered: bool) -> list[str]:
    """Facts that explain why the app didn't come up, from its logs and exit code."""
    notes: list[str] = []
    if exit_code is not None and exit_code in _EXIT_CODES:
        notes.append(f"Exit code {exit_code}: {_EXIT_CODES[exit_code]}.")
    for module in dict.fromkeys(_NO_MODULE.findall(logs)):
        notes.append(f"Python can't import `{module}`: that dependency isn't installed in the image "
                     "(or the start command runs a different Python than the one dependencies went into).")
    raised = _EXCEPTION.findall(logs)
    if raised:
        notes.append(f"The app logged an error: {raised[-1].strip()[:200]}")
    if "Address already in use" in logs:
        notes.append("The app tried to use a port that something else inside the container already uses.")
    if exit_code is None and not answered:
        listens = {(host, int(port)) for host, port in _LISTEN.findall(logs)}
        if any(host in ("127.0.0.1", "localhost") for host, _ in listens):
            notes.append("The app listens on 127.0.0.1 inside the container, which can't be reached from "
                         "outside it. It must listen on 0.0.0.0 (e.g. --host 0.0.0.0 or APP_HOST=0.0.0.0).")
        other = sorted({port for _, port in listens if port != container_port})
        if other:
            notes.append(f"The app says it listens on port {', '.join(map(str, other))}, but the image "
                         f"EXPOSEs {container_port}. Make them the same (EXPOSE, or the app's port setting).")
        if not listens:
            notes.append(f"The app is running but nothing answered on port {container_port}. Check that it "
                         "listens on 0.0.0.0 and on the port in EXPOSE.")
    return notes


def _run_summary(args: RunArgs) -> str:
    wanted = (args.version or "").strip().lower()
    if wanted in ("previous", "prev", "last", "rollback"):
        return f"Roll '{args.project}' back to its previous healthy version"
    if wanted and wanted not in ("latest", "newest"):
        return f"Run {args.version} of '{args.project}' and check it answers"
    return f"Run a container of '{args.project}' and check it answers"


@registry.tool(
    risk=Risk.CHANGE,
    summary=_run_summary,
    platform="project",
    preview=_preview_run,
    session_group="build-loop",
    session_scope_field="project",
)
async def project_run_container(args: RunArgs, preview: dict, ctx: RunContext | None = None) -> dict:
    """Run a built version of the project as a container (port from EXPOSE, reachable from this Mac only, settings from the project's .env) and check that the app answers HTTP requests. If a version is already running, the new one is started next to it and checked first: the old one keeps running unless the new one answers (safe redeploy). version="previous" rolls back to the last healthy version. If it exits or doesn't answer, you get the exit code and the end of its logs: fix the Dockerfile with project_write_file, rebuild with project_build_image, then run again."""
    ws = _workspace(args.project)
    run = preview["run"]
    env = _env(ws)
    logs = _Logs(ctx, list(env.values()))
    store = get_store()
    name, final_url = run["name"], f"http://localhost:{run['host_port']}"
    facts = {"Container": name, "Version": run["image"], "Port": f"{final_url} → container port {run['container_port']}"}

    def record(healthy: bool) -> None:
        if run["number"] is not None:
            store.record_run(ws.name, run["number"], healthy)

    def prepare():
        with docker_tools._docker_errors():
            client = docker_tools.get_client()
            if client.images.get(run["image"]).id != run["image_id"]:
                raise ToolError("The image was rebuilt after you approved this, so I didn't run it. Ask again to see the new card.")
            for stale_id in run["stale_ids"]:
                _remove_ours(client, stale_id, ws.name)
            return client

    client = await asyncio.to_thread(prepare)

    if not run["swap"]:
        # Nothing healthy is running: replace whatever we left (stopped/exited) and start.
        def replace_and_start():
            with docker_tools._docker_errors():
                if run["replace_id"]:
                    _remove_ours(client, run["replace_id"], ws.name)
                if _existing(client, name) is not None:
                    raise ToolError(f"A container called '{name}' appeared after you approved this, so I didn't touch it.")
                return _start(client, run, name, run["host_port"], env, ws.name)

        container = await asyncio.to_thread(replace_and_start)
        health = await _check(container, final_url, logs, ctx)
        if health.answered:
            record(True)
            return _ok(run, final_url, health, facts, logs)
        record(False)
        await _fail(container, run, final_url, logs, ctx, health, facts, stop_on_silence=True)

    # Safe switch: the running version stays up until the new one has answered.
    side_url = f"http://localhost:{run['side_port']}"
    candidate = await asyncio.to_thread(_start, client, run, f"{name}-next", run["side_port"], env, ws.name)
    health = await _check(candidate, side_url, logs, ctx)
    if not health.answered:
        record(False)
        await asyncio.to_thread(_remove_ours, client, candidate.id, ws.name)
        await _fail(candidate, run, side_url, logs, ctx, health, facts, stop_on_silence=False,
                    prefix="The new version didn't come up on its trial port, so nothing was switched: the "
                           "version that was running is still running. ")

    def switch():
        with docker_tools._docker_errors():
            _remove_ours(client, candidate.id, ws.name)
            old = client.containers.get(run["replace_id"])
            old.stop(timeout=5)
            old.rename(f"{name}-old")  # kept until the new one answers on the real port
            return old, _start(client, run, name, run["host_port"], env, ws.name)

    old, container = await asyncio.to_thread(switch)
    health = await _check(container, final_url, logs, ctx)
    if health.answered:
        record(True)
        await asyncio.to_thread(_remove_ours, client, old.id, ws.name)
        return _ok(run, final_url, health, facts, logs, switched=True)

    def restore():
        with docker_tools._docker_errors():
            _remove_ours(client, container.id, ws.name)
            old.rename(name)
            old.start()

    record(False)
    await asyncio.to_thread(restore)
    await _fail(container, run, final_url, logs, ctx, health, facts, stop_on_silence=False,
                prefix="The new version answered on its trial port but not on the real one, so I put the "
                       "previous version back (it's running again). ")


def _start(client, run: dict, name: str, port: int, env: dict[str, str], project: str):
    return client.containers.run(
        run["image"], name=name, detach=True, environment=env,
        ports={f"{run['container_port']}/tcp": (_settings.bind_address, port)},
        labels={LABEL: project, VERSION_LABEL: run["image"], "ops-agent.image": run["image"]},
    )


def _remove_ours(client, container_id: str, project: str) -> None:
    """Remove a container, but only one this agent created for this project (checked right here)."""
    try:
        container = client.containers.get(container_id)
    except NotFound:
        return
    if container.labels.get(LABEL) == project:
        container.remove(force=True)


@dataclass
class Health:
    """Outcome of a health check."""

    answered: tuple[str, int] | None = None  # (path, status) that passed
    exit_code: int | None = None  # the container exited
    unhealthy: tuple[str, int, int, int] | None = None  # (path, status, first attempt, last attempt) of the 5xx run
    attempts: int = 0
    took: str = ""


async def _probe_paths(url: str) -> tuple[str, str | None, int | None]:
    """One attempt: ("up" | "unhealthy" | "none", path, status). See [run] health_paths in agent.toml."""
    paths = _settings.health_paths
    for i, path in enumerate(paths):
        status = await _probe(f"{url}{path}")
        if status is None:
            return "none", path, None  # nothing listens yet; the other paths won't answer either
        if status >= 500:
            return "unhealthy", path, status  # the app's own answer: never fall through to another path
        if status in (404, 405) and i < len(paths) - 1:
            continue  # this path doesn't exist; only the last path may count a 404 as "up"
        return "up", path, status
    return "none", None, None


async def _check(container, url: str, logs: _Logs, ctx: RunContext | None) -> Health:
    """Attempts every health_interval_s, for at most health_timeout_s, streaming the logs meanwhile."""
    started = time.monotonic()
    deadline = started + _settings.health_timeout_s
    result = Health()
    seen = 0  # bytes of the log already shown
    errors: list[tuple[str, int, int]] = []  # consecutive 5xx answers: (path, status, attempt)

    def took() -> str:
        return f"{time.monotonic() - started:.1f} s"

    while time.monotonic() < deadline:
        if ctx is not None and ctx.stopped.is_set():
            break
        result.attempts += 1
        state = await asyncio.to_thread(_poll, container)
        await logs.add(state["logs"][seen:])
        seen = len(state["logs"])
        if state["status"] in ("exited", "dead"):
            result.exit_code, result.took = state["exit_code"], took()
            return result
        verdict, path, status = await _probe_paths(url)
        if verdict == "up":
            result.answered, result.took = (path, status), took()
            return result
        if verdict == "unhealthy":
            errors.append((path, status, result.attempts))
            if len(errors) >= _settings.health_failure_threshold:
                result.unhealthy = (path, status, errors[0][2], result.attempts)
                result.took = took()
                return result
        else:
            errors.clear()  # only errors IN A ROW count
        await asyncio.sleep(max(0.0, min(_settings.health_interval_s, deadline - time.monotonic())))
    final = await asyncio.to_thread(_poll, container)
    await logs.add(final["logs"][seen:])
    if final["status"] in ("exited", "dead"):
        result.exit_code = final["exit_code"]
    if errors and result.unhealthy is None:
        path, status, first = errors[0]
        result.unhealthy = (path, status, first, errors[-1][2])
    result.took = took()
    return result


def _health_rule() -> str:
    attempts = max(1, int(_settings.health_timeout_s / _settings.health_interval_s))
    return (f"GET {' → '.join(_settings.health_paths)} every {_settings.health_interval_s:g} s, up to "
            f"{attempts} attempts; {_settings.health_failure_threshold} error answers (5xx) in a row fail it")


def _ok(run: dict, url: str, health: Health, facts: dict, logs: _Logs, switched: bool = False) -> dict:
    path, status = health.answered  # type: ignore[misc]
    version = run["image"].split(":", 1)[-1]
    return {
        "kind": "container", "ok": True, "url": url,
        "title": f"{version} is live at {url}" if switched else f"Running {run['name']} ({version}) at {url}",
        "facts": {**facts, "Answered": f"HTTP {status} on {path}, attempt {health.attempts}, after {health.took}",
                  **({"Switch": "done; the previous version was removed (its image is kept for rollback)"} if switched else {})},
        "log": logs.tail(200),
    }


async def _fail(container, run: dict, url: str, logs: _Logs, ctx: RunContext | None, health: Health,
                facts: dict, *, stop_on_silence: bool, prefix: str = "") -> None:
    exit_code = health.exit_code
    if ctx is not None and ctx.stopped.is_set():
        reason = prefix + "Stopped by you while waiting for the app to answer."
        if stop_on_silence:
            reason += f" The container '{run['name']}' is still running."
        diagnosis: list[str] = []
    else:
        diagnosis = diagnose("\n".join(logs.lines), exit_code, run["container_port"],
                             answered=health.unhealthy is not None)
        if exit_code is not None:
            reason = prefix + f"The container stopped right away (exit code {exit_code})."
        elif health.unhealthy is not None:
            path, status, first, last = health.unhealthy
            count = last - first + 1
            reason = prefix + (f"The app is up, but {path} answered HTTP {status} {count} times in a row "
                               f"(attempts {first}–{last} of the health check), so it reports itself unhealthy.")
        else:
            reason = prefix + (f"The container was running, but nothing answered on {url} in "
                               f"{health.attempts} attempts ({_settings.health_timeout_s} s).")
        if exit_code is None and stop_on_silence:
            await asyncio.to_thread(_stop, container)  # frees the port for the next attempt
            reason += " I stopped it."
        if diagnosis:
            reason += "\nWhy: " + " ".join(diagnosis)
    raise ToolError(reason, result={
        "kind": "container", "ok": False, "title": f"{run['image'].split(':', 1)[-1]} didn't come up",
        "error": reason, "diagnosis": diagnosis, "url": url,
        "facts": {**facts, "Health check": f"{health.attempts} attempts in {health.took}"}, "log": logs.tail(),
    })


def _poll(container) -> dict[str, Any]:
    """Status, exit code and the whole log so far (the caller shows only the new part)."""
    with docker_tools._docker_errors():
        container.reload()
        raw = container.logs(stdout=True, stderr=True) or b""
    state = container.attrs.get("State") or {}
    return {"status": container.status, "exit_code": state.get("ExitCode"),
            "logs": raw.decode("utf-8", errors="replace")}


def _stop(container) -> None:
    try:
        container.stop(timeout=3)
    except Exception:  # noqa: BLE001  (already gone is fine)
        pass
