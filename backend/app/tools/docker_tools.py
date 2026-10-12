"""Docker tools: read-only inspection (Phase 1) and container/image actions (Phase 2)."""

from __future__ import annotations

import difflib
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import docker
from docker.errors import APIError, DockerException, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from docker.utils import parse_repository_tag
from pydantic import Field

from app.tools.registry import Risk, ToolArgs, ToolError, registry

_client: docker.DockerClient | None = None

# Docker Desktop, Colima and OrbStack don't always expose /var/run/docker.sock.
_SOCKETS = (
    Path.home() / ".docker/run/docker.sock",
    Path.home() / ".colima/default/docker.sock",
    Path.home() / ".orbstack/run/docker.sock",
)


def get_client() -> docker.DockerClient:
    global _client
    if _client is not None:
        return _client
    candidates: list[str | None] = [None]  # None = DOCKER_HOST / default socket
    candidates += [f"unix://{p}" for p in _SOCKETS if p.exists()]
    for url in candidates:
        try:
            client = docker.from_env() if url is None else docker.DockerClient(base_url=url)
            client.ping()
        except DockerException:
            continue
        _client = client
        return client
    raise ToolError("Can't reach Docker. Is Docker Desktop running?")


def ping() -> bool:
    try:
        get_client().ping()
        return True
    except (ToolError, DockerException):
        _reset()
        return False


def _reset() -> None:
    global _client
    _client = None


@contextmanager
def _docker_errors() -> Iterator[None]:
    try:
        yield
    except APIError as e:
        # The daemon answered, so the connection is fine; just report what it said.
        raise ToolError(f"Docker refused: {e.explanation or e}") from e
    except DockerException as e:
        _reset()
        raise ToolError(f"Docker error: {getattr(e, 'explanation', None) or e}") from e


def _names(containers: list[Container]) -> list[str]:
    return [c.attrs["Names"][0].lstrip("/") for c in containers if c.attrs.get("Names")]


def _find_container(ref: str) -> Container:
    client = get_client()
    try:
        return client.containers.get(ref)
    except NotFound:
        names = _names(client.containers.list(all=True, sparse=True))
        close = [n for n in names if ref.lower() in n.lower()]
        close = close or difflib.get_close_matches(ref, names, n=5, cutoff=0.5)
        if close:
            raise ToolError(f"No container named '{ref}'. Did you mean: {', '.join(close)}?")
        existing = ", ".join(names) if names else "none"
        raise ToolError(f"No container named '{ref}'. Existing containers: {existing}.")


def _list_ports(ports: list[dict] | None) -> str:
    out = []
    for p in ports or []:
        private = f"{p.get('PrivatePort')}/{p.get('Type', 'tcp')}"
        if p.get("PublicPort"):
            out.append(f"{p.get('IP') or '0.0.0.0'}:{p['PublicPort']}->{private}")
        else:
            out.append(private)
    return ", ".join(sorted(set(out)))


def _inspect_ports(ports: dict | None) -> list[str]:
    out = []
    for private, bindings in (ports or {}).items():
        if not bindings:
            out.append(private)
        for b in bindings or []:
            out.append(f"{b.get('HostIp') or '0.0.0.0'}:{b.get('HostPort')}->{private}")
    return sorted(set(out))


class NoArgs(ToolArgs):
    pass


class ListContainersArgs(ToolArgs):
    all: bool = Field(False, description="Also include stopped/exited containers")


class ContainerArgs(ToolArgs):
    container: str = Field(description="Container name or ID")


class ContainerLogsArgs(ToolArgs):
    container: str = Field(description="Container name or ID")
    tail: int = Field(100, ge=1, le=2000, description="Number of most recent log lines")


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: "List all containers (including stopped)" if a.all else "List running containers",
)
def docker_list_containers(args: ListContainersArgs) -> dict:
    """List Docker containers with their image, state, status and ports."""
    with _docker_errors():
        containers = get_client().containers.list(all=args.all, sparse=True)
    rows = [
        {
            "name": (c.attrs.get("Names") or ["?"])[0].lstrip("/"),
            "image": c.attrs.get("Image", ""),
            "state": c.attrs.get("State", ""),
            "status": c.attrs.get("Status", ""),
            "ports": _list_ports(c.attrs.get("Ports")),
            "id": c.id[:12],
        }
        for c in containers
    ]
    return {"kind": "table", "columns": ["name", "image", "state", "status", "ports", "id"], "rows": rows}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Read the last {a.tail} log lines of '{a.container}'",
)
def docker_container_logs(args: ContainerLogsArgs) -> dict:
    """Get the most recent log lines of a container."""
    with _docker_errors():
        container = _find_container(args.container)
        raw = container.logs(tail=args.tail, timestamps=False)
    text = raw.decode("utf-8", errors="replace")
    return {"kind": "logs", "title": container.name, "text": text or "(no log output)"}


@registry.tool(risk=Risk.READ, summary=lambda a: f"Inspect container '{a.container}'")
def docker_inspect_container(args: ContainerArgs) -> dict:
    """Show details of one container: state, exit code, health, restart count, ports, mounts, networks and environment variable names (not values)."""
    with _docker_errors():
        container = _find_container(args.container)
        attrs = container.attrs
    state = attrs.get("State") or {}
    config = attrs.get("Config") or {}
    network = attrs.get("NetworkSettings") or {}
    host = attrs.get("HostConfig") or {}
    data = {
        "name": container.name,
        "id": container.short_id,
        "image": config.get("Image"),
        "command": " ".join(config.get("Cmd") or []),
        "created": attrs.get("Created"),
        "state": {
            k: state.get(k)
            for k in ("Status", "Running", "ExitCode", "Error", "OOMKilled", "StartedAt", "FinishedAt")
        },
        "health": (state.get("Health") or {}).get("Status"),
        "restart_count": attrs.get("RestartCount"),
        "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
        "ports": _inspect_ports(network.get("Ports")),
        "mounts": [
            f"{m.get('Source') or m.get('Name')} -> {m.get('Destination')}" for m in attrs.get("Mounts") or []
        ],
        "networks": list((network.get("Networks") or {}).keys()),
        # Values can hold secrets, so only names are exposed.
        "env_names": [e.split("=", 1)[0] for e in config.get("Env") or []],
    }
    return {"kind": "json", "title": container.name, "data": data}


@registry.tool(risk=Risk.READ, summary=lambda a: "List local images")
def docker_list_images(args: NoArgs) -> dict:
    """List local Docker images with their tags, IDs and sizes."""
    with _docker_errors():
        images = get_client().images.list()
    rows = [
        {
            "tags": ", ".join(i.tags) or "<none>",
            "id": i.short_id.removeprefix("sha256:"),
            "size": f"{i.attrs.get('Size', 0) / 1e6:.0f} MB",
            "created": str(i.attrs.get("Created", ""))[:19],
        }
        for i in images
    ]
    return {"kind": "table", "columns": ["tags", "id", "size", "created"], "rows": rows}


# ---------------------------------------------------------------------------
# Phase 2: actions. Each has a `preview` that runs before the approval card:
# it shows what will be affected and rejects actions that can't work.
# ---------------------------------------------------------------------------

# Loose check for an image reference such as "nginx", "nginx:1.27" or "ghcr.io/org/app:v1".
_IMAGE_REF = re.compile(r"^[a-z0-9][a-z0-9._/:@-]{0,254}$")


def _state(container: Container) -> dict:
    return container.attrs.get("State") or {}


def _container_facts(container: Container) -> dict[str, str]:
    state = _state(container)
    status = state.get("Status", "unknown")
    facts = {"Container": container.name, "Status": status}
    if status == "running":
        facts["Running since"] = str(state.get("StartedAt", ""))[:19].replace("T", " ")
    elif status == "exited":
        facts["Exit code"] = str(state.get("ExitCode"))
        facts["Exited at"] = str(state.get("FinishedAt", ""))[:19].replace("T", " ")
    facts["Image"] = (container.attrs.get("Config") or {}).get("Image", "")
    ports = _inspect_ports((container.attrs.get("NetworkSettings") or {}).get("Ports"))
    if ports:
        facts["Ports"] = ", ".join(ports)
    return facts


def _preview(facts: dict[str, str], warnings: list[str] | None = None) -> dict:
    return {"facts": facts, "warnings": warnings or []}


def _done(message: str, container: Container | None = None) -> dict:
    facts: dict[str, str] = {}
    if container is not None:
        container.reload()
        facts = _container_facts(container)
    return {"kind": "facts", "title": message, "facts": facts}


def _find_image(ref: str) -> Image:
    try:
        return get_client().images.get(ref)
    except NotFound:
        raise ToolError(f"No local image '{ref}'. Use the image list to see what exists.")


class StopArgs(ToolArgs):
    container: str = Field(description="Container name or ID")
    timeout: int = Field(10, ge=0, le=300, description="Seconds to wait for a clean shutdown before killing")


class RemoveContainerArgs(ToolArgs):
    container: str = Field(description="Container name or ID")
    force: bool = Field(False, description="Kill and remove the container even if it is running. Only if the user asked to force it")
    remove_volumes: bool = Field(False, description="Also delete anonymous volumes attached to the container. Only if the user asked")


class ImageArgs(ToolArgs):
    image: str = Field(description='Image reference, e.g. "nginx", "nginx:1.27" or "ghcr.io/org/app:v1"')


class RemoveImageArgs(ToolArgs):
    image: str = Field(description="Image tag or ID")
    force: bool = Field(False, description="Remove even if stopped containers use it. Only if the user asked to force it")


# ---- start / stop / restart ----------------------------------------------------


def _preview_start(args: ContainerArgs) -> dict:
    with _docker_errors():
        container = _find_container(args.container)
    if _state(container).get("Status") == "running":
        raise ToolError(f"'{container.name}' is already running.")
    return _preview(_container_facts(container))


@registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Start container '{a.container}'", preview=_preview_start)
def docker_start_container(args: ContainerArgs) -> dict:
    """Start a stopped or exited container."""
    with _docker_errors():
        container = _find_container(args.container)
        container.start()
        return _done(f"Started {container.name}", container)


def _preview_stop(args: StopArgs) -> dict:
    with _docker_errors():
        container = _find_container(args.container)
    if _state(container).get("Status") != "running":
        raise ToolError(f"'{container.name}' isn't running (status: {_state(container).get('Status')}).")
    return _preview(_container_facts(container), [f"Anything served by '{container.name}' will be unavailable until it is started again."])


@registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Stop container '{a.container}'", preview=_preview_stop)
def docker_stop_container(args: StopArgs) -> dict:
    """Stop a running container (graceful shutdown, then kill after the timeout)."""
    with _docker_errors():
        container = _find_container(args.container)
        container.stop(timeout=args.timeout)
        return _done(f"Stopped {container.name}", container)


def _preview_restart(args: StopArgs) -> dict:
    with _docker_errors():
        container = _find_container(args.container)
    warnings = []
    if _state(container).get("Status") == "running":
        warnings.append(f"'{container.name}' will be briefly unavailable while it restarts.")
    return _preview(_container_facts(container), warnings)


@registry.tool(risk=Risk.CHANGE, summary=lambda a: f"Restart container '{a.container}'", preview=_preview_restart)
def docker_restart_container(args: StopArgs) -> dict:
    """Restart a container (stops it if running, then starts it again)."""
    with _docker_errors():
        container = _find_container(args.container)
        container.restart(timeout=args.timeout)
        return _done(f"Restarted {container.name}", container)


# ---- pull ----------------------------------------------------------------------


def _split_ref(ref: str) -> tuple[str, str]:
    ref = ref.strip()
    if not _IMAGE_REF.match(ref):
        raise ToolError(f"'{ref}' doesn't look like a valid image reference.")
    repo, tag = parse_repository_tag(ref)
    return repo, tag or "latest"


def _compose_build(repo: str) -> tuple[str, str] | None:
    """(project, service) if `repo` is an image Docker Compose built locally for some container.

    Compose names images it builds "<project>-<service>", so such an image has no registry.
    """
    for c in get_client().containers.list(all=True, sparse=True):
        labels = c.attrs.get("Labels") or {}
        project = labels.get("com.docker.compose.project")
        service = labels.get("com.docker.compose.service")
        if project and service and repo == f"{project}-{service}":
            return project, service
    return None


def _compose_project_images(name: str) -> list[str]:
    """Images of the containers in Compose project `name` (empty if it isn't a project)."""
    images = {
        c.attrs.get("Image", "")
        for c in get_client().containers.list(all=True, sparse=True)
        if (c.attrs.get("Labels") or {}).get("com.docker.compose.project") == name
    }
    return sorted(i for i in images if i)


def _not_pullable_hint(repo: str) -> str | None:
    """Explain why `repo` can't be pulled, for the mix-ups that happen in practice."""
    project_images = _compose_project_images(repo)
    if project_images:
        built = [i for i in project_images if _compose_build(i.split(":")[0])]
        pullable = [i for i in project_images if i not in built]
        hint = f"'{repo}' is a Docker Compose project, not an image."
        if built:
            hint += (
                f" Built locally (can't be pulled; rebuild with `docker compose build` in the "
                f"project folder): {', '.join(built)}."
            )
        if pullable:
            hint += f" From a registry (can be pulled by name): {', '.join(pullable)}."
        return hint
    built = _compose_build(repo)
    if built:
        project, service = built
        return (
            f"'{repo}' was built locally by Docker Compose (project '{project}'), so there's no "
            f"registry to pull it from. To update it, rebuild it from the project folder with "
            f"`docker compose build {service}`."
        )
    try:
        container = get_client().containers.get(repo)
    except NotFound:
        return None
    image = (container.attrs.get("Config") or {}).get("Image", "")
    hint = f"'{repo}' is a container, not an image. Its image is '{image}'."
    image_built = _compose_build(image.split(":")[0])
    if image_built:
        hint += (
            f" That image was built locally by Docker Compose, so it can't be pulled; rebuild it "
            f"with `docker compose build {image_built[1]}` instead."
        )
    return hint


def _preview_pull(args: ImageArgs) -> dict:
    repo, tag = _split_ref(args.image)
    with _docker_errors():
        hint = _not_pullable_hint(repo)
    if hint:
        raise ToolError(hint)
    facts = {"Image": f"{repo}:{tag}"}
    try:
        with _docker_errors():
            existing = get_client().images.get(f"{repo}:{tag}")
        facts["Local copy"] = f"yes ({existing.short_id.removeprefix('sha256:')}), it will be updated if a newer one exists"
    except (NotFound, ToolError):
        facts["Local copy"] = "no, it will be downloaded"
    registry_name = repo.split("/")[0] if "." in repo.split("/")[0] or ":" in repo.split("/")[0] else "Docker Hub"
    return _preview(facts, [f"Downloads from {registry_name}."])


@registry.tool(
    risk=Risk.CHANGE,
    summary=lambda a: f"Pull image '{a.image}'",
    preview=_preview_pull,
    grounded=("image",),
)
def docker_pull_image(args: ImageArgs) -> dict:
    """Download (pull) an image from its registry."""
    repo, tag = _split_ref(args.image)
    try:
        with _docker_errors():
            image = get_client().images.pull(repo, tag=tag)
    except ToolError as e:
        if "pull access denied" in str(e) or "not found" in str(e).lower():
            raise ToolError(
                f"The registry has no image '{repo}:{tag}' that this machine can access. Either it "
                "doesn't exist (check the name and tag), or it's private and needs `docker login` first."
            ) from e
        raise
    return {
        "kind": "facts",
        "title": f"Pulled {repo}:{tag}",
        "facts": {
            "Tags": ", ".join(image.tags) or "<none>",
            "ID": image.short_id.removeprefix("sha256:"),
            "Size": f"{image.attrs.get('Size', 0) / 1e6:.0f} MB",
        },
    }


# ---- stats (read) ------------------------------------------------------------


def _cpu_percent(stats: dict) -> float:
    cpu, pre = stats.get("cpu_stats") or {}, stats.get("precpu_stats") or {}
    cpu_delta = (cpu.get("cpu_usage") or {}).get("total_usage", 0) - (pre.get("cpu_usage") or {}).get("total_usage", 0)
    system_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    cpus = cpu.get("online_cpus") or len((cpu.get("cpu_usage") or {}).get("percpu_usage") or []) or 1
    if cpu_delta <= 0 or system_delta <= 0:
        return 0.0
    return cpu_delta / system_delta * cpus * 100


def _mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


@registry.tool(risk=Risk.READ, summary=lambda a: f"Read live CPU and memory usage of '{a.container}'")
def docker_container_stats(args: ContainerArgs) -> dict:
    """Show live resource usage of a running container: CPU, memory, network and disk I/O, process count."""
    with _docker_errors():
        container = _find_container(args.container)
        if _state(container).get("Status") != "running":
            raise ToolError(f"'{container.name}' isn't running, so it has no live stats.")
        stats = container.stats(stream=False)
    mem = stats.get("memory_stats") or {}
    cache = (mem.get("stats") or {}).get("inactive_file", (mem.get("stats") or {}).get("cache", 0))
    used, limit = mem.get("usage", 0) - cache, mem.get("limit", 0)
    nets = (stats.get("networks") or {}).values()
    blk = (stats.get("blkio_stats") or {}).get("io_service_bytes_recursive") or []
    return {
        "kind": "facts",
        "title": f"{container.name} · live usage",
        "facts": {
            "CPU": f"{_cpu_percent(stats):.1f} %",
            "Memory": f"{_mb(used)} / {_mb(limit)}" + (f" ({used / limit * 100:.1f} %)" if limit else ""),
            "Network in / out": f"{_mb(sum(n.get('rx_bytes', 0) for n in nets))} / {_mb(sum(n.get('tx_bytes', 0) for n in nets))}",
            "Disk read / write": f"{_mb(sum(b.get('value', 0) for b in blk if b.get('op', '').lower() == 'read'))} / "
            f"{_mb(sum(b.get('value', 0) for b in blk if b.get('op', '').lower() == 'write'))}",
            "Processes": str((stats.get("pids_stats") or {}).get("current", "?")),
        },
    }


# ---- remove (destructive) ----------------------------------------------------------


def _preview_remove_container(args: RemoveContainerArgs) -> dict:
    with _docker_errors():
        container = _find_container(args.container)
    running = _state(container).get("Status") == "running"
    if running and not args.force:
        raise ToolError(f"'{container.name}' is running. Stop it first, or ask me to force-remove it.")
    warnings = ["The container and its writable layer are deleted permanently. Named volumes are kept."]
    if running:
        warnings.insert(0, "It is running and will be killed without a clean shutdown.")
    if args.remove_volumes:
        warnings.append("Anonymous volumes attached to it will be deleted too.")
    return _preview(_container_facts(container), warnings)


@registry.tool(
    risk=Risk.DESTRUCTIVE,
    summary=lambda a: f"{'Force-remove' if a.force else 'Remove'} container '{a.container}'",
    confirm_field="container",
    preview=_preview_remove_container,
)
def docker_remove_container(args: RemoveContainerArgs) -> dict:
    """Permanently delete a container. It must be stopped unless force is true."""
    with _docker_errors():
        container = _find_container(args.container)
        name = container.name
        container.remove(force=args.force, v=args.remove_volumes)
    return {"kind": "facts", "title": f"Removed {name}", "facts": {}}


def _image_users(image: Image) -> list[str]:
    containers = get_client().containers.list(all=True, sparse=True)
    return [
        f"{(c.attrs.get('Names') or ['?'])[0].lstrip('/')} ({c.attrs.get('State', '?')})"
        for c in containers
        if c.attrs.get("ImageID") == image.id
    ]


def _preview_remove_image(args: RemoveImageArgs) -> dict:
    with _docker_errors():
        image = _find_image(args.image)
        users = _image_users(image)
    if any(u.endswith("(running)") for u in users):
        raise ToolError(f"It's used by running containers: {', '.join(users)}. Stop and remove them first.")
    if users and not args.force:
        raise ToolError(f"It's used by containers: {', '.join(users)}. Remove them first, or ask me to force it.")
    facts = {
        "Image": ", ".join(image.tags) or image.short_id,
        "ID": image.short_id.removeprefix("sha256:"),
        "Size": f"{image.attrs.get('Size', 0) / 1e6:.0f} MB",
        "Used by": ", ".join(users) or "no containers",
    }
    return _preview(facts, ["The image is deleted locally; you'd have to pull or build it again."])


@registry.tool(
    risk=Risk.DESTRUCTIVE,
    summary=lambda a: f"{'Force-remove' if a.force else 'Remove'} image '{a.image}'",
    confirm_field="image",
    preview=_preview_remove_image,
)
def docker_remove_image(args: RemoveImageArgs) -> dict:
    """Permanently delete a local image. It must not be used by any container unless force is true."""
    with _docker_errors():
        get_client().images.remove(args.image, force=args.force)
    return {"kind": "facts", "title": f"Removed image {args.image}", "facts": {}}
