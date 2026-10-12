"""Docker tools against an in-memory fake client; no real Docker is touched."""

from __future__ import annotations

import pytest
from docker.errors import APIError, ImageNotFound, NotFound

from app.tools import docker_tools as dt
from app.tools.registry import ToolError, registry


class FakeContainer:
    def __init__(self, name: str, status: str, image_id: str = "sha256:img1", exit_code: int = 0):
        self.name, self.id, self.short_id = name, f"{name}-id-000000000000", f"{name}-id"
        self.image_id = image_id
        self.calls: list[tuple] = []
        self.attrs = {
            "State": {"Status": status, "ExitCode": exit_code, "StartedAt": "2026-09-27T10:00:00Z",
                      "FinishedAt": "2026-09-27T11:00:00Z"},
            "Config": {"Image": "web:latest"},
            "NetworkSettings": {"Ports": {"80/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}]}},
        }

    def _set(self, status):
        self.attrs["State"]["Status"] = status

    def start(self):
        self.calls.append(("start",)); self._set("running")

    def stop(self, timeout):
        self.calls.append(("stop", timeout)); self._set("exited")

    def restart(self, timeout):
        self.calls.append(("restart", timeout)); self._set("running")

    def remove(self, force, v):
        self.calls.append(("remove", force, v))

    def reload(self):
        pass

    def stats(self, stream):
        return {
            "cpu_stats": {"cpu_usage": {"total_usage": 300}, "system_cpu_usage": 2000, "online_cpus": 2},
            "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000},
            "memory_stats": {"usage": 60_000_000, "limit": 200_000_000, "stats": {"inactive_file": 10_000_000}},
            "networks": {"eth0": {"rx_bytes": 1_000_000, "tx_bytes": 2_000_000}},
            "blkio_stats": {"io_service_bytes_recursive": [{"op": "read", "value": 3_000_000}, {"op": "write", "value": 4_000_000}]},
            "pids_stats": {"current": 7},
        }


class Sparse:
    def __init__(self, c: FakeContainer):
        self.attrs = {"Names": [f"/{c.name}"], "State": c.attrs["State"]["Status"], "ImageID": c.image_id,
                      "Image": c.attrs["Config"]["Image"], "Labels": c.attrs["Config"].get("Labels", {})}


class FakeImage:
    def __init__(self, id_: str, tags: list[str]):
        self.id, self.short_id, self.tags = id_, id_[:19], tags
        self.attrs = {"Size": 50_000_000}


class FakeContainers:
    def __init__(self, items):
        self.items = {c.name: c for c in items}

    def get(self, ref):
        if ref in self.items:
            return self.items[ref]
        raise NotFound(f"no such container {ref}")

    def list(self, all=False, sparse=False):
        return [Sparse(c) for c in self.items.values()]


class FakeImages:
    def __init__(self, items):
        self.items = items
        self.calls: list[tuple] = []

    def get(self, ref):
        for img in self.items:
            if ref in img.tags or ref == img.id:
                return img
        raise ImageNotFound(f"no such image {ref}")

    def pull(self, repo, tag=None):
        self.calls.append(("pull", repo, tag))
        if repo == "private-thing":
            raise APIError("denied", explanation="pull access denied for private-thing, repository does not exist or may require 'docker login'")
        return FakeImage("sha256:pulled", [f"{repo}:{tag}"])

    def remove(self, ref, force=False):
        self.calls.append(("remove", ref, force))


class FakeClient:
    def __init__(self, containers, images):
        self.containers, self.images = FakeContainers(containers), FakeImages(images)

    def ping(self):
        return True


@pytest.fixture
def docker(monkeypatch):
    client = FakeClient(
        [FakeContainer("web", "running"), FakeContainer("worker", "exited", exit_code=137),
         FakeContainer("old", "exited", image_id="sha256:img2")],
        [FakeImage("sha256:img1", ["web:latest"]), FakeImage("sha256:img2", ["old:1"]),
         FakeImage("sha256:img3", ["unused:1"])],
    )
    monkeypatch.setattr(dt, "_client", client)
    return client


def run(name, **kwargs):
    tool = registry.get(name)
    args = tool.args_model.model_validate(kwargs)
    return tool, args


def preview(name, **kwargs):
    tool, args = run(name, **kwargs)
    return tool.preview(args)


def call(name, **kwargs):
    tool, args = run(name, **kwargs)
    return tool.func(args)


def test_start_refuses_running_container(docker):
    with pytest.raises(ToolError, match="already running"):
        preview("docker_start_container", container="web")


def test_start_exited_container(docker):
    facts = preview("docker_start_container", container="worker")["facts"]
    assert facts["Status"] == "exited" and facts["Exit code"] == "137"
    result = call("docker_start_container", container="worker")
    assert docker.containers.items["worker"].calls == [("start",)]
    assert result["facts"]["Status"] == "running"


def test_stop_refuses_stopped_and_stops_running(docker):
    with pytest.raises(ToolError, match="isn't running"):
        preview("docker_stop_container", container="worker")
    assert preview("docker_stop_container", container="web")["warnings"]
    call("docker_stop_container", container="web", timeout=5)
    assert docker.containers.items["web"].calls == [("stop", 5)]


def test_restart(docker):
    call("docker_restart_container", container="web")
    assert docker.containers.items["web"].calls == [("restart", 10)]


def test_unknown_container_suggests_names(docker):
    with pytest.raises(ToolError, match="Did you mean: web"):
        preview("docker_restart_container", container="we")


def test_remove_running_container_needs_force(docker):
    with pytest.raises(ToolError, match="is running"):
        preview("docker_remove_container", container="web")
    warnings = preview("docker_remove_container", container="web", force=True)["warnings"]
    assert "killed" in warnings[0]
    call("docker_remove_container", container="web", force=True)
    assert docker.containers.items["web"].calls == [("remove", True, False)]


def test_remove_container_is_destructive_with_typed_confirmation():
    tool = registry.get("docker_remove_container")
    assert tool.risk.value == "destructive" and tool.confirm_field == "container"


def test_remove_image_checks_users(docker):
    with pytest.raises(ToolError, match="running containers"):
        preview("docker_remove_image", image="web:latest")
    with pytest.raises(ToolError, match="or ask me to force"):
        preview("docker_remove_image", image="old:1")
    assert preview("docker_remove_image", image="old:1", force=True)["facts"]["Used by"] == "old (exited)"
    assert preview("docker_remove_image", image="unused:1")["facts"]["Used by"] == "no containers"
    call("docker_remove_image", image="unused:1")
    assert docker.images.calls == [("remove", "unused:1", False)]


def test_remove_missing_image(docker):
    with pytest.raises(ToolError, match="No local image"):
        preview("docker_remove_image", image="nope:1")


def test_pull(docker):
    with pytest.raises(ToolError, match="valid image reference"):
        preview("docker_pull_image", image="Bad Image!")
    assert "downloaded" in preview("docker_pull_image", image="nginx:1.27")["facts"]["Local copy"]
    assert "updated" in preview("docker_pull_image", image="unused:1")["facts"]["Local copy"]
    call("docker_pull_image", image="nginx:1.27")
    assert docker.images.calls == [("pull", "nginx", "1.27")]


def test_stats(docker):
    facts = call("docker_container_stats", container="web")["facts"]
    assert facts["CPU"] == "40.0 %"  # 200/1000 * 2 cpus
    assert facts["Memory"] == "50.0 MB / 200.0 MB (25.0 %)"
    assert facts["Processes"] == "7"
    with pytest.raises(ToolError, match="no live stats"):
        call("docker_container_stats", container="worker")


def test_extra_args_rejected():
    tool = registry.get("docker_stop_container")
    with pytest.raises(Exception):
        tool.args_model.model_validate({"container": "web", "rm": "-rf"})


@pytest.fixture
def compose(docker):
    auth = FakeContainer("authentication-service", "exited")
    auth.attrs["Config"] = {
        "Image": "shop-authentication-service",
        "Labels": {"com.docker.compose.project": "shop", "com.docker.compose.service": "authentication-service"},
    }
    docker.containers.items[auth.name] = auth
    return docker


def test_pull_container_name_explains_the_mixup(compose):
    with pytest.raises(ToolError) as e:
        preview("docker_pull_image", image="authentication-service")
    msg = str(e.value)
    assert "is a container, not an image" in msg and "shop-authentication-service" in msg
    assert "docker compose build authentication-service" in msg
    assert compose.images.calls == []


def test_pull_compose_built_image_is_refused(compose):
    with pytest.raises(ToolError, match="built locally by Docker Compose"):
        preview("docker_pull_image", image="shop-authentication-service")


def test_pull_denied_is_explained_without_guessing(docker):
    with pytest.raises(ToolError, match="doesn't exist .* or it's private"):
        call("docker_pull_image", image="private-thing")


def test_pull_names_the_registry(docker):
    assert preview("docker_pull_image", image="nginx")["warnings"] == ["Downloads from Docker Hub."]
    assert preview("docker_pull_image", image="ghcr.io/org/app:v1")["warnings"] == ["Downloads from ghcr.io."]


def test_pull_compose_project_name_is_refused(compose):
    db = FakeContainer("db", "running")
    db.attrs["Config"] = {"Image": "postgres:15", "Labels": {"com.docker.compose.project": "shop",
                                                             "com.docker.compose.service": "db"}}
    compose.containers.items["db"] = db
    with pytest.raises(ToolError) as e:
        preview("docker_pull_image", image="shop")
    msg = str(e.value)
    assert "is a Docker Compose project" in msg
    assert "Built locally (can't be pulled" in msg and "shop-authentication-service" in msg
    assert "can be pulled by name): postgres:15" in msg


def test_pull_is_grounded():
    assert registry.get("docker_pull_image").grounded == ("image",)
