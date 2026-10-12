"""Step 5: versions, safe switch (redeploy), rollback and cleanup, against a fake Docker."""

import pytest
from docker.errors import NotFound

from app.config import RunSettings
from app.store import ImageVersion, Store
from app.tools import code_tools, deploy_tools, docker_tools, run_tools  # noqa: F401  (registers the tools)
from app.tools.registry import RunContext, ToolError, registry

TITLE = "org.opencontainers.image.title"


class Image:
    def __init__(self, id, tags, size=100_000_000, project="shop-api"):
        self.id = id
        self.short_id = id[:19]
        self.tags = tags
        self.labels = {TITLE: project}
        self.attrs = {"Config": {"ExposedPorts": {"8080/tcp": {}}}, "Size": size, "Created": "2026-10-08T10:00:00"}


class Container:
    def __init__(self, docker, name, image_id, labels, port, status="running"):
        self.docker = docker
        self.id = f"c-{name}-{len(docker.created)}"
        self.name = name
        self.labels = labels
        self.status = status
        self.port = port
        self.attrs = {"Image": image_id, "State": {"ExitCode": 0},
                      "HostConfig": {"PortBindings": {"8080/tcp": [{"HostPort": str(port)}]}}}
        self.events = []

    def reload(self):
        pass

    def logs(self, stdout=True, stderr=True):
        return b"Uvicorn running on http://0.0.0.0:8080\n"

    def remove(self, force=False):
        self.events.append("removed")
        self.docker.containers_by_id.pop(self.id, None)

    def stop(self, timeout=10):
        self.events.append("stopped")
        self.status = "exited"

    def start(self):
        self.events.append("started")
        self.status = "running"

    def rename(self, name):
        self.events.append(f"renamed:{name}")
        self.name = name


class Docker:
    def __init__(self, images):
        self.images_by_tag = {t: img for img in images for t in img.tags}
        self.all_images = list(images)
        self.containers_by_id = {}
        self.created = []
        self.removed_images = []
        docker = self

        class Images:
            def get(self, ref):
                if ref not in docker.images_by_tag:
                    raise NotFound(ref)
                return docker.images_by_tag[ref]

            def list(self, all=False, filters=None):
                key, value = filters["label"].split("=", 1)
                return [i for i in docker.all_images if i.labels.get(key) == value]

            def remove(self, image_id, force=False):
                docker.removed_images.append(image_id)
                docker.all_images = [i for i in docker.all_images if i.id != image_id]
                docker.images_by_tag = {t: i for t, i in docker.images_by_tag.items() if i.id != image_id}

        class Containers:
            def list(self, all=False, filters=None):
                found = list(docker.containers_by_id.values())
                if filters and "name" in filters:
                    found = [c for c in found if filters["name"] == f"^/{c.name}$"]
                return found

            def get(self, container_id):
                if container_id not in docker.containers_by_id:
                    raise NotFound(container_id)
                return docker.containers_by_id[container_id]

            def run(self, image, name, ports, labels, **kwargs):
                port = list(ports.values())[0][1]
                c = Container(docker, name, docker.images_by_tag[image].id, labels, port)
                docker.created.append(c)
                docker.containers_by_id[c.id] = c
                return c

        self.images = Images()
        self.containers = Containers()

    def add_container(self, name, image, version, port=8080, status="running", ours=True):
        labels = {run_tools.LABEL: "shop-api", run_tools.VERSION_LABEL: version} if ours else {}
        c = Container(self, name, image.id, labels, port, status)
        self.containers_by_id[c.id] = c
        return c


V1 = Image("sha256:111", ["shop-api:v1-aaa1111"])
V2 = Image("sha256:222", ["shop-api:v2-bbb2222"])
V3 = Image("sha256:333", ["shop-api:v3-ccc3333", "shop-api:latest"])
OLD = Image("sha256:000", ["shop-api:e54f34f-dirty"])  # from before versions were recorded


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "shop-api"
    root.mkdir()
    store = Store("sqlite://")
    store.add_project("shop-api", str(root.resolve()))
    code_tools.configure(store)
    for n, img, healthy in ((1, V1, True), (2, V2, True), (3, V3, None)):
        store.add_version(ImageVersion(project="shop-api", number=n, tag=img.tags[0], image_id=img.id,
                                       commit=img.tags[0].split("-")[-1], size_bytes=img.attrs["Size"], healthy=healthy))
    run_tools.configure(RunSettings.model_construct(**{**RunSettings().model_dump(), "health_timeout_s": 1,
                                                       "health_interval_s": 0.05, "health_failure_threshold": 3}))
    monkeypatch.setattr(run_tools, "port_free", lambda port, address: True)
    healthy_ports = {}

    async def probe(url):
        return healthy_ports.get(int(url.split(":")[2].split("/")[0]))

    monkeypatch.setattr(run_tools, "_probe", probe)

    def make(images=(V1, V2, V3)):
        docker = Docker(images)
        monkeypatch.setattr(docker_tools, "get_client", lambda: docker)
        return docker

    return store, make, healthy_ports


async def run(version=None):
    tool = registry.get("project_run_container")
    args = tool.args_model.model_validate({"project": "shop-api", **({"version": version} if version else {})})
    prev = tool.preview(args)

    async def progress(line):
        pass

    return prev, await tool.call(args, prev, RunContext(progress=progress))


# ---- safe switch (redeploy) ---------------------------------------------------------------------


async def test_a_new_version_is_checked_next_to_the_running_one_then_switched(env):
    store, make, healthy = env
    docker = make()
    old = docker.add_container("shop-api", V2, "shop-api:v2-bbb2222")
    healthy.update({8081: 200, 8080: 200})
    prev, result = await run()
    assert "v2-bbb2222 → v3-ccc3333" in prev["facts"]["Switch"] and "port 8081" in prev["facts"]["Switch"]
    assert result["ok"] and result["title"] == "v3-ccc3333 is live at http://localhost:8080"
    trial, final = docker.created
    assert (trial.name, trial.port, trial.events) == ("shop-api-next", 8081, ["removed"])
    assert (final.name, final.port, final.labels[run_tools.VERSION_LABEL]) == ("shop-api", 8080, "shop-api:v3-ccc3333")
    assert old.events == ["stopped", "renamed:shop-api-old", "removed"]
    assert store.versions("shop-api")[0].healthy is True


async def test_if_the_new_version_fails_its_trial_the_running_one_is_untouched(env):
    store, make, healthy = env
    docker = make()
    old = docker.add_container("shop-api", V2, "shop-api:v2-bbb2222")
    healthy.update({8080: 200})  # the trial port never answers
    with pytest.raises(ToolError, match="nothing was switched: the version that was running is still running"):
        await run()
    assert old.events == [] and old.status == "running"
    assert docker.created[0].events == ["removed"]
    assert store.versions("shop-api")[0].healthy is False


async def test_if_the_switch_fails_on_the_real_port_the_old_version_is_put_back(env):
    store, make, healthy = env
    docker = make()
    old = docker.add_container("shop-api", V2, "shop-api:v2-bbb2222")
    healthy.update({8081: 200})  # answers on the trial port, not on the real one
    with pytest.raises(ToolError, match="put the previous version back"):
        await run()
    assert old.events == ["stopped", "renamed:shop-api-old", "renamed:shop-api", "started"]
    assert old.name == "shop-api" and old.status == "running"
    assert docker.created[1].events == ["removed"]


# ---- rollback -----------------------------------------------------------------------------------


async def test_previous_means_the_last_healthy_version_before_the_running_one(env):
    store, make, healthy = env
    docker = make()
    docker.add_container("shop-api", V3, "shop-api:v3-ccc3333")
    store.record_run("shop-api", 3, True)
    healthy.update({8081: 200, 8080: 200})
    prev, result = await run("previous")
    assert prev["run"]["image"] == "shop-api:v2-bbb2222"
    assert registry.get("project_run_container").summarize(
        registry.get("project_run_container").args_model(project="shop-api", version="previous")
    ) == "Roll 'shop-api' back to its previous healthy version"
    assert result["title"] == "v2-bbb2222 is live at http://localhost:8080"


async def test_a_specific_version_can_be_run_but_not_a_made_up_or_removed_one(env):
    store, make, healthy = env
    make(images=(V2, V3))  # v1's image was removed
    healthy.update({8080: 200})
    prev, _ = await run("v2")
    assert prev["run"]["image"] == "shop-api:v2-bbb2222"
    with pytest.raises(ToolError, match="There's no version 'v9' of shop-api. Versions: v3, v2, v1"):
        await run("v9")
    with pytest.raises(ToolError, match="v1 was removed"):
        await run("v1")


async def test_nothing_to_roll_back_to(env):
    store, make, _ = env
    docker = make()
    docker.add_container("shop-api", V1, "shop-api:v1-aaa1111")
    with pytest.raises(ToolError, match="no version older than v1-aaa1111 that passed a health check"):
        await run("previous")


async def test_a_version_that_never_passed_is_warned_about(env):
    store, make, _ = env
    make()
    tool = registry.get("project_run_container")
    prev = tool.preview(tool.args_model(project="shop-api"))
    assert "v3-ccc3333 has never passed a health check." in prev["warnings"]


# ---- history and cleanup ------------------------------------------------------------------------


def test_versions_lists_health_status_and_old_untracked_images(env):
    store, make, _ = env
    docker = make(images=(V2, V3, OLD))  # v1 was removed
    docker.add_container("shop-api", V2, "shop-api:v2-bbb2222")
    tool = registry.get("project_versions")
    rows = {r["version"]: r for r in tool.func(tool.args_model(project="shop-api"))["rows"]}
    assert rows["v3"]["health check"] == "not run yet" and rows["v3"]["status"] == ""
    assert rows["v2"]["status"] == "● running" and rows["v2"]["health check"] == "passed"
    assert rows["v1"]["status"] == "removed"
    assert rows["e54f34f-dirty"]["health check"].startswith("unknown")


def test_cleanup_keeps_newest_running_and_healthy_and_removes_the_rest(env):
    store, make, _ = env
    for n in range(4, 7):  # v4-v6: more healthy versions than `keep`
        store.add_version(ImageVersion(project="shop-api", number=n, tag=f"shop-api:v{n}", image_id=f"sha256:{n}" * 3,
                                       healthy=True, size_bytes=1))
    extra = [Image(f"sha256:{n}" * 3, [f"shop-api:v{n}"]) for n in range(4, 7)]
    docker = make(images=(V1, V2, V3, OLD, *extra))
    docker.add_container("shop-api", V1, "shop-api:v1-aaa1111")  # v1 is running
    tool = registry.get("project_prune_images")
    args = tool.args_model(project="shop-api", keep=2)
    prev = tool.preview(args)
    assert prev["facts"]["Removes"] == "v2, v4, e54f34f-dirty"
    assert "v3 (newest build)" in prev["facts"]["Keeps"] and "v1 (used by a container)" in prev["facts"]["Keeps"]
    assert tool.confirm_text(args) == "shop-api"  # destructive: the user types the name
    result = tool.run(args, prev)
    assert set(docker.removed_images) == {V2.id, OLD.id, "sha256:4" * 3}
    assert result["facts"]["Removed"] == "v2, v4, e54f34f-dirty"


def test_cleanup_with_nothing_to_remove_is_refused(env):
    store, make, _ = env
    make(images=(V3,))
    tool = registry.get("project_prune_images")
    with pytest.raises(ToolError, match="Nothing to clean up"):
        tool.preview(tool.args_model(project="shop-api"))



async def test_a_new_version_whose_health_endpoint_fails_is_never_switched_in(env):
    store, make, healthy = env
    docker = make()
    old = docker.add_container("shop-api", V2, "shop-api:v2-bbb2222")
    healthy.update({8081: 500, 8080: 200})  # the user's test: /health raises -> HTTP 500
    with pytest.raises(ToolError, match="nothing was switched.*answered HTTP 500 3 times in a row"):
        await run()
    assert old.events == [] and old.status == "running"
    assert store.versions("shop-api")[0].healthy is False
