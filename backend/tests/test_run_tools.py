"""project_run_container with a fake Docker client and a fake HTTP probe (nothing real is started)."""

import json

import pytest
from docker.errors import NotFound

from app.config import RunSettings
from app.store import Store
from app.tools import code_tools, docker_tools, run_tools
from app.tools.registry import RunContext, ToolError, registry


class FakeImage:
    def __init__(self, id="sha256:abc123", ports=("8080/tcp",)):
        self.id = id
        self.short_id = id[:19]
        self.attrs = {"Config": {"ExposedPorts": {p: {} for p in ports}}}


class FakeContainer:
    def __init__(self, name, labels=None, status="running", logs=(), exit_code=0, host_port=None):
        self.id = f"id-{name}"
        self.name = name
        self.labels = labels or {}
        self.status = status
        self.script = list(logs)  # log text added on each poll
        self.log = b""
        self.exit_code = exit_code
        self.removed = self.stopped = False
        self.attrs = {"State": {"ExitCode": exit_code},
                      "HostConfig": {"PortBindings": {"8080/tcp": [{"HostPort": str(host_port)}]} if host_port else {}}}

    def reload(self):
        if self.script:
            chunk = self.script.pop(0)
            if chunk == "EXIT":
                self.status = "exited"
            else:
                self.log += chunk.encode()

    def logs(self, stdout=True, stderr=True):
        return self.log

    def remove(self, force=False):
        self.removed = True

    def rename(self, name):
        self.name = name

    def start(self):
        self.status = "running"

    def stop(self, timeout=10):
        self.stopped = True


class FakeDocker:
    def __init__(self, image=None, existing=None, script=(), exit_code=0):
        self.image = image
        self.existing = existing
        self.script = script
        self.exit_code = exit_code
        self.started = []
        outer = self

        class Images:
            def get(self, ref):
                if outer.image is None:
                    raise NotFound("no image")
                return outer.image

        class Containers:
            def list(self, all=False, filters=None):
                e = outer.existing
                return [e] if e is not None and not e.removed and filters["name"] == f"^/{e.name}$" else []

            def get(self, id):
                if outer.existing is not None and outer.existing.id == id:
                    return outer.existing
                raise NotFound("gone")

            def run(self, image, **kwargs):
                outer.started.append({"image": image, **kwargs})
                c = FakeContainer(kwargs["name"], kwargs["labels"], logs=outer.script, exit_code=outer.exit_code)
                return c

        self.images = Images()
        self.containers = Containers()


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "shop-api"
    root.mkdir()
    (root / ".env").write_text('APP_API_KEY="super-secret-value-123"\nAPP_PORT=8080\n')
    store = Store("sqlite://")
    store.add_project("shop-api", str(root.resolve()))
    code_tools.configure(store)
    run_tools.configure(RunSettings.model_construct(**{**RunSettings().model_dump(), "health_timeout_s": 2,
                                                       "health_interval_s": 0.05, "health_failure_threshold": 3}))
    monkeypatch.setattr(run_tools, "port_free", lambda port, address: True)
    return root


def use(monkeypatch, fake, answers=None):
    """answers: list of HTTP statuses (or None) the probe returns, one per call."""
    monkeypatch.setattr(docker_tools, "get_client", lambda: fake)
    queue = list(answers or [])

    async def probe(url):
        return queue.pop(0) if queue else None

    monkeypatch.setattr(run_tools, "_probe", probe)


def preview():
    tool = registry.get("project_run_container")
    return tool.preview(tool.args_model.model_validate({"project": "shop-api"}))


async def run(prev=None):
    tool = registry.get("project_run_container")
    args = tool.args_model.model_validate({"project": "shop-api"})
    lines = []

    async def progress(line):
        lines.append(line)

    result = await tool.call(args, prev or tool.preview(args), RunContext(progress=progress))
    return result, lines


# ---- pre-check -----------------------------------------------------------------------------------


def test_card_shows_port_container_and_setting_names_only(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage()))
    p = preview()
    assert p["facts"]["Port"] == "http://localhost:8080 → container port 8080 (this Mac only)"
    assert p["facts"]["Container"] == "shop-api"
    assert p["facts"]["Settings from .env"] == "APP_API_KEY, APP_PORT (values not shown)"
    assert "super-secret-value-123" not in json.dumps(p)


def test_no_image_or_no_expose_is_refused(project, monkeypatch):
    use(monkeypatch, FakeDocker(None))
    with pytest.raises(ToolError, match="Build it first with project_build_image"):
        preview()
    use(monkeypatch, FakeDocker(FakeImage(ports=())))
    with pytest.raises(ToolError, match="doesn't EXPOSE a port"):
        preview()


def test_containers_we_didnt_create_are_never_touched(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage(), existing=FakeContainer("shop-api", labels={})))
    with pytest.raises(ToolError, match="I didn't create it, so I won't touch it"):
        preview()


def test_our_own_previous_container_is_replaced(project, monkeypatch):
    old = FakeContainer("shop-api", labels={run_tools.LABEL: "shop-api"}, status="exited", host_port=8080)
    fake = FakeDocker(FakeImage(), existing=old, script=["up\n"])
    use(monkeypatch, fake, answers=[200])
    p = preview()
    assert p["facts"]["Replaces"] == "the container 'shop-api' I started earlier (exited)"
    assert p["facts"]["Port"].startswith("http://localhost:8080")  # its own port is reused


async def test_a_busy_port_moves_to_the_next_free_one(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage()))
    monkeypatch.setattr(run_tools, "port_free", lambda port, address: port != 8080)
    p = preview()
    assert p["run"]["host_port"] == 8081 and "in use on this Mac" in p["warnings"][0]


def test_a_port_setting_that_differs_from_expose_is_warned_without_its_value(project, monkeypatch):
    (project / ".env").write_text("APP_PORT=9000\n")
    use(monkeypatch, FakeDocker(FakeImage()))
    warning = next(w for w in preview()["warnings"] if "APP_PORT" in w)
    assert "9000" not in warning and "different port than the image EXPOSEs (8080)" in warning


# ---- running --------------------------------------------------------------------------------------


async def test_a_healthy_app_reports_its_url(project, monkeypatch):
    fake = FakeDocker(FakeImage(), script=["Starting with key super-secret-value-123\n", "Uvicorn running on http://0.0.0.0:8080\n"])
    # attempt 1: nothing listens yet; attempt 2: no /health or /healthz route, / answers 404 (the fallback)
    use(monkeypatch, fake, answers=[None, 404, 404, 404])
    result, lines = await run()
    assert result["ok"] and result["url"] == "http://localhost:8080"
    assert result["facts"]["Answered"].startswith("HTTP 404 on /, attempt 2, after")
    started = fake.started[0]
    assert started["ports"] == {"8080/tcp": ("127.0.0.1", 8080)}
    assert started["environment"]["APP_API_KEY"] == "super-secret-value-123"  # only Docker gets the value
    assert started["labels"][run_tools.LABEL] == "shop-api"
    assert "super-secret-value-123" not in json.dumps(result) and "super-secret-value-123" not in "".join(lines)


async def test_a_stopped_old_container_is_removed_before_starting(project, monkeypatch):
    old = FakeContainer("shop-api", labels={run_tools.LABEL: "shop-api"}, status="exited")
    fake = FakeDocker(FakeImage(), existing=old, script=["up\n"])
    use(monkeypatch, fake, answers=[200])
    await run()
    assert old.removed and len(fake.started) == 1


async def test_an_exiting_app_explains_why(project, monkeypatch):
    fake = FakeDocker(FakeImage(), script=["Traceback...\nModuleNotFoundError: No module named 'uvicorn'\n", "EXIT"], exit_code=1)
    use(monkeypatch, fake)
    with pytest.raises(ToolError) as err:
        await run()
    message = str(err.value)
    assert "stopped right away (exit code 1)" in message
    assert "Python can't import `uvicorn`" in message
    assert "ModuleNotFoundError" in err.value.result["log"]


async def test_an_app_listening_on_the_wrong_address_and_port_is_explained_and_stopped(project, monkeypatch):
    fake = FakeDocker(FakeImage(), script=["Uvicorn running on http://127.0.0.1:8000\n"])
    use(monkeypatch, fake)
    with pytest.raises(ToolError) as err:
        await run()
    message = str(err.value)
    assert "nothing answered on http://localhost:8080 in " in message and "attempts (2 s). I stopped it." in message
    assert "listens on 127.0.0.1 inside the container" in message
    assert "listens on port 8000, but the image EXPOSEs 8080" in message


async def test_an_image_rebuilt_after_approval_is_not_run(project, monkeypatch):
    fake = FakeDocker(FakeImage())
    use(monkeypatch, fake)
    approved = preview()
    fake.image = FakeImage(id="sha256:newer")
    with pytest.raises(ToolError, match="rebuilt after you approved"):
        await run(approved)
    assert fake.started == []


def test_exit_code_127_is_explained():
    notes = run_tools.diagnose("", 127, 8080, answered=False)
    assert notes == ["Exit code 127: the start command wasn't found in the image (CMD/ENTRYPOINT names a program that isn't installed)."]



# ---- health check rules ----------------------------------------------------------------------------


async def test_an_unhealthy_health_endpoint_fails_even_if_another_path_would_answer(project, monkeypatch):
    fake = FakeDocker(FakeImage(), script=["Traceback (most recent call last):\nRuntimeError: broken on purpose\n"])
    use(monkeypatch, fake, answers=[500, 500, 500, 404])  # /healthz would say 404: it must never be asked
    with pytest.raises(ToolError) as err:
        await run()
    message = str(err.value)
    assert "/health answered HTTP 500 3 times in a row (attempts 1–3 of the health check)" in message
    assert "The app logged an error: RuntimeError: broken on purpose" in message
    assert "I stopped it." in message


async def test_one_error_while_starting_is_tolerated(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage(), script=["up\n"]), answers=[503, 200])
    result, _ = await run()
    assert result["facts"]["Answered"].startswith("HTTP 200 on /health, attempt 2")


async def test_a_missing_health_route_falls_through_to_the_next_path(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage(), script=["up\n"]), answers=[404, 200])
    result, _ = await run()
    assert result["facts"]["Answered"].startswith("HTTP 200 on /healthz, attempt 1")


def test_the_card_states_the_health_check_rule(project, monkeypatch):
    use(monkeypatch, FakeDocker(FakeImage()))
    assert preview()["facts"]["Health check"] == (
        "GET /health → /healthz → / every 0.05 s, up to 40 attempts; 3 error answers (5xx) in a row fail it"
    )
