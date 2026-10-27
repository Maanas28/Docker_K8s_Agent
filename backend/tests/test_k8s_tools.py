"""Kubernetes read tools against a fake cluster API (no real cluster needed)."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from kubernetes.client.rest import ApiException

from app.config import KubernetesSettings
from app.tools import k8s_tools
from app.tools.registry import ToolError, registry

NOW = datetime.now(timezone.utc)


def ago(minutes):
    return NOW - timedelta(minutes=minutes)


def state(running=None, waiting=None, terminated=None):
    return NS(running=running, waiting=waiting, terminated=terminated)


def container_status(name="web", ready=True, restarts=0, st=None, last=None, image="shop:v2"):
    return NS(name=name, ready=ready, restart_count=restarts, image=image,
              state=st or state(running=NS(started_at=ago(5))), last_state=last or state())


def pod(name, statuses, phase="Running", conditions=(), env=(), limits=None, probe=True):
    spec_container = NS(
        name="web", image="shop:v2", ports=[NS(container_port=8080, protocol="TCP")],
        env=[NS(name=n) for n in env], env_from=[NS(config_map_ref=None, secret_ref=NS(name="app-secrets"))],
        resources=NS(requests={"memory": "64Mi"}, limits=limits or {}),
        readiness_probe=NS(http_get=NS(path="/health", port=8080)) if probe else None, liveness_probe=None,
    )
    return NS(
        metadata=NS(name=name, namespace="default", creation_timestamp=ago(30), deletion_timestamp=None),
        spec=NS(containers=[spec_container], node_name="docker-desktop"),
        status=NS(phase=phase, container_statuses=statuses, conditions=list(conditions), start_time=ago(30)),
    )


def event(reason, message, type="Warning", minutes=1, name="web-1"):
    return NS(reason=reason, message=message, type=type, count=3, last_timestamp=ago(minutes),
              metadata=NS(creation_timestamp=ago(minutes)), involved_object=NS(kind="Pod", name=name))


class Core:
    def __init__(self, pods=(), events=(), logs="", fail=None):
        self.pods = {p.metadata.name: p for p in pods}
        self.events = list(events)
        self.logs = logs
        self.fail = fail
        self.calls = []

    def _maybe_fail(self):
        if self.fail:
            raise self.fail

    def list_namespaced_pod(self, namespace, _request_timeout=None, **kw):
        self._maybe_fail()
        self.calls.append(("list_pods", namespace, kw))
        return NS(items=list(self.pods.values()))

    def list_pod_for_all_namespaces(self, _request_timeout=None, **kw):
        return NS(items=list(self.pods.values()))

    def read_namespaced_pod(self, name, namespace, _request_timeout=None):
        self._maybe_fail()
        if name not in self.pods:
            raise ApiException(status=404, reason="Not Found")
        return self.pods[name]

    def list_namespaced_event(self, namespace, _request_timeout=None, field_selector=None):
        self.calls.append(("events", namespace, field_selector))
        return NS(items=self.events)

    def read_namespaced_pod_log(self, name, namespace, container, previous, tail_lines, _request_timeout=None):
        self.calls.append(("logs", name, container, previous, tail_lines))
        if previous and not self.logs:
            raise ApiException(status=400, reason="Bad Request")
        return self.logs


@pytest.fixture
def cluster(monkeypatch):
    k8s_tools.configure(KubernetesSettings())

    def use(core=None, apps=None, version=None):
        fake = k8s_tools.Apis("docker-desktop", core or Core(), apps, version)
        monkeypatch.setattr(k8s_tools, "apis", lambda: fake)
        return fake
    return use


def call(name, **kwargs):
    tool = registry.get(name)
    return tool.func(tool.args_model.model_validate(kwargs))


# ---- which cluster --------------------------------------------------------------------------------


def contexts(monkeypatch, names, active):
    monkeypatch.setattr(k8s_tools.k8s_config, "list_kube_config_contexts",
                        lambda config_file=None: ([{"name": n} for n in names], {"name": active} if active else None))


def test_only_allowed_contexts_are_used(monkeypatch):
    k8s_tools.configure(KubernetesSettings())
    contexts(monkeypatch, ["prod-eu", "docker-desktop"], active="prod-eu")
    assert k8s_tools.pick_context() == "docker-desktop"  # the current context isn't allowed, so it's skipped
    contexts(monkeypatch, ["prod-eu"], active="prod-eu")
    with pytest.raises(ToolError, match=r"None of the Kubernetes contexts I'm allowed to use \(docker-desktop\)"):
        k8s_tools.pick_context()


def test_no_kubeconfig_says_how_to_turn_kubernetes_on(monkeypatch):
    k8s_tools.configure(KubernetesSettings())

    def missing(config_file=None):
        raise k8s_tools.ConfigException("Invalid kube-config file. No configuration found.")

    monkeypatch.setattr(k8s_tools.k8s_config, "list_kube_config_contexts", missing)
    with pytest.raises(ToolError, match="Turn on Kubernetes in Docker Desktop"):
        k8s_tools.pick_context()


def test_kubernetes_tools_only_read():
    from app.tools import load_all
    load_all()
    tools = [t for t in registry.all() if t.platform == "kubernetes"]
    assert len(tools) == 8 and all(t.risk.value == "read" for t in tools)
    assert registry.supports("kubernetes", "read") and not registry.supports("kubernetes", "change")


# ---- pods -------------------------------------------------------------------------------------------


def test_pod_list_shows_kubectl_style_status(cluster):
    crashing = pod("web-1", [container_status(ready=False, restarts=6, st=state(waiting=NS(reason="CrashLoopBackOff", message="back-off 5m")))])
    healthy = pod("web-2", [container_status()])
    cluster(Core(pods=[crashing, healthy]))
    rows = {r["name"]: r for r in call("k8s_list_pods")["rows"]}
    assert rows["web-1"] == {"name": "web-1", "ready": "0/1", "status": "CrashLoopBackOff", "restarts": 6, "age": "30m", "node": "docker-desktop"}
    assert rows["web-2"]["status"] == "Running" and rows["web-2"]["ready"] == "1/1"


def test_describe_explains_a_crash_loop_and_hides_env_values(cluster):
    crashing = pod(
        "web-1",
        [container_status(ready=False, restarts=6, st=state(waiting=NS(reason="CrashLoopBackOff", message="back-off")),
                          last=state(terminated=NS(reason="Error", exit_code=1)))],
        env=["APP_API_KEY", "DATABASE_PASSWORD"],
    )
    cluster(Core(pods=[crashing], events=[event("BackOff", "Back-off restarting failed container")]))
    data = call("k8s_describe_pod", pod="web-1")["data"]
    assert data["status"] == "CrashLoopBackOff"
    assert data["why"][0].startswith("Container web keeps crashing (exit code 1, Error) and has restarted 6 times")
    assert "previous=true" in data["why"][0]
    web = data["containers"][0]
    assert web["env_names"] == ["APP_API_KEY", "DATABASE_PASSWORD"] and web["env_from"] == ["secret/app-secrets"]
    assert web["last_state"] == "terminated: Error (exit code 1)" and web["probe"] == "readiness: GET /health on port 8080"
    assert "Back-off restarting failed container (x3)" in data["events"][0]


@pytest.mark.parametrize("pod_obj, expected", [
    (pod("p", [container_status(ready=False, st=state(waiting=NS(reason="ImagePullBackOff", message='pull access denied for shop:v9')))]),
     "can't get its image shop:v2: pull access denied for shop:v9"),
    (pod("p", [container_status(ready=False, st=state(waiting=NS(reason="CreateContainerConfigError", message='secret "db" not found')))]),
     'can\'t be created: secret "db" not found. Usually a Secret, ConfigMap or key'),
    (pod("p", [container_status(restarts=2, last=state(terminated=NS(reason="OOMKilled", exit_code=137)))], limits={"memory": "128Mi"}),
     "was killed for using too much memory (OOMKilled); its memory limit is 128Mi."),
    (pod("p", [], phase="Pending", conditions=[NS(type="PodScheduled", status="False", reason="Unschedulable",
                                                  message="0/1 nodes are available: 1 Insufficient memory.")]),
     "can't be placed on a node: 0/1 nodes are available: 1 Insufficient memory."),
])
def test_why_facts_for_common_failures(pod_obj, expected):
    assert any(expected in note for note in k8s_tools.explain_pod(pod_obj, []))


def test_failing_probes_are_explained_from_events():
    p = pod("p", [container_status(ready=False)])
    notes = k8s_tools.explain_pod(p, [event("Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 500")])
    assert notes == ["A health probe is failing: Readiness probe failed: HTTP probe failed with statuscode: 500"]


def test_unknown_pod_is_a_clear_error(cluster):
    cluster(Core())
    with pytest.raises(ToolError, match="Pod 'nope' in namespace 'default' doesn't exist"):
        call("k8s_describe_pod", pod="nope")


# ---- logs -------------------------------------------------------------------------------------------


def test_previous_logs_are_masked(cluster):
    core = Core(pods=[pod("web-1", [container_status()])],
                logs="Traceback...\nRuntimeError: token=ghp_" + "a" * 36 + "\n")
    cluster(core)
    result = call("k8s_pod_logs", pod="web-1", previous=True)
    assert result["title"] == "web-1/web (previous run)"
    assert "ghp_" not in result["text"] and "RuntimeError" in result["text"]
    assert core.calls[-1] == ("logs", "web-1", "web", True, 200)


def test_no_previous_run_is_explained(cluster):
    cluster(Core(pods=[pod("web-1", [container_status()])], logs=""))
    with pytest.raises(ToolError, match="No logs"):
        call("k8s_pod_logs", pod="web-1", previous=True)


# ---- the rest ---------------------------------------------------------------------------------------


def test_deployments_services_and_events(cluster):
    deploy = NS(metadata=NS(name="web", creation_timestamp=ago(120)),
                spec=NS(replicas=2, template=NS(spec=NS(containers=[NS(image="shop:v2")]))),
                status=NS(ready_replicas=1, updated_replicas=2, available_replicas=1))
    service = NS(metadata=NS(name="web", creation_timestamp=ago(120)),
                 spec=NS(type="NodePort", cluster_ip="10.96.0.12", selector={"app": "web"},
                         ports=[NS(port=80, node_port=30080, protocol="TCP")]))

    class Core2(Core):
        def list_namespaced_service(self, namespace, _request_timeout=None):
            return NS(items=[service])

    apps = NS(list_namespaced_deployment=lambda namespace, _request_timeout=None: NS(items=[deploy]))
    core = Core2(events=[event("BackOff", "Back-off", minutes=5), event("Failed", "ErrImagePull", minutes=1)])
    cluster(core, apps=apps)
    assert call("k8s_list_deployments")["rows"][0] == {"name": "web", "ready": "1/2", "up-to-date": 2, "available": 1, "images": "shop:v2", "age": "2h"}
    assert call("k8s_list_services")["rows"][0]["ports"] == "80:30080/TCP"
    events = call("k8s_events")["rows"]
    assert [e["reason"] for e in events] == ["Failed", "BackOff"]  # newest first
    assert core.calls[-1] == ("events", "default", "type=Warning")


def test_an_unreachable_cluster_is_explained(cluster):
    import urllib3
    cluster(Core(fail=urllib3.exceptions.MaxRetryError(None, "/api", "connection refused")))
    with pytest.raises(ToolError, match="Can't reach the Kubernetes cluster"):
        call("k8s_list_pods")


def test_results_serialize_for_the_model(cluster):
    cluster(Core(pods=[pod("web-1", [container_status()])]))
    json.dumps(call("k8s_describe_pod", pod="web-1"))  # no objects that can't be sent to the model



def test_system_pods_are_hidden_when_listing_all_namespaces(cluster):
    mine = pod("web-1", [container_status()])
    system = pod("coredns-1", [container_status()])
    system.metadata.namespace = "kube-system"
    cluster(Core(pods=[mine, system]))
    result = call("k8s_list_pods", all_namespaces=True)
    assert [r["name"] for r in result["rows"]] == ["web-1"]
    assert result["note"].startswith("+1 Kubernetes system pods hidden")
    shown = call("k8s_list_pods", all_namespaces=True, include_system=True)
    assert [r["name"] for r in shown["rows"]] == ["web-1", "coredns-1"] and "note" not in shown
    tool = registry.get("k8s_list_pods")
    assert tool.summarize(tool.args_model(all_namespaces=True)) == "List pods in all namespaces"
    assert "(including system pods)" in tool.summarize(tool.args_model(all_namespaces=True, include_system=True))
