"""Kubernetes, read-only (Phase 3): see what runs in a cluster, and why it fails.

Talks to the cluster through the official Kubernetes Python SDK. Safety:
- only kubeconfig contexts listed in agent.toml [kubernetes] allowed_contexts are ever used, and
  code (not the model) picks the context; kubeconfig credentials never reach the model or the UI;
- nothing is changed (changes are Phase 4); Secret values are never read, and environment
  variables in pod specs are shown by name only.

`k8s_describe_pod` is our own "kubectl describe": state, conditions, events, plus "why" facts for
the usual failures (CrashLoopBackOff, image pull errors, OOMKilled, unschedulable, failing probes).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import urllib3
from kubernetes import client as k8s
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException
from pydantic import Field

from app.config import KubernetesSettings
from app.redact import redact
from app.tools.registry import Risk, ToolArgs, ToolError, registry

_settings = KubernetesSettings()


@dataclass
class Apis:
    context: str
    core: Any
    apps: Any
    version: Any


_cache: dict[str, Apis] = {}


def configure(settings: KubernetesSettings) -> None:
    global _settings
    _settings = settings
    _cache.clear()


def pick_context() -> str:
    """The context to use: the current one if it's allowed, else the first allowed one that exists."""
    try:
        contexts, active = k8s_config.list_kube_config_contexts(config_file=_settings.kubeconfig)
    except (ConfigException, OSError):
        raise ToolError(
            "There's no Kubernetes configuration on this Mac (~/.kube/config). Turn on Kubernetes in "
            "Docker Desktop (Settings → Kubernetes → Enable Kubernetes), then try again."
        ) from None
    names = [c["name"] for c in contexts or []]
    if active and active.get("name") in _settings.allowed_contexts:
        return active["name"]
    for name in _settings.allowed_contexts:
        if name in names:
            return name
    raise ToolError(
        f"None of the Kubernetes contexts I'm allowed to use ({', '.join(_settings.allowed_contexts)}) "
        f"is configured on this Mac ({len(names)} other context(s) exist). Turn on Kubernetes in Docker "
        "Desktop, or add a context to [kubernetes] allowed_contexts in agent.toml (only clusters it's "
        "fine for the agent to read)."
    )


def apis() -> Apis:
    context = pick_context()
    if context not in _cache:
        try:
            api = k8s_config.new_client_from_config(config_file=_settings.kubeconfig, context=context)
        except (ConfigException, OSError) as e:
            raise ToolError(f"Can't load the Kubernetes context '{context}': {e}") from None
        _cache[context] = Apis(context, k8s.CoreV1Api(api), k8s.AppsV1Api(api), k8s.VersionApi(api))
    return _cache[context]


def _api_message(e: ApiException) -> str:
    try:
        return json.loads(e.body or "{}").get("message") or e.reason or ""
    except ValueError:
        return (e.body or e.reason or "")[:300]


@contextmanager
def _k8s_errors(what: str = "") -> Iterator[None]:
    try:
        yield
    except ApiException as e:
        if e.status == 404:
            raise ToolError(f"{what or 'That object'} doesn't exist (check the name and the namespace).") from None
        if e.status in (401, 403):
            raise ToolError(f"The cluster refused access (HTTP {e.status}): {_api_message(e)}") from None
        raise ToolError(f"Kubernetes answered HTTP {e.status}: {_api_message(e)}") from None
    except (urllib3.exceptions.HTTPError, OSError):
        _cache.clear()  # reconnect next time
        raise ToolError(
            "Can't reach the Kubernetes cluster. Is Kubernetes running in Docker Desktop (green "
            "'Kubernetes running' in its status bar)?"
        ) from None


def ping() -> dict[str, Any]:
    """For the header: which context, and does it answer."""
    try:
        a = apis()
        a.version.get_code(_request_timeout=2)
        return {"ok": True, "context": a.context}
    except Exception:  # noqa: BLE001  (any failure just means "not reachable")
        _cache.clear()
        try:
            return {"ok": False, "context": pick_context()}
        except ToolError:
            return {"ok": False, "context": None}


def _timeout() -> int:
    return _settings.request_timeout_s


# ---- formatting -------------------------------------------------------------------------------


def _age(ts: datetime | None) -> str:
    if ts is None:
        return ""
    seconds = int((datetime.now(timezone.utc) - ts).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{max(seconds, 0)}s"


def _statuses(pod) -> list:
    return list(pod.status.container_statuses or []) if pod.status else []


def pod_status(pod) -> str:
    """The one-word status kubectl shows: a container's waiting/terminated reason beats the phase."""
    if pod.metadata.deletion_timestamp:
        return "Terminating"
    for cs in _statuses(pod):
        state = cs.state
        if state and state.waiting and state.waiting.reason:
            return state.waiting.reason
        if state and state.terminated and state.terminated.reason:
            return state.terminated.reason
    return (pod.status.phase if pod.status else None) or "Unknown"


def _ready(pod) -> str:
    statuses = _statuses(pod)
    return f"{sum(1 for cs in statuses if cs.ready)}/{len(pod.spec.containers)}"


def _state(state) -> str:
    if state is None:
        return "unknown"
    if state.running:
        return f"running for {_age(state.running.started_at)}"
    if state.waiting:
        return f"waiting: {state.waiting.reason}" + (f" ({state.waiting.message})" if state.waiting.message else "")
    if state.terminated:
        t = state.terminated
        return f"terminated: {t.reason or 'exited'} (exit code {t.exit_code})"
    return "unknown"


def _event_time(e) -> datetime | None:
    return e.last_timestamp or getattr(e, "event_time", None) or e.metadata.creation_timestamp


# ---- why a pod isn't healthy -----------------------------------------------------------------------


def explain_pod(pod, events: list) -> list[str]:
    """Plain-language facts about why a pod isn't healthy, from fields Kubernetes reports."""
    notes: list[str] = []
    limits = {c.name: ((c.resources.limits or {}) if c.resources else {}) for c in pod.spec.containers}
    for cs in _statuses(pod):
        waiting = cs.state.waiting if cs.state else None
        last = cs.last_state.terminated if cs.last_state else None
        reason = waiting.reason if waiting else None
        if reason == "CrashLoopBackOff":
            how = f"exit code {last.exit_code}" + (f", {last.reason}" if last and last.reason else "") if last else "it exits"
            notes.append(
                f"Container {cs.name} keeps crashing ({how}) and has restarted {cs.restart_count} times; "
                "Kubernetes now waits longer before each restart. The logs of its previous run "
                "(k8s_pod_logs with previous=true) show why."
            )
        elif reason in ("ImagePullBackOff", "ErrImagePull", "InvalidImageName", "ErrImageNeverPull"):
            notes.append(
                f"Container {cs.name} can't get its image {cs.image}: {waiting.message or reason}. Check the "
                "name and tag. For an image built on this Mac, Docker Desktop's cluster can use it "
                "directly with imagePullPolicy: IfNotPresent (or Never) and a tag that exists locally."
            )
        elif reason in ("CreateContainerConfigError", "CreateContainerError"):
            notes.append(
                f"Container {cs.name} can't be created: {waiting.message or reason}. Usually a Secret, "
                "ConfigMap or key it refers to doesn't exist."
            )
        terminated = (cs.state.terminated if cs.state else None) or last
        if terminated is not None and terminated.reason == "OOMKilled":
            limit = limits.get(cs.name, {}).get("memory")
            notes.append(f"Container {cs.name} was killed for using too much memory (OOMKilled)"
                         + (f"; its memory limit is {limit}." if limit else "."))
    if pod.status and pod.status.phase == "Pending":
        unscheduled = next((c for c in pod.status.conditions or []
                            if c.type == "PodScheduled" and c.status == "False"), None)
        if unscheduled is not None:
            notes.append(f"The pod can't be placed on a node: {unscheduled.message or unscheduled.reason}.")
    probe = next((e for e in sorted(events, key=lambda e: _event_time(e) or datetime.min.replace(tzinfo=timezone.utc),
                                    reverse=True) if e.reason == "Unhealthy"), None)
    if probe is not None:
        notes.append(f"A health probe is failing: {probe.message}")
    return notes


# ---- tools --------------------------------------------------------------------------------------


class NoArgs(ToolArgs):
    pass


class NamespaceArgs(ToolArgs):
    namespace: str = Field("default", description="Kubernetes namespace")


class ListPodsArgs(ToolArgs):
    namespace: str = Field("default", description="Kubernetes namespace (the user's own pods are usually in default)")
    all_namespaces: bool = Field(
        False,
        description="Only when the user asks about every namespace. For 'my pods' or a single app, leave it false.",
    )
    include_system: bool = Field(
        False,
        description="With all_namespaces: also show Kubernetes' own system pods (kube-system…). Only when the "
                    "user asks about system pods or cluster components.",
    )
    label_selector: str | None = Field(None, description='Only pods with these labels, e.g. "app=web"')


# Namespaces that hold Kubernetes' own components, not the user's apps.
SYSTEM_NAMESPACES = {"kube-system", "kube-public", "kube-node-lease"}


class PodArgs(ToolArgs):
    pod: str = Field(description="Pod name, exactly as listed by k8s_list_pods")
    namespace: str = Field("default", description="Kubernetes namespace")


class PodLogsArgs(ToolArgs):
    pod: str = Field(description="Pod name, exactly as listed by k8s_list_pods")
    namespace: str = Field("default", description="Kubernetes namespace")
    container: str | None = Field(None, description="Container name, only needed if the pod has several")
    tail: int | None = Field(None, ge=1, le=2000, description="How many of the last lines to fetch")
    previous: bool = Field(False, description="Logs of the previous run of a container that crashed and restarted")


class EventsArgs(ToolArgs):
    namespace: str = Field("default", description="Kubernetes namespace")
    warnings_only: bool = Field(True, description="Only Warning events (the ones that explain problems)")


@registry.tool(risk=Risk.READ, summary=lambda a: "Show the Kubernetes cluster and its nodes", platform="kubernetes")
def k8s_cluster_info(args: NoArgs) -> dict:
    """Which Kubernetes cluster (context) I use, its version, and its nodes with their status."""
    a = apis()
    with _k8s_errors():
        version = a.version.get_code(_request_timeout=_timeout())
        nodes = a.core.list_node(_request_timeout=_timeout()).items
    rows = []
    for n in nodes:
        ready = next((c.status for c in n.status.conditions or [] if c.type == "Ready"), "Unknown")
        roles = [k.split("/", 1)[1] for k in (n.metadata.labels or {}) if k.startswith("node-role.kubernetes.io/")]
        rows.append(f"{n.metadata.name} ({'Ready' if ready == 'True' else 'NotReady'}"
                    f"{', ' + ', '.join(roles) if roles else ''}, {n.status.node_info.kubelet_version})")
    return {"kind": "facts", "title": f"Kubernetes cluster: {a.context}",
            "facts": {"Context": a.context, "Kubernetes version": version.git_version, "Nodes": "; ".join(rows) or "none"}}


@registry.tool(risk=Risk.READ, summary=lambda a: "List Kubernetes namespaces", platform="kubernetes")
def k8s_list_namespaces(args: NoArgs) -> dict:
    """List the namespaces in the cluster."""
    a = apis()
    with _k8s_errors():
        items = a.core.list_namespace(_request_timeout=_timeout()).items
    rows = [{"name": n.metadata.name, "status": n.status.phase, "age": _age(n.metadata.creation_timestamp)} for n in items]
    return {"kind": "table", "columns": ["name", "status", "age"], "rows": rows}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: (f"List pods in all namespaces{' (including system pods)' if a.include_system else ''}"
                       if a.all_namespaces else f"List pods in namespace '{a.namespace}'"),
    platform="kubernetes",
)
def k8s_list_pods(args: ListPodsArgs) -> dict:
    """List pods with how many containers are ready, their status (e.g. Running, CrashLoopBackOff, ImagePullBackOff, Pending), restarts, age and node."""
    a = apis()
    kwargs = {"label_selector": args.label_selector} if args.label_selector else {}
    with _k8s_errors(f"Namespace '{args.namespace}'"):
        if args.all_namespaces:
            pods = a.core.list_pod_for_all_namespaces(_request_timeout=_timeout(), **kwargs).items
        else:
            pods = a.core.list_namespaced_pod(args.namespace, _request_timeout=_timeout(), **kwargs).items
    hidden = 0
    if args.all_namespaces and not args.include_system:
        # Even if the model over-reaches with all_namespaces, the user sees their own pods.
        kept = [p for p in pods if p.metadata.namespace not in SYSTEM_NAMESPACES]
        hidden, pods = len(pods) - len(kept), kept
    columns = (["namespace"] if args.all_namespaces else []) + ["name", "ready", "status", "restarts", "age", "node"]
    rows = [{
        "namespace": p.metadata.namespace, "name": p.metadata.name, "ready": _ready(p), "status": pod_status(p),
        "restarts": sum(cs.restart_count for cs in _statuses(p)), "age": _age(p.metadata.creation_timestamp),
        "node": p.spec.node_name or "",
    } for p in pods]
    result = {"kind": "table", "columns": columns, "rows": [{c: r[c] for c in columns} for r in rows]}
    if hidden:
        result["note"] = (f"+{hidden} Kubernetes system pods hidden ({', '.join(sorted(SYSTEM_NAMESPACES))}). "
                          "Ask for system pods to see them.")
    return result


@registry.tool(risk=Risk.READ, summary=lambda a: f"Describe pod '{a.pod}' ({a.namespace})", platform="kubernetes")
def k8s_describe_pod(args: PodArgs) -> dict:
    """Details of one pod: each container's state (and why it's waiting or crashed), restarts, image, probes, resources, environment variable names (not values), conditions, recent events, plus plain-language reasons why it isn't healthy."""
    a = apis()
    with _k8s_errors(f"Pod '{args.pod}' in namespace '{args.namespace}'"):
        pod = a.core.read_namespaced_pod(args.pod, args.namespace, _request_timeout=_timeout())
        events = a.core.list_namespaced_event(args.namespace, field_selector=f"involvedObject.name={args.pod}",
                                              _request_timeout=_timeout()).items
    statuses = {cs.name: cs for cs in _statuses(pod)}
    containers = []
    for c in pod.spec.containers:
        cs = statuses.get(c.name)
        probe = c.readiness_probe or c.liveness_probe
        containers.append({
            "name": c.name,
            "image": c.image,
            "ready": bool(cs and cs.ready),
            "restarts": cs.restart_count if cs else 0,
            "state": _state(cs.state) if cs else "not started",
            "last_state": _state(cs.last_state) if cs and cs.last_state and (cs.last_state.terminated or cs.last_state.waiting) else None,
            "ports": [f"{p.container_port}/{p.protocol}" for p in c.ports or []],
            # Values can hold secrets: names (and where they come from) only.
            "env_names": [e.name for e in c.env or []],
            "env_from": [(f.config_map_ref and f"configmap/{f.config_map_ref.name}") or (f.secret_ref and f"secret/{f.secret_ref.name}")
                         for f in c.env_from or []],
            "resources": {"requests": dict(c.resources.requests or {}), "limits": dict(c.resources.limits or {})} if c.resources else {},
            "probe": f"{'readiness' if c.readiness_probe else 'liveness'}: "
                     + (f"GET {probe.http_get.path} on port {probe.http_get.port}" if probe and probe.http_get else "exec/tcp")
                     if probe else None,
        })
    recent = sorted(events, key=lambda e: _event_time(e) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:15]
    data = {
        "pod": pod.metadata.name,
        "namespace": pod.metadata.namespace,
        "status": pod_status(pod),
        "why": explain_pod(pod, events),
        "node": pod.spec.node_name,
        "started": _age(pod.status.start_time) + " ago" if pod.status and pod.status.start_time else None,
        "containers": containers,
        "conditions": {c.type: c.status + (f" ({c.reason})" if c.reason else "") for c in (pod.status.conditions or [])},
        "events": [f"{_age(_event_time(e))} ago {e.type} {e.reason}: {redact(e.message or '')[0]}"
                   + (f" (x{e.count})" if e.count and e.count > 1 else "") for e in recent],
    }
    return {"kind": "json", "title": f"Pod {pod.metadata.name}", "data": data}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Read {'previous ' if a.previous else ''}logs of pod '{a.pod}' ({a.namespace})",
    platform="kubernetes",
)
def k8s_pod_logs(args: PodLogsArgs) -> dict:
    """The last lines of a pod's logs. Use previous=true to see why a container crashed before it was restarted (CrashLoopBackOff)."""
    a = apis()
    container = args.container
    with _k8s_errors(f"Pod '{args.pod}' in namespace '{args.namespace}'"):
        if container is None:
            pod = a.core.read_namespaced_pod(args.pod, args.namespace, _request_timeout=_timeout())
            names = [c.name for c in pod.spec.containers]
            if len(names) > 1:
                raise ToolError(f"Pod '{args.pod}' has several containers ({', '.join(names)}); say which one.")
            container = names[0]
        try:
            text = a.core.read_namespaced_pod_log(
                args.pod, args.namespace, container=container, previous=args.previous,
                tail_lines=args.tail or _settings.log_tail_lines, _request_timeout=_timeout(),
            )
        except ApiException as e:
            if e.status == 400:  # e.g. no previous run, or the container hasn't started yet
                raise ToolError(f"No logs: {_api_message(e)}") from None
            raise
    masked, _ = redact(text or "")
    title = f"{args.pod}/{container}" + (" (previous run)" if args.previous else "")
    return {"kind": "logs", "title": title, "text": masked or "(no log output)"}


@registry.tool(risk=Risk.READ, summary=lambda a: f"List deployments in namespace '{a.namespace}'", platform="kubernetes")
def k8s_list_deployments(args: NamespaceArgs) -> dict:
    """List deployments with ready/desired replicas, up-to-date and available counts, and their images."""
    a = apis()
    with _k8s_errors(f"Namespace '{args.namespace}'"):
        items = a.apps.list_namespaced_deployment(args.namespace, _request_timeout=_timeout()).items
    rows = [{
        "name": d.metadata.name,
        "ready": f"{d.status.ready_replicas or 0}/{d.spec.replicas if d.spec.replicas is not None else 1}",
        "up-to-date": d.status.updated_replicas or 0,
        "available": d.status.available_replicas or 0,
        "images": ", ".join(c.image for c in d.spec.template.spec.containers),
        "age": _age(d.metadata.creation_timestamp),
    } for d in items]
    return {"kind": "table", "columns": ["name", "ready", "up-to-date", "available", "images", "age"], "rows": rows}


@registry.tool(risk=Risk.READ, summary=lambda a: f"List services in namespace '{a.namespace}'", platform="kubernetes")
def k8s_list_services(args: NamespaceArgs) -> dict:
    """List services with their type, cluster IP, ports (with node ports) and which pods they select."""
    a = apis()
    with _k8s_errors(f"Namespace '{args.namespace}'"):
        items = a.core.list_namespaced_service(args.namespace, _request_timeout=_timeout()).items
    rows = [{
        "name": s.metadata.name,
        "type": s.spec.type,
        "cluster ip": s.spec.cluster_ip,
        "ports": ", ".join(f"{p.port}{':' + str(p.node_port) if p.node_port else ''}/{p.protocol}" for p in s.spec.ports or []),
        "selects": ", ".join(f"{k}={v}" for k, v in (s.spec.selector or {}).items()),
        "age": _age(s.metadata.creation_timestamp),
    } for s in items]
    return {"kind": "table", "columns": ["name", "type", "cluster ip", "ports", "selects", "age"], "rows": rows}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Show {'warning ' if a.warnings_only else ''}events in namespace '{a.namespace}'",
    platform="kubernetes",
)
def k8s_events(args: EventsArgs) -> dict:
    """Recent events in a namespace, newest first (by default only warnings: failed scheduling, image pull errors, crashes, failing probes)."""
    a = apis()
    kwargs = {"field_selector": "type=Warning"} if args.warnings_only else {}
    with _k8s_errors(f"Namespace '{args.namespace}'"):
        items = a.core.list_namespaced_event(args.namespace, _request_timeout=_timeout(), **kwargs).items
    items = sorted(items, key=lambda e: _event_time(e) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:50]
    rows = [{
        "age": _age(_event_time(e)),
        "type": e.type,
        "reason": e.reason,
        "object": f"{(e.involved_object.kind or '').lower()}/{e.involved_object.name}",
        "message": redact(e.message or "")[0][:300],
        "count": e.count or 1,
    } for e in items]
    return {"kind": "table", "columns": ["age", "type", "reason", "object", "message", "count"], "rows": rows}
