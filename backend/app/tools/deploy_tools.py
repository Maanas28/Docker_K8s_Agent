"""Version history and cleanup of a project's images (Step 5).

Redeploy and rollback are project_run_container (safe switch, version="previous"); this module
adds what's needed around them:

- project_versions (read): every build of the project, whether it passed a health check, which one
  is running, and images left from before versions were recorded.
- project_prune_images (destructive, type the project name): remove old images, keeping the newest
  build, the running version and the last N versions that passed a health check. Images used by any
  container are never removed.
"""

from __future__ import annotations

from datetime import datetime

from docker.errors import NotFound
from pydantic import Field

from app.store import ImageVersion
from app.tools import docker_tools
from app.tools.build_tools import _repo_name
from app.tools.code_tools import _workspace, get_store
from app.tools.registry import Risk, ToolArgs, ToolError, registry
from app.tools.run_tools import LABEL, VERSION_LABEL, _existing

TITLE_LABEL = "org.opencontainers.image.title"  # set on every image project_build_image builds


class ProjectArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")


class PruneArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    keep: int = Field(3, ge=1, le=20, description="How many of the newest versions that passed a health check to keep")


def _mb(n: int | float | None) -> str:
    return f"{(n or 0) / 1_000_000:.0f} MB"


def _local(ts: datetime | None) -> str:
    return ts.astimezone().strftime("%b %d %H:%M") if ts else ""


def _project_images(client, project: str) -> list:
    """Every image built for the project, including untagged leftovers of earlier builds."""
    return client.images.list(all=False, filters={"label": f"{TITLE_LABEL}={project}"})


def _version_name(image, versions: list[ImageVersion]) -> str:
    record = next((v for v in versions if v.image_id == image.id), None)
    if record:
        return f"v{record.number}"
    tags = [t.split(":", 1)[-1] for t in image.tags if not t.endswith(":latest")]
    return ", ".join(tags) if tags else f"untagged {image.short_id.removeprefix('sha256:')[:12]}"


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"List the versions of '{a.project}'",
    platform="project",
    scope_field="project",
)
def project_versions(args: ProjectArgs) -> dict:
    """List every built version of a project (newest first): when it was built, its git commit, size, whether it passed a health check, and which one is running. Use it before a rollback or a cleanup."""
    ws = _workspace(args.project)
    versions = get_store().versions(ws.name)
    with docker_tools._docker_errors():
        client = docker_tools.get_client()
        images = {img.id: img for img in _project_images(client, ws.name)}
        container = _existing(client, _repo_name(ws.name))
    running_tag = (container.labels.get(VERSION_LABEL) if container is not None and container.status == "running"
                   and container.labels.get(LABEL) == ws.name else None)
    rows = []
    for v in versions:
        status = ("● running" if v.tag == running_tag else "") if v.image_id in images else "removed"
        rows.append({
            "version": f"v{v.number}",
            "built": _local(v.built_at),
            "commit": (v.commit or "—") + (" + uncommitted changes" if v.dirty else ""),
            "size": _mb(v.size_bytes),
            "health check": {True: "passed", False: "failed"}.get(v.healthy, "not run yet"),
            "status": status,
        })
    recorded = {v.image_id for v in versions}
    for img in images.values():
        if img.id not in recorded:  # built before versions were recorded
            rows.append({
                "version": _version_name(img, versions), "built": img.attrs.get("Created", "")[:16].replace("T", " "),
                "commit": img.labels.get("org.opencontainers.image.revision", "—"), "size": _mb(img.attrs.get("Size")),
                "health check": "unknown (from before version tracking)", "status": "",
            })
    if not rows:
        raise ToolError(f"No images have been built for {ws.name} yet. Build one with project_build_image.")
    return {"kind": "table", "columns": ["version", "built", "commit", "size", "health check", "status"], "rows": rows}


def _plan_prune(args: PruneArgs) -> dict:
    ws = _workspace(args.project)
    versions = get_store().versions(ws.name)
    with docker_tools._docker_errors():
        client = docker_tools.get_client()
        images = _project_images(client, ws.name)
        used = {(c.attrs.get("Image") or "") for c in client.containers.list(all=True)}
        try:
            latest_id = client.images.get(f"{_repo_name(ws.name)}:latest").id
        except NotFound:
            latest_id = None
    present = {img.id for img in images}
    healthy = [v.image_id for v in versions if v.healthy and v.image_id in present][: args.keep]
    keep_reasons: dict[str, str] = {}
    for image_id in used & present:
        keep_reasons[image_id] = "used by a container"
    if latest_id in present:
        keep_reasons.setdefault(latest_id, "newest build")
    for image_id in healthy:
        keep_reasons.setdefault(image_id, "passed its health check")
    numbers = {v.image_id: v.number for v in versions}
    # Recorded versions by number (v1, v2, …), then images from before versions were recorded.
    remove = sorted((img for img in images if img.id not in keep_reasons),
                    key=lambda img: (numbers.get(img.id, 10**9), img.tags[0] if img.tags else img.id))
    by_id = {img.id: img for img in images}
    return {
        "ws": ws,
        "keep": [(by_id[i], why) for i, why in keep_reasons.items()],
        "remove": remove,
        "versions": versions,
        "bytes": sum(img.attrs.get("Size") or 0 for img in remove),
    }


def _preview_prune(args: PruneArgs) -> dict:
    plan = _plan_prune(args)
    versions = plan["versions"]
    if not plan["remove"]:
        raise ToolError(
            f"Nothing to clean up for {plan['ws'].name}: every image is either the newest build, used by a "
            f"container, or one of the last {args.keep} versions that passed a health check."
        )
    return {
        "facts": {
            "Project": plan["ws"].name,
            "Removes": ", ".join(_version_name(img, versions) for img in plan["remove"]),
            "Frees": _mb(plan["bytes"]),
            "Keeps": "; ".join(f"{_version_name(img, versions)} ({why})" for img, why in plan["keep"]) or "nothing",
        },
        "warnings": ["Removed versions can't be rolled back to. Their builds would have to be redone."],
        "prune": {"ids": [img.id for img in plan["remove"]]},
    }


@registry.tool(
    risk=Risk.DESTRUCTIVE,
    summary=lambda a: f"Remove old images of '{a.project}' (keep the last {a.keep} healthy versions)",
    platform="project",
    confirm_field="project",
    preview=_preview_prune,
)
def project_prune_images(args: PruneArgs, preview: dict) -> dict:
    """Remove a project's old images to free disk space. Keeps the newest build, anything a container uses, and the last `keep` versions that passed a health check (default 3). The user must type the project name to confirm."""
    plan = _plan_prune(args)  # checked again now: only what was approved, and only if still unused
    approved = set(preview["prune"]["ids"])
    removed, freed, skipped = [], 0, []
    with docker_tools._docker_errors():
        client = docker_tools.get_client()
        for img in plan["remove"]:
            if img.id not in approved:
                continue  # wasn't on the card the user approved
            try:
                client.images.remove(img.id, force=True)  # force: also drops its extra tags
            except NotFound:
                continue
            removed.append(_version_name(img, plan["versions"]))
            freed += img.attrs.get("Size") or 0
        skipped = [_version_name(by, plan["versions"]) for by, _ in plan["keep"] if by.id in approved]
    facts = {"Removed": ", ".join(removed) or "nothing", "Freed": _mb(freed)}
    if skipped:
        facts["Kept after all"] = ", ".join(skipped) + " (now in use, or now among the kept versions)"
    return {"kind": "facts", "title": f"Cleaned up {plan['ws'].name}", "facts": facts}
