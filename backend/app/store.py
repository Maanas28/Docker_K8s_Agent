"""Persistent state: registered projects, "always allow" grants, the audit log and model usage."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select


def _now() -> datetime:
    return datetime.now(UTC)


class AlwaysAllow(SQLModel, table=True):
    tool: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_now)


class Project(SQLModel, table=True):
    """A code folder the user registered in the UI. The agent can only read inside these."""

    name: str = Field(primary_key=True)
    path: str = Field(unique=True)  # absolute, symlinks resolved
    # "local" today; "git" later (cloned into a managed folder), with the same Workspace rules.
    source: str = "local"
    created_at: datetime = Field(default_factory=_now)


class ProjectReadGrant(SQLModel, table=True):
    """ "Always allow reading this project": read-only code tools run without asking, for this project only."""

    project: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_now)


class AuditEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_now, index=True)
    session_id: str
    tool: str
    args: str  # JSON
    risk: str
    # approve | always_allow | auto | reject | expired | invalid_args | unknown_tool | precheck_failed | ungrounded | repeat_refused
    decision: str
    # ok | error | not_run (detail says why, e.g. "stopped by the user")
    outcome: str
    detail: str = ""


class ModelUsage(SQLModel, table=True):
    """Requests and tokens used from one usage pool (a provider limit) on one provider day."""

    pool: str = Field(primary_key=True)
    day: str = Field(primary_key=True)  # YYYY-MM-DD in the pool's reset time zone
    requests: int = 0
    tokens: int = 0
    # Part of `tokens` that was estimated because the provider didn't report usage.
    estimated_tokens: int = 0
    # When the provider's own "remaining" figure last corrected `requests` (None = counted here only).
    synced_at: datetime | None = None
    # The provider refused requests for the rest of its day (a daily 429).
    blocked: bool = False


class ImageVersion(SQLModel, table=True):
    """One successful build of a project (Step 5): what was built, and whether it ever ran healthy."""

    project: str = Field(primary_key=True)
    number: int = Field(primary_key=True)  # v1, v2, … per project
    tag: str  # e.g. "sample-project:v4-e54f34f-dirty"
    image_id: str
    commit: str | None = None
    dirty: bool = False
    size_bytes: int | None = None
    built_at: datetime = Field(default_factory=_now)
    # None = never run; True = passed a health check at least once; False = only ever failed.
    healthy: bool | None = None
    last_run_at: datetime | None = None


class Store:
    def __init__(self, url: str) -> None:
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if url == "sqlite://":  # in-memory (tests): share one connection
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        SQLModel.metadata.create_all(self.engine)

    def is_always_allowed(self, tool: str) -> bool:
        with Session(self.engine) as s:
            return s.get(AlwaysAllow, tool) is not None

    def allow_always(self, tool: str) -> None:
        with Session(self.engine) as s:
            if s.get(AlwaysAllow, tool) is None:
                s.add(AlwaysAllow(tool=tool))
                s.commit()

    def revoke(self, tool: str) -> bool:
        with Session(self.engine) as s:
            row = s.get(AlwaysAllow, tool)
            if row is None:
                return False
            s.delete(row)
            s.commit()
            return True

    def always_allowed(self) -> list[AlwaysAllow]:
        with Session(self.engine) as s:
            return list(s.exec(select(AlwaysAllow).order_by(AlwaysAllow.tool)))

    # ---- projects -----------------------------------------------------------

    def add_project(self, name: str, path: str) -> Project:
        with Session(self.engine) as s:
            project = Project(name=name, path=path)
            s.add(project)
            s.commit()
            s.refresh(project)
            return project

    def get_project(self, name: str) -> Project | None:
        with Session(self.engine) as s:
            return s.get(Project, name)

    def projects(self) -> list[Project]:
        with Session(self.engine) as s:
            return list(s.exec(select(Project).order_by(Project.name)))

    def project_by_path(self, path: str) -> Project | None:
        with Session(self.engine) as s:
            return s.exec(select(Project).where(Project.path == path)).first()

    def remove_project(self, name: str) -> bool:
        with Session(self.engine) as s:
            project = s.get(Project, name)
            if project is None:
                return False
            grant = s.get(ProjectReadGrant, name)
            if grant is not None:
                s.delete(grant)
            s.delete(project)
            s.commit()
            return True

    def project_reads_allowed(self, name: str) -> bool:
        with Session(self.engine) as s:
            return s.get(ProjectReadGrant, name) is not None

    def allow_project_reads(self, name: str) -> None:
        with Session(self.engine) as s:
            if s.get(ProjectReadGrant, name) is None:
                s.add(ProjectReadGrant(project=name))
                s.commit()

    def revoke_project_reads(self, name: str) -> bool:
        with Session(self.engine) as s:
            grant = s.get(ProjectReadGrant, name)
            if grant is None:
                return False
            s.delete(grant)
            s.commit()
            return True

    # ---- model usage (see quota.py) ------------------------------------------

    def usage(self, pool: str, day: str) -> ModelUsage:
        with Session(self.engine) as s:
            return s.get(ModelUsage, (pool, day)) or ModelUsage(pool=pool, day=day)

    def add_usage(self, pool: str, day: str, *, requests: int, tokens: int, estimated_tokens: int = 0) -> None:
        with Session(self.engine) as s:
            row = s.get(ModelUsage, (pool, day)) or ModelUsage(pool=pool, day=day)
            row.requests += requests
            row.tokens += tokens
            row.estimated_tokens += estimated_tokens
            s.add(row)
            s.commit()

    def sync_requests(self, pool: str, day: str, requests: int) -> None:
        """The provider said how many requests it has counted today; that number wins."""
        with Session(self.engine) as s:
            row = s.get(ModelUsage, (pool, day)) or ModelUsage(pool=pool, day=day)
            row.requests = requests
            row.synced_at = _now()
            s.add(row)
            s.commit()

    def block_for_day(self, pool: str, day: str) -> None:
        with Session(self.engine) as s:
            row = s.get(ModelUsage, (pool, day)) or ModelUsage(pool=pool, day=day)
            row.blocked = True
            s.add(row)
            s.commit()

    # ---- image versions (Step 5) -----------------------------------------------

    def next_version(self, project: str) -> int:
        with Session(self.engine) as s:
            last = s.exec(select(ImageVersion.number).where(ImageVersion.project == project)
                          .order_by(ImageVersion.number.desc())).first()
            return (last or 0) + 1

    def add_version(self, version: ImageVersion) -> None:
        with Session(self.engine) as s:
            s.merge(version)
            s.commit()

    def versions(self, project: str) -> list[ImageVersion]:
        """Newest first."""
        with Session(self.engine) as s:
            return list(s.exec(select(ImageVersion).where(ImageVersion.project == project)
                               .order_by(ImageVersion.number.desc())))

    def record_run(self, project: str, number: int, healthy: bool) -> None:
        """A health check result. Once a version has passed, a later failure doesn't erase that."""
        with Session(self.engine) as s:
            row = s.get(ImageVersion, (project, number))
            if row is None:
                return
            row.healthy = True if healthy or row.healthy else False
            row.last_run_at = _now()
            s.add(row)
            s.commit()

    # ---- audit ---------------------------------------------------------------

    def audit(
        self,
        *,
        session_id: str,
        tool: str,
        args: dict,
        risk: str,
        decision: str,
        outcome: str,
        detail: str = "",
    ) -> None:
        entry = AuditEntry(
            session_id=session_id,
            tool=tool,
            args=json.dumps(args, default=str),
            risk=risk,
            decision=decision,
            outcome=outcome,
            detail=detail[:2000],
        )
        with Session(self.engine) as s:
            s.add(entry)
            s.commit()

    def recent_audit(self, limit: int = 50) -> list[AuditEntry]:
        with Session(self.engine) as s:
            stmt = select(AuditEntry).order_by(AuditEntry.id.desc()).limit(limit)
            return list(s.exec(stmt))
