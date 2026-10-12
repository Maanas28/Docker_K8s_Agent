"""Sandboxed access to one registered project folder.

Every file the agent touches goes through a Workspace. It guarantees that:
- paths stay inside the project root (no "../", absolute paths or symlinks that lead outside),
- secret files (.env, private keys, credentials) are never read; only their existence and,
  for env files, their variable names are reported,
- dependency and build folders are skipped when listing or searching,
- everything returned has secret-looking values masked,
- writes are limited to a few Docker files at the project root, happen atomically, and are refused
  if the file changed after the user reviewed the diff.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from docker.utils.build import exclude_paths

from app.redact import redact
from app.tools.registry import ToolError

# Folders that are never listed or searched (still readable by exact path, except .git).
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".idea", ".vscode", "dist", "build", "target", ".gradle", ".next",
    ".nuxt", ".cache", "coverage", ".terraform",
}
# Never read, not even by exact path.
BLOCKED_DIRS = {".git"}

_SECRET_NAMES = [
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.kdbx",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "credentials*.json", "service-account*.json", "*secret*.json",
    ".npmrc", ".pypirc", ".netrc", ".htpasswd", ".git-credentials",
]
# .env files are secret, except templates that are meant to be committed.
_ENV_TEMPLATES = (".example", ".sample", ".template", ".dist")

MAX_READ_BYTES = 200_000
MAX_LINES = 400

# Generated lock files: huge, machine-written, and useless for understanding a project. Reading one
# would fill the model's context window, so the agent gets a short note pointing at the manifest.
LOCK_FILES = {
    "uv.lock": "pyproject.toml", "poetry.lock": "pyproject.toml", "pdm.lock": "pyproject.toml",
    "Pipfile.lock": "Pipfile", "package-lock.json": "package.json", "yarn.lock": "package.json",
    "pnpm-lock.yaml": "package.json", "bun.lockb": "package.json", "Cargo.lock": "Cargo.toml",
    "go.sum": "go.mod", "Gemfile.lock": "Gemfile", "composer.lock": "composer.json",
}

# Files that explain how a project is built, configured and started, in the order they're picked
# for project_context. Patterns are matched against project-relative paths.
KEY_FILE_PATTERNS = (
    # dependency manifests
    "pyproject.toml", "requirements.txt", "requirements/*.txt", "setup.py", "setup.cfg", "Pipfile",
    "package.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json",
    # existing container files
    "Dockerfile", ".dockerignore", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    # entry points
    "*/__main__.py", "__main__.py", "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py",
    "*/main.py", "*/app.py", "*/wsgi.py", "*/asgi.py",
    "index.js", "server.js", "src/index.js", "src/index.ts", "src/main.ts", "src/server.ts",
    "main.go", "cmd/*/main.go",
    # configuration (templates only; real .env files are secret)
    "*/config.py", "config.py", "*/settings.py", "settings.py", ".env.example", ".env.sample",
)

# The only files the agent may write, and only at the project root (for now).
WRITABLE_FILES = ("Dockerfile", ".dockerignore")
MAX_WRITE_BYTES = 20_000


def content_hash(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def is_secret_file(name: str) -> bool:
    lower = name.lower()
    if lower == ".env" or (lower.startswith(".env.") and not lower.endswith(_ENV_TEMPLATES)):
        return True
    if lower.endswith(".env") and not lower.endswith(_ENV_TEMPLATES):  # e.g. prod.env
        return True
    return any(fnmatch.fnmatch(lower, pattern) for pattern in _SECRET_NAMES)


def env_var_names(path: Path) -> list[str]:
    """Variable names in an env file, without their values."""
    names = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name = line.split("=", 1)[0].removeprefix("export ").strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                    names.append(name)
    except OSError:
        pass
    return names


def env_values(path: Path) -> dict[str, str]:
    """Variables of an env file WITH their values. Only for handing to Docker when running a
    container; never return these to the model or the UI."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # KEY="value" or KEY='value'
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()  # KEY=value  # comment
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
            values[name] = value
    return values


@dataclass
class SecretFile:
    path: str
    variables: list[str] | None  # names only, for env files


class Workspace:
    def __init__(self, name: str, root: str | Path) -> None:
        self.name = name
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"The folder for project '{name}' no longer exists: {self.root}")

    # ---- paths ---------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """Turn a project-relative path into an absolute one, refusing anything outside the root."""
        rel = (rel or ".").strip()
        if rel.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", rel):
            raise ToolError(f"Use a path relative to the project, not '{rel}'.")
        target = (self.root / rel).resolve()  # follows symlinks, so a link to /etc is caught here
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"'{rel}' is outside the project '{self.name}'.")
        if any(part in BLOCKED_DIRS for part in target.relative_to(self.root).parts):
            raise ToolError(f"'{rel}' is inside a folder I don't read.")
        return target

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    # ---- reading -------------------------------------------------------------

    def read(self, rel: str, start_line: int = 1, max_lines: int = MAX_LINES) -> dict:
        path = self.resolve(rel)
        if not path.exists():
            raise ToolError(f"There's no '{rel}' in project '{self.name}'.")
        if path.is_dir():
            raise ToolError(f"'{rel}' is a folder. List it instead of reading it.")
        if path.name in LOCK_FILES:
            return self.describe_lock_file(path)
        if is_secret_file(path.name):
            secret = self.describe_secret(path)
            facts = {"File": secret.path, "Contents": "hidden, this is a secret file"}
            if secret.variables is not None:
                facts["Variables"] = ", ".join(secret.variables) or "(none)"
            return {"kind": "facts", "title": f"{secret.path} (secret file, values hidden)", "facts": facts}
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"'{rel}' is too large to read ({path.stat().st_size // 1000} KB). Search it instead.")
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            raise ToolError(f"'{rel}' is a binary file.")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        end = min(len(lines), start + max_lines - 1)
        text, masked = redact("\n".join(lines[start - 1 : end]), self.rel(path))
        return {
            "kind": "file",
            "title": self.rel(path),
            "text": text,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "truncated": end < len(lines),
            "redacted": masked,
        }

    def describe_lock_file(self, path: Path) -> dict:
        manifest = LOCK_FILES[path.name]
        return {
            "kind": "facts",
            "title": f"{self.rel(path)} (lock file, not read)",
            "facts": {
                "File": self.rel(path),
                "Size": f"{path.stat().st_size // 1000} KB",
                "Note": f"Generated file with exact package versions. Read {manifest} for the dependencies.",
            },
        }

    def key_files(self, budget: int = 24_000, per_file: int = 6_000) -> tuple[list[dict], list[dict]]:
        """The files that explain how the project is built and run, within a character budget.

        Returns (included, skipped). Each included file is {path, text, lines, truncated, redacted};
        each skipped one is {path, reason}. Contents are masked exactly like read().
        """
        candidates: list[Path] = []
        for pattern in KEY_FILE_PATTERNS:
            matches = sorted(self.root.glob(pattern))
            candidates += [m for m in matches if m.is_file() and m not in candidates]
        included, skipped, used = [], [], 0
        for path in candidates:
            rel = self.rel(path)
            if not self._inside(path) or any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
                continue
            if is_secret_file(path.name):
                continue  # secret files are listed by project_overview, never read
            if used >= budget:
                skipped.append({"path": rel, "reason": "context budget used up; read it if needed"})
                continue
            try:
                result = self.read(rel)
            except ToolError as e:
                skipped.append({"path": rel, "reason": str(e)})
                continue
            text = result["text"]
            room = min(per_file, budget - used)
            truncated = result["truncated"] or len(text) > room
            text = text[:room]
            used += len(text)
            included.append({
                "path": rel, "text": text, "lines": result["total_lines"],
                "truncated": truncated, "redacted": result["redacted"],
            })
        skipped += [
            {"path": self.rel(p), "reason": f"lock file ({p.stat().st_size // 1000} KB), not needed"}
            for p in sorted(self.root.iterdir()) if p.name in LOCK_FILES
        ]
        return included, skipped

    # ---- writing ---------------------------------------------------------------

    def plan_write(self, rel: str, content: str) -> dict:
        """Everything the approval card needs: old vs new, a diff, and a hash of the current file."""
        if rel not in WRITABLE_FILES:
            raise ToolError(f"I can only write {' or '.join(WRITABLE_FILES)} at the project root, not '{rel}'.")
        path = self.resolve(rel)
        if path.is_symlink() or path.is_dir():
            raise ToolError(f"'{rel}' in project '{self.name}' is a link or a folder, so I won't overwrite it.")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"That {rel} is too large ({len(data) // 1000} KB); keep it under {MAX_WRITE_BYTES // 1000} KB.")
        if not content.strip():
            raise ToolError(f"The {rel} content is empty.")
        new = content if content.endswith("\n") else content + "\n"
        old_bytes = path.read_bytes() if path.exists() else None
        old = old_bytes.decode("utf-8", errors="replace") if old_bytes is not None else None
        diff = "".join(difflib.unified_diff(
            (old or "").splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{rel}" if old is not None else "/dev/null", tofile=f"b/{rel}",
        ))
        return {
            "path": rel,
            "new": old is None,
            "unchanged": old == new,
            "content": new,
            "old_lines": len(old.splitlines()) if old is not None else 0,
            "lines": len(new.splitlines()),
            "diff": diff,
            "hash": content_hash(old_bytes),  # what the user reviewed; checked again before writing
        }

    def write(self, rel: str, content: str, expected_hash: str | None) -> Path:
        """Write atomically, but only if the file is still exactly what the user reviewed."""
        if rel not in WRITABLE_FILES:
            raise ToolError(f"I can only write {' or '.join(WRITABLE_FILES)}.")
        path = self.resolve(rel)
        current = content_hash(path.read_bytes()) if path.exists() else None
        if current != expected_hash:
            raise ToolError(
                f"{rel} changed after you reviewed the diff, so I didn't overwrite it. "
                "Ask again to see the new diff."
            )
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{rel}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            if path.exists():
                os.chmod(tmp, path.stat().st_mode & 0o777)
            else:
                os.chmod(tmp, 0o644)
            os.replace(tmp, path)  # atomic: the file is either old or new, never half-written
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path

    def top_level_names(self) -> set[str]:
        return {p.name for p in self.root.iterdir()}

    def read_raw(self, rel: str) -> str | None:
        """Unmasked text of a non-secret file, for internal checks only (never sent to the LLM)."""
        path = self.resolve(rel)
        if not path.is_file() or is_secret_file(path.name):
            return None
        return path.read_text(errors="replace")

    def build_context(self) -> dict:
        """What `docker build` would send from this folder: files left after .dockerignore.

        Uses docker-py's own .dockerignore matcher (`**`, `!` exceptions…) rather than our own,
        because the secret check below must agree with what Docker really sends.
        """
        ignore = self.read_raw(".dockerignore")
        patterns = [
            line.strip() for line in (ignore or "").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        included = exclude_paths(str(self.root), list(patterns))  # it appends to the list it gets
        files, size, secrets, dep_dirs = 0, 0, [], set()
        for rel in included:
            path = self.root / rel
            if not path.is_file() or path.is_symlink():
                continue
            files += 1
            size += path.stat().st_size
            parts = PurePosixPath(rel).parts
            skipped = next((p for p in parts[:-1] if p in SKIP_DIRS), None)
            if skipped:
                # e.g. .venv/…/certifi/cacert.pem: a package's file, not the project's secret.
                # Sending the folder at all is reported separately.
                dep_dirs.add(skipped)
            elif is_secret_file(path.name):
                secrets.append(PurePosixPath(rel).as_posix())
        return {
            "files": files, "bytes": size, "secrets": sorted(secrets),
            "dependency_dirs": sorted(dep_dirs), "has_dockerignore": ignore is not None,
        }

    def describe_secret(self, path: Path) -> SecretFile:
        is_env = path.name.lower().startswith(".env") or path.name.lower().endswith(".env")
        return SecretFile(self.rel(path), env_var_names(path) if is_env else None)

    # ---- listing -------------------------------------------------------------

    def walk(self, rel: str = ".", max_depth: int = 3, max_entries: int = 300):
        """Yield (path, is_dir, depth) in sorted order, skipping dependency/build folders."""
        start = self.resolve(rel)
        if not start.is_dir():
            raise ToolError(f"'{rel}' isn't a folder in project '{self.name}'.")
        count = 0

        def visit(folder: Path, depth: int):
            nonlocal count
            try:
                entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for entry in entries:
                if count >= max_entries:
                    return
                if entry.is_symlink() and not self._inside(entry):
                    continue  # links that lead outside the project are ignored
                is_dir = entry.is_dir()
                if is_dir and entry.name in SKIP_DIRS:
                    continue
                count += 1
                yield entry, is_dir, depth
                if is_dir and depth < max_depth:
                    yield from visit(entry, depth + 1)

        yield from visit(start, 1)

    def _inside(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return resolved == self.root or self.root in resolved.parents

    def tree(self, rel: str = ".", max_depth: int = 3, max_entries: int = 300) -> tuple[str, bool]:
        lines, count = [], 0
        for entry, is_dir, depth in self.walk(rel, max_depth, max_entries):
            count += 1
            marker = "/" if is_dir else (" (secret, not read)" if is_secret_file(entry.name) else "")
            lines.append(f"{'  ' * (depth - 1)}{entry.name}{marker}")
        return "\n".join(lines) or "(empty)", count >= max_entries

    def secret_files(self) -> list[SecretFile]:
        return [
            self.describe_secret(entry)
            for entry, is_dir, _ in self.walk(".", max_depth=4, max_entries=2000)
            if not is_dir and is_secret_file(entry.name)
        ]

    # ---- searching -----------------------------------------------------------

    def search(self, pattern: str, glob: str | None = None, max_results: int = 50,
               ignore_case: bool = True) -> tuple[list[dict], bool]:
        flags = re.I if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)
        results: list[dict] = []
        for entry, is_dir, _ in self.walk(".", max_depth=12, max_entries=5000):
            if is_dir or is_secret_file(entry.name) or entry.name in LOCK_FILES:
                continue
            rel = self.rel(entry)
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(entry.name, glob)):
                continue
            try:
                if entry.stat().st_size > MAX_READ_BYTES:
                    continue
                data = entry.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8000]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    masked, _ = redact(line.strip()[:300], rel)
                    results.append({"path": rel, "line": number, "text": masked})
                    if len(results) >= max_results:
                        return results, True
        return results, False

    # ---- git -----------------------------------------------------------------

    def git_file_status(self, rel: str) -> str | None:
        """'untracked', 'modified', 'clean' or None (not a git repo)."""
        if not (self.root / ".git").exists():
            return None
        try:
            out = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
                 "status", "--porcelain", "--", rel],
                cwd=self.root, capture_output=True, text=True, timeout=5,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        line = out.stdout.strip()
        if not line:
            return "clean" if (self.root / rel).exists() else "untracked"
        return "untracked" if line.startswith("??") else "modified"

    def git_info(self) -> dict[str, str] | None:
        """Branch, commit and whether there are uncommitted changes. Read-only."""
        if not (self.root / ".git").exists():
            return None

        def git(*args: str) -> str:
            # fsmonitor and hooks are disabled so a repo's own config can't make `git` run code.
            out = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", *args],
                cwd=self.root, capture_output=True, text=True, timeout=5,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            )
            return out.stdout.strip() if out.returncode == 0 else ""

        try:
            branch = git("rev-parse", "--abbrev-ref", "HEAD")
            commit = git("rev-parse", "--short", "HEAD")
            changes = git("status", "--porcelain")
        except (OSError, subprocess.TimeoutExpired):
            return None
        return {
            "branch": branch or "(unknown)",
            "commit": commit or "(no commits yet)",
            "uncommitted changes": f"{len(changes.splitlines())} files" if changes else "none",
        }


def safe_project_path(raw: str) -> Path:
    """Validate a folder the user wants to register as a project."""
    path = Path(os.path.expanduser(raw.strip())).resolve()
    if not path.is_dir():
        raise ValueError(f"'{raw}' isn't an existing folder.")
    home = Path.home().resolve()
    if path == Path(path.anchor) or path == home or path in home.parents:
        raise ValueError("Register a specific project folder, not your home folder or the disk root.")
    if home not in path.parents and len(path.parts) > 1:
        top = path.parts[1]
        # Never anything under these; for the rest only the top folders themselves are refused,
        # so e.g. /opt/my-project is fine but /opt, /private/etc or /private/var are not.
        if top in {"etc", "System", "usr", "bin", "sbin", "Library", "Applications", "cores", "dev"}:
            raise ValueError(f"'{path}' is a system folder and can't be a project.")
        if len(path.parts) == 2 or (top == "private" and len(path.parts) == 3):
            raise ValueError(f"'{path}' is too broad. Register a specific project folder inside it.")
    if path in {home / "Library", home / ".ssh", home / "Documents", home / "Desktop", home / "Downloads"}:
        raise ValueError(f"'{path}' is too broad or sensitive. Register a specific project folder inside it.")
    if any(part in {".ssh", ".gnupg", ".aws", ".kube", ".docker"} for part in PurePosixPath(path.as_posix()).parts):
        raise ValueError(f"'{path}' is a credentials folder and can't be a project.")
    return path
