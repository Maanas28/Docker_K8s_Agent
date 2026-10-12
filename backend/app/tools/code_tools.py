"""Tools for the user's registered code projects.

Every tool takes a `project` name, which must be one the user registered in the UI, and goes
through a Workspace, so it can't leave that folder or open secret files. Reading is read-only;
the one write tool can only create or replace a Dockerfile or .dockerignore at the project root,
after rule checks and the user's approval of the exact diff.
"""

from __future__ import annotations

import json
import re
import tomllib
from typing import Literal

from pydantic import Field

from app.dockerfile_check import check_dockerfile, check_dockerignore
from app.store import Store
from app.tools.registry import Risk, ToolArgs, ToolError, registry
from app.workspace import MAX_WRITE_BYTES, SKIP_DIRS, Workspace

_store: Store | None = None
_build_facts_on = True  # agent.toml [context] build_facts


def configure(store: Store, build_facts: bool = True) -> None:
    """Give the tools access to the registered projects (called once at startup)."""
    global _store, _build_facts_on
    _store = store
    _build_facts_on = build_facts


def get_store() -> Store:
    if _store is None:
        raise ToolError("Projects aren't set up.")
    return _store


def _workspace(name: str) -> Workspace:
    if _store is None:
        raise ToolError("Projects aren't set up.")
    project = _store.get_project(name)
    if project is None:
        known = ", ".join(p.name for p in _store.projects())
        hint = f" Registered projects: {known}." if known else " No projects are registered yet; add one under Projects."
        raise ToolError(f"There's no registered project called '{name}'.{hint}")
    return Workspace(project.name, project.path)


# Files that identify a stack, and what they mean.
_MANIFESTS = {
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "setup.py": "Python",
    "Pipfile": "Python (Pipenv)",
    "uv.lock": "Python (uv)",
    "poetry.lock": "Python (Poetry)",
    "package.json": "Node.js",
    "package-lock.json": "Node.js (npm)",
    "yarn.lock": "Node.js (Yarn)",
    "pnpm-lock.yaml": "Node.js (pnpm)",
    "bun.lockb": "Bun",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
    "build.gradle.kts": "Kotlin/Java (Gradle)",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "mix.exs": "Elixir",
}
_DOCKER_FILES = ("Dockerfile", ".dockerignore", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")


class ProjectArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")


class ListFilesArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    path: str = Field(".", description="Folder inside the project, relative to its root")
    depth: int = Field(2, ge=1, le=6, description="How many folder levels to show")


class ReadFileArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    path: str = Field(description="File path relative to the project root, e.g. app/main.py")
    start_line: int = Field(1, ge=1, description="First line to read")
    max_lines: int = Field(200, ge=1, le=400, description="How many lines to read")


class SearchArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    pattern: str = Field(min_length=1, max_length=200, description="Text or regular expression to find")
    glob: str | None = Field(None, description='Only search matching files, e.g. "*.py" or "src/*"')
    max_results: int = Field(50, ge=1, le=200)


class NoArgs(ToolArgs):
    pass


@registry.tool(risk=Risk.READ, summary=lambda a: "List registered projects", platform="project")
def project_list(args: NoArgs) -> dict:
    """List the code projects the user has registered, with their folders."""
    if _store is None:
        raise ToolError("Projects aren't set up.")
    rows = [{"name": p.name, "path": p.path, "source": p.source} for p in _store.projects()]
    return {"kind": "table", "columns": ["name", "path", "source"], "rows": rows}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Get an overview of project '{a.project}'",
    platform="project",
    scope_field="project",
)
def project_overview(args: ProjectArgs) -> dict:
    """Overview of a project: detected languages and tools, Docker files present, secret files (names only), git branch/commit, and the folder tree. No file contents; use project_context to also get the key files."""
    ws = _workspace(args.project)
    facts, tree = _overview(ws)
    return {"kind": "overview", "title": f"{ws.name} overview", "facts": facts, "tree": tree}


def _overview(ws: Workspace) -> tuple[dict[str, str], str]:
    """Facts and folder tree shared by project_overview and project_context."""
    top = ws.top_level_names()
    stack = sorted({label for name, label in _MANIFESTS.items() if name in top})
    docker = [name for name in _DOCKER_FILES if name in top]
    secrets = ws.secret_files()
    tree, truncated = ws.tree(".", max_depth=3, max_entries=200)
    facts = {
        "Project": ws.name,
        "Folder": str(ws.root),
        "Detected": ", ".join(stack) or "nothing recognised from the top-level files",
        "Docker files": ", ".join(docker) or "none",
    }
    if secrets:
        facts["Secret files (not read)"] = "; ".join(
            f"{s.path} ({', '.join(s.variables)})" if s.variables else s.path for s in secrets
        )
    git = ws.git_info()
    facts.update({f"Git {k}": v for k, v in git.items()} if git else {"Git": "not a git repository"})
    return facts, tree + ("\n… (more files not shown)" if truncated else "")


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Understand project '{a.project}' (overview + key files)",
    platform="project",
    scope_field="project",
)
def project_context(args: ProjectArgs) -> dict:
    """Everything needed to understand how a project is built and run, in one call: the overview (stack, Docker files, secret file names, git, folder tree) PLUS the contents of its key files (dependency manifest, entry point, config, existing Dockerfile/.dockerignore). Lock files are skipped. Use this first; read other files only if something is missing."""
    ws = _workspace(args.project)
    facts, tree = _overview(ws)
    if _build_facts_on:
        facts.update(build_facts(ws))
    files, skipped = ws.key_files()
    return {
        "kind": "context",
        "title": f"{ws.name}: overview and {len(files)} key files",
        "facts": facts,
        "tree": tree,
        "files": files,
        "skipped": skipped,
    }


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"List files in '{a.project}'" + ("" if a.path in (".", "") else f" / {a.path}"),
    platform="project",
    scope_field="project",
)
def project_list_files(args: ListFilesArgs) -> dict:
    """List files and folders inside a project (dependency and build folders are skipped)."""
    ws = _workspace(args.project)
    tree, truncated = ws.tree(args.path, max_depth=args.depth)
    title = ws.name if args.path in (".", "") else f"{ws.name}/{args.path.strip('/')}"
    return {"kind": "tree", "title": title, "text": tree + ("\n… (more not shown)" if truncated else "")}


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Read '{a.path}' in '{a.project}'",
    platform="project",
    scope_field="project",
)
def project_read_file(args: ReadFileArgs) -> dict:
    """Read a text file from a project (up to 400 lines at a time; use start_line for more). Secret files like .env only show variable names."""
    return _workspace(args.project).read(args.path, args.start_line, args.max_lines)


@registry.tool(
    risk=Risk.READ,
    summary=lambda a: f"Search '{a.project}' for “{a.pattern}”" + (f" in {a.glob}" if a.glob else ""),
    platform="project",
    scope_field="project",
)
def project_search(args: SearchArgs) -> dict:
    """Search a project's files for text or a regular expression. Returns matching lines with file paths and line numbers."""
    rows, truncated = _workspace(args.project).search(args.pattern, args.glob, args.max_results)
    return {
        "kind": "table",
        "columns": ["path", "line", "text"],
        "rows": rows,
        **({"note": f"Showing the first {len(rows)} matches."} if truncated else {}),
    }


# ---- build facts (agent.toml [context] build_facts) --------------------------------------------
# Facts detected by code, so the model doesn't have to guess them (an 8B model often wrote
# `pip install -r requirements.txt` for uv projects, or the wrong start command). They are
# stated as facts with their source; the model still decides how to write the Dockerfile.

# (lock file or manifest, package manager), most specific first.
_PACKAGE_MANAGERS = (
    ("uv.lock", "uv"), ("poetry.lock", "Poetry"), ("pdm.lock", "PDM"), ("Pipfile.lock", "Pipenv"),
    ("Pipfile", "Pipenv"), ("requirements.txt", "pip"),
    ("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "Yarn"), ("bun.lockb", "Bun"), ("package-lock.json", "npm"),
    ("package.json", "npm (no lock file)"), ("go.mod", "Go modules"), ("Cargo.toml", "Cargo"),
    ("pom.xml", "Maven"), ("build.gradle", "Gradle"), ("build.gradle.kts", "Gradle"),
    ("Gemfile", "Bundler"), ("composer.json", "Composer"), ("pyproject.toml", "pip (pyproject.toml only)"),
)
# A port number next to the word "port", or passed to listen(…) / --port.
_PORT = re.compile(r"(?i)(?:\bport\w*['\"]?\s*[:=,]\s*[^\n]{0,30}?|listen\(\s*|--port[ =])\b(\d{2,5})\b")
_NOT_SOURCE = ("Dockerfile", ".dockerignore", "compose", "test", ".md", ".lock", ".example", ".sample")


def build_facts(ws: Workspace) -> dict[str, str]:
    top = ws.top_level_names()
    facts: dict[str, str] = {}
    found = next(((name, pm) for name, pm in _PACKAGE_MANAGERS if name in top), None)
    if found:
        facts["Package manager (detected)"] = f"{found[1]} (from {found[0]})"
    if _python_like(top):
        present = [n for n in ("pyproject.toml", "uv.lock", "poetry.lock", "requirements.txt", "Pipfile", "setup.py") if n in top]
        facts["Dependency files"] = ", ".join(present) + ("" if "requirements.txt" in top else " (there is NO requirements.txt)")
    start = _start_commands(ws, top)
    if start:
        facts["Start command (detected)"] = " or ".join(start)
    ports = _port_candidates(ws)
    if ports:
        facts["Port (detected, check)"] = "; ".join(ports)
    return facts


def _start_commands(ws: Workspace, top: set[str]) -> list[str]:
    out: list[str] = []
    pyproject = ws.read_raw("pyproject.toml") if "pyproject.toml" in top else None
    if pyproject:
        try:
            scripts = tomllib.loads(pyproject).get("project", {}).get("scripts", {})
        except tomllib.TOMLDecodeError:
            scripts = {}
        out += [f"`{name}` (script in pyproject.toml)" for name in list(scripts)[:2]]
    for entry in sorted(ws.root.iterdir()):
        if entry.is_dir() and entry.name not in SKIP_DIRS and (entry / "__main__.py").is_file():
            out.append(f"`python -m {entry.name}` (from {entry.name}/__main__.py)")
    if "manage.py" in top:
        out.append("`python manage.py runserver` (Django, from manage.py)")
    out += [f"`python {name}` (from {name})" for name in ("main.py", "app.py") if name in top and _python_like(top)]
    package_json = ws.read_raw("package.json") if "package.json" in top else None
    if package_json:
        try:
            pkg = json.loads(package_json)
        except ValueError:
            pkg = {}
        if isinstance(pkg, dict) and isinstance(pkg.get("scripts"), dict) and pkg["scripts"].get("start"):
            out.append(f"`npm start` (runs `{pkg['scripts']['start']}`, from package.json)")
        elif isinstance(pkg, dict) and pkg.get("main"):
            out.append(f"`node {pkg['main']}` (main in package.json)")
    return out[:3]


def _python_like(top: set[str]) -> bool:
    return bool({"pyproject.toml", "requirements.txt", "setup.py", "Pipfile"} & top)


def _port_candidates(ws: Workspace) -> list[str]:
    rows, _ = ws.search(_PORT.pattern, max_results=40)
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if any(marker in row["path"] for marker in _NOT_SOURCE):
            continue  # an old Dockerfile or a test isn't evidence of the app's port
        match = _PORT.search(row["text"])
        if not match or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        out.append(f"{match.group(1)} ({row['path']}:{row['line']}: {row['text'][:80]})")
    return out[:3]


# ---- writing (Step 2) ---------------------------------------------------------------------------


class WriteFileArgs(ToolArgs):
    project: str = Field(description="Name of a registered project")
    path: Literal["Dockerfile", ".dockerignore"] = Field(description="Which file to write, at the project root")
    content: str = Field(
        min_length=1, max_length=MAX_WRITE_BYTES,
        description="The COMPLETE new file content (not a snippet or a diff)",
    )
    reason: str = Field(
        min_length=5, max_length=400,
        description="One or two sentences for the user: why you are writing this file. If it fixes an "
                    "error (e.g. a failed build), say which error and what you change to fix it.",
    )


_GIT_NOTE = {
    "untracked": "new file, not committed yet",
    "clean": "committed; this becomes an uncommitted change",
    "modified": "already has uncommitted changes",
}


# A literal backslash-n followed by indentation or a command: a line break that was escaped twice
# when the model wrote the tool call, e.g. `apt-get update && \\n    apt-get install …` on one line.
_ESCAPED_NEWLINE = re.compile(r"\\n(?=\s{2,}\S|[A-Z]{3,} )")


# Tool-call or reasoning markup inside a file's content: the provider didn't split the model's call
# into separate arguments (seen with Qwen's XML-style calls), so the rest of the call ended up in
# the file. Matches e.g. `</parameter`, `<parameter=`, `<tool_call>`, `<function=`, `</think>`,
# and special tokens like `<|im_end|>`.
_CALL_MARKUP = re.compile(
    r"</?(?:parameter|tool_call|tool_response|function|think|invoke|arg_key|arg_value)\b|<\|[a-z_]+\|>", re.I
)


def _preview_write(args: WriteFileArgs) -> dict:
    ws = _workspace(args.project)
    leaked = next(((n, line) for n, line in enumerate(args.content.splitlines(), 1) if _CALL_MARKUP.search(line)), None)
    if leaked:
        raise ToolError(
            f"Line {leaked[0]} of the {args.path} contains tool-call markup (`{leaked[1].strip()[:60]}`), so the "
            "call's arguments weren't separated properly and the rest of the call ended up in the file. "
            "Send `content` with only the file itself, and the explanation in the separate `reason` argument."
        )
    bad = [n for n, line in enumerate(args.content.splitlines(), 1) if _ESCAPED_NEWLINE.search(line)]
    if bad:
        raise ToolError(
            f"Line {bad[0]} of the {args.path} contains a literal '\\n' (a backslash and an n) where a "
            "line break was meant. Write real line breaks in the content instead."
        )
    plan = ws.plan_write(args.path, args.content)
    if plan["unchanged"]:
        raise ToolError(f"{args.path} already has exactly this content, so there's nothing to write.")

    top = ws.top_level_names()
    if args.path == "Dockerfile":
        check = check_dockerfile(plan["content"], ws.read_raw(".dockerignore"), top)
    else:
        check = check_dockerignore(plan["content"], top)
    if check.errors:
        problems = "\n".join(f"- {e}" for e in check.errors)
        raise ToolError(
            f"The {args.path} has problems, so I haven't shown it to you yet:\n{problems}\n"
            f"The content must be only the {args.path} itself; write each file in its own call."
        )

    facts = {
        "Project": ws.name,
        "File": args.path,
        "Action": "create new file" if plan["new"] else f"replace existing file ({plan['old_lines']} → {plan['lines']} lines)",
        **check.facts,
    }
    warnings = list(check.warnings)
    if not plan["new"]:
        warnings.insert(0, f"This replaces the existing {args.path}. Check the diff for anything that would be lost.")
    git = ws.git_info()
    if git:
        status = ws.git_file_status(args.path)
        facts["Git"] = f"{git['branch']} · {_GIT_NOTE.get(status or '', status or '?')}"
        if status == "modified":
            warnings.append(f"You have uncommitted edits to {args.path}; they'll be replaced.")
    return {
        "reason": args.reason,  # shown above the diff, so the user knows why before approving
        "facts": facts,
        "warnings": warnings,
        "file": {k: plan[k] for k in ("path", "new", "content", "diff", "lines", "hash")},
    }


@registry.tool(
    risk=Risk.CHANGE,
    summary=lambda a: f"Write {a.path} in '{a.project}'",
    platform="project",
    preview=_preview_write,
    session_group="build-loop",
    session_scope_field="project",
)
def project_write_file(args: WriteFileArgs, preview: dict) -> dict:
    """Create or replace the Dockerfile or .dockerignore at a project's root. Pass the complete file content. Write .dockerignore before a Dockerfile that copies the whole project. The user reviews the exact diff before anything is written."""
    ws = _workspace(args.project)
    # Write exactly what the user approved, and only if the file is still what they reviewed.
    path = ws.write(args.path, preview["file"]["content"], expected_hash=preview["file"]["hash"])
    status = ws.git_file_status(args.path)
    facts = {"File": ws.rel(path), "Lines": str(preview["file"]["lines"])}
    if status in ("untracked", "modified"):
        facts["Git"] = "uncommitted change. Review and commit it when you're happy with it"
    return {"kind": "facts", "title": f"Wrote {args.path} in {ws.name}", "facts": facts}
