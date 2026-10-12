"""Rule checks for Dockerfiles and .dockerignore files the agent wants to write.

Errors block the write and go back to the model to fix (no approval card is shown).
Warnings are shown on the approval card so the user can decide.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field

from app.redact import MASK
from app.workspace import is_secret_file

# Every instruction Docker's parser accepts. Anything else on its own line fails the build with
# "unknown instruction", so it's refused when the file is written instead.
INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT", "VOLUME",
    "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
}
# A heredoc (`RUN <<EOF` … `EOF`): the lines in between are its content, not instructions.
_HEREDOC = re.compile(r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")

_SECRET_NAME = re.compile(r"(?i)(secret|token|passw(or)?d|pwd|api[_-]?key|access[_-]?key|private[_-]?key|credential)")


@dataclass
class Instruction:
    command: str  # upper-case, e.g. "FROM"
    args: str
    line: int
    raw: str = ""  # the instruction as written, for error messages


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)


def parse(text: str) -> list[Instruction]:
    """Split a Dockerfile into instructions, joining `\\` line continuations and skipping comments."""
    out: list[Instruction] = []
    buffer, start = "", 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if buffer and line.startswith("#"):
            continue  # comments inside a continued instruction
        if not buffer:
            start = number
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        command, _, args = buffer.partition(" ")
        out.append(Instruction(command.upper(), args.strip(), start, buffer))
        buffer = ""
    if buffer:
        command, _, args = buffer.partition(" ")
        out.append(Instruction(command.upper(), args.strip(), start, buffer))
    return out


def _copy_sources(args: str) -> list[str] | None:
    """Source paths of a COPY/ADD, or None when copying from another build stage (--from)."""
    if args.startswith("["):  # JSON form: COPY ["src", "dest"]
        items = re.findall(r'"([^"]*)"', args)
    else:
        try:
            items = shlex.split(args)
        except ValueError:
            items = args.split()
    flags = [i for i in items if i.startswith("--")]
    if any(f.startswith("--from") for f in flags):
        return None
    paths = [i for i in items if not i.startswith("--")]
    return paths[:-1]  # the last one is the destination


def ignored(name: str, patterns: list[str]) -> bool:
    """Would `name` (a top-level file or folder) be excluded by these .dockerignore patterns?"""
    excluded = False
    for raw in patterns:
        negate = raw.startswith("!")
        pattern = raw.lstrip("!").strip().lstrip("/").rstrip("/")
        if not pattern:
            continue
        for candidate in {pattern, pattern.removeprefix("**/")}:
            if fnmatch.fnmatch(name, candidate) or candidate in {name + "/**", name + "/*"}:
                excluded = not negate
                break
    return excluded


def dockerignore_patterns(text: str | None) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def check_dockerignore(text: str, top_level: set[str]) -> CheckResult:
    """`top_level` is the set of file and folder names at the project root."""
    result = CheckResult()
    patterns = dockerignore_patterns(text)
    if MASK in text:
        result.errors.append(f"It contains '{MASK}', a masked value; write real patterns only.")
    must = [n for n in sorted(top_level) if is_secret_file(n)]
    if ".git" in top_level:
        must.append(".git")
    for name in must:
        if not ignored(name, patterns):
            result.errors.append(f"It must exclude '{name}' so it never ends up in the image (e.g. add `{name}`).")
    for name in (".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build", "target"):
        if name in top_level and not ignored(name, patterns):
            result.warnings.append(f"'{name}' isn't excluded, so it will be sent to Docker and may bloat the image.")
    result.facts["Excludes"] = ", ".join(patterns[:12]) + (" …" if len(patterns) > 12 else "") or "nothing"
    return result


def check_dockerfile(text: str, dockerignore: str | None, top_level: set[str]) -> CheckResult:
    """`dockerignore` is the project's current .dockerignore content (None if it has none)."""
    result = CheckResult()
    if MASK in text:
        result.errors.append(f"It contains '{MASK}', a masked value from a file you read. Don't copy masked values.")
    steps = parse(text)
    if not steps:
        result.errors.append("It's empty.")
        return result

    terminators: list[str] = []
    for s in steps:
        if terminators:
            if s.raw.strip() == terminators[0]:
                terminators.pop(0)
            continue  # a line inside a heredoc
        if s.command not in INSTRUCTIONS:
            result.errors.append(
                f"Line {s.line}: `{s.raw.strip()[:60]}` isn't a Dockerfile instruction, so Docker would "
                "fail to parse the file. Remove it, or put it after a # if it's meant as a comment."
            )
        terminators = _HEREDOC.findall(s.args)

    first = next((s for s in steps if s.command != "ARG"), None)
    if first is None or first.command != "FROM":
        found = first.raw.strip()[:60] if first else "nothing"
        where = f"line {first.line} is '{found}'" if first else "there is no instruction"
        result.errors.append(f"The first instruction (after optional ARGs) must be FROM, but {where}.")

    stages = [i for i, s in enumerate(steps) if s.command == "FROM"]
    final = steps[stages[-1]:] if stages else steps
    stage_names = {m.group(1).lower() for s in steps if s.command == "FROM"
                   if (m := re.search(r"\bAS\s+(\S+)", s.args, re.I))}

    if not any(s.command in ("CMD", "ENTRYPOINT") for s in final):
        result.errors.append("The final stage needs a CMD or ENTRYPOINT that starts the app.")
    for s in steps:
        if s.command in ("CMD", "ENTRYPOINT") and s.args.startswith("[") and "$" in s.args:
            result.errors.append(
                f"Line {s.line}: {s.command} [...] (exec form) doesn't expand variables like $APP_PORT, "
                "so the app would get the literal text. Use a fixed value, or let the app read the env var itself."
            )

    patterns = dockerignore_patterns(dockerignore)
    secrets_at_root = [n for n in top_level if is_secret_file(n)]
    for s in steps:
        if s.command in ("COPY", "ADD"):
            sources = _copy_sources(s.args)
            if sources is None:
                continue
            for src in sources:
                base = src.rstrip("/").split("/")[-1]
                if base and is_secret_file(base):
                    result.errors.append(f"Line {s.line} copies '{src}', a secret file. Pass secrets at run time instead.")
                if src in (".", "./", "*"):
                    if dockerignore is None:
                        result.errors.append(
                            f"Line {s.line} copies the whole project but there's no .dockerignore yet. "
                            "Write the .dockerignore first."
                        )
                    else:
                        leaked = [n for n in secrets_at_root + ([".git"] if ".git" in top_level else [])
                                  if not ignored(n, patterns)]
                        if leaked:
                            result.errors.append(
                                f"Line {s.line} copies the whole project, and .dockerignore doesn't exclude "
                                f"{', '.join(leaked)}. Fix the .dockerignore first."
                            )
            if s.command == "ADD" and re.search(r"https?://", s.args):
                result.warnings.append(f"Line {s.line} uses ADD with a URL; downloads aren't verified. Prefer RUN curl with a checksum.")
        if s.command in ("ENV", "ARG"):
            for name, value in _assignments(s):
                if _SECRET_NAME.search(name) and value and not value.startswith("$"):
                    result.errors.append(
                        f"Line {s.line} hard-codes a value for {name}. Secrets must be passed when the "
                        "container runs (env file), not baked into the image."
                    )

    bases = []
    for s in steps:
        if s.command != "FROM":
            continue
        image = next((t for t in s.args.split() if not t.startswith("--")), "")
        bases.append(image)
        lower = image.lower()
        if lower == "scratch" or lower in stage_names or "$" in image:
            continue
        name_part = lower.split("@")[0]
        tag = name_part.rsplit(":", 1)[1] if ":" in name_part.split("/")[-1] else None
        if "@sha256:" not in lower and (tag is None or tag == "latest"):
            result.warnings.append(f"Base image '{image}' isn't pinned to a version; builds may change over time.")

    users = [s.args.split()[0] for s in final if s.command == "USER" and s.args]
    if not users or users[-1].split(":")[0] in ("root", "0"):
        result.warnings.append("The container runs as root. Add a non-root USER in the final stage.")
    exposed = [p for s in final if s.command == "EXPOSE" for p in s.args.split()]
    if not exposed:
        result.warnings.append("No EXPOSE, so it's not documented which port the app listens on.")

    start = next((s for s in reversed(final) if s.command in ("CMD", "ENTRYPOINT")), None)
    result.facts.update({
        "Base image": " → ".join(bases) or "?",
        "Stages": str(len(stages)),
        "Starts with": start.args if start else "?",
        "Port": ", ".join(exposed) or "not declared",
        "Runs as": users[-1] if users else "root",
    })
    return result


def _assignments(step: Instruction) -> list[tuple[str, str]]:
    """(name, value) pairs of an ENV/ARG, in both `K=V K2=V2` and legacy `K V` forms."""
    args = step.args
    if "=" not in args.split(" ", 1)[0]:
        name, _, value = args.partition(" ")
        return [(name, value.strip().strip('"\''))] if step.command == "ENV" else [(name, "")]
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    pairs = []
    for token in tokens:
        name, sep, value = token.partition("=")
        pairs.append((name, value if sep else ""))
    return pairs
