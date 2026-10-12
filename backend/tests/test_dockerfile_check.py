import pytest

from app.dockerfile_check import check_dockerfile, check_dockerignore, ignored, parse

GOOD = """\
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && \\
    uv sync --frozen --no-dev
COPY . .
RUN useradd --create-home app
USER app
EXPOSE 8080
CMD ["uv", "run", "python", "-m", "app"]
"""
IGNORE = ".env\n.env.*\n!.env.example\n.git\n.venv\n__pycache__\n"
TOP = {"app", "pyproject.toml", "uv.lock", ".env", ".env.example", ".git", ".venv", "README.md"}


def test_parse_joins_continuations_and_skips_comments():
    steps = parse(GOOD)
    assert [s.command for s in steps][:4] == ["FROM", "WORKDIR", "COPY", "RUN"]
    assert steps[3].args.startswith("pip install uv") and "uv sync" in steps[3].args and steps[3].line == 5


def test_good_dockerfile_passes_with_facts():
    r = check_dockerfile(GOOD, IGNORE, TOP)
    assert r.errors == [] and r.warnings == []
    assert r.facts["Port"] == "8080" and r.facts["Runs as"] == "app" and r.facts["Base image"] == "python:3.12-slim"
    assert "python" in r.facts["Starts with"]


@pytest.mark.parametrize(
    "dockerfile, ignore, error",
    [
        ("RUN echo hi\nCMD x", IGNORE, "must be FROM, but line 1 is 'RUN echo hi'"),
        (".dockerignore\n.env\n\nFROM python:3.12\nCMD x", IGNORE, "but line 1 is '.dockerignore'"),
        ("FROM python:3.12\nRUN echo hi", IGNORE, "CMD or ENTRYPOINT"),
        ("FROM python:3.12\nCOPY .env /app/.env\nCMD x", IGNORE, "secret file"),
        ("FROM python:3.12\nCOPY . .\nCMD x", None, "no .dockerignore yet"),
        ("FROM python:3.12\nCOPY . .\nCMD x", ".venv\n", "doesn't exclude .env, .git"),
        ("FROM python:3.12\nENV API_KEY=abc123\nCMD x", IGNORE, "hard-codes a value for API_KEY"),
        ("FROM python:3.12\nENV DB_PASSWORD hunter2\nCMD x", IGNORE, "hard-codes a value for DB_PASSWORD"),
        ("FROM python:3.12\nENV TOKEN=[REDACTED]\nCMD x", IGNORE, "masked value"),
        ("", IGNORE, "empty"),
        ('FROM python:3.12\nCMD ["uvicorn", "app:app", "--port", "$APP_PORT"]', IGNORE, "doesn't expand variables"),
    ],
)
def test_blocking_problems(dockerfile, ignore, error):
    errors = check_dockerfile(dockerfile, ignore, TOP).errors
    assert any(error in e for e in errors), errors


def test_allowed_patterns_are_not_errors():
    text = (
        "ARG PY=3.12\nFROM python:${PY}-slim AS build\nENV APP_PORT=8080 API_KEY=${API_KEY}\n"
        "ARG GITHUB_TOKEN\nFROM gcr.io/distroless/python3@sha256:abc\n"
        "COPY --from=build /app/.env.example /app/\nUSER 1000\nEXPOSE 8080\nENTRYPOINT [\"python\"]"
    )
    assert check_dockerfile(text, None, TOP).errors == []


@pytest.mark.parametrize(
    "dockerfile, warning",
    [
        ("FROM python\nUSER app\nEXPOSE 1\nCMD x", "isn't pinned"),
        ("FROM python:latest\nUSER app\nEXPOSE 1\nCMD x", "isn't pinned"),
        ("FROM python:3.12\nEXPOSE 1\nCMD x", "runs as root"),
        ("FROM python:3.12\nUSER root\nEXPOSE 1\nCMD x", "runs as root"),
        ("FROM python:3.12\nUSER app\nCMD x", "No EXPOSE"),
        ("FROM python:3.12\nADD https://x.io/f /f\nUSER app\nEXPOSE 1\nCMD x", "ADD with a URL"),
    ],
)
def test_warnings(dockerfile, warning):
    r = check_dockerfile(dockerfile, IGNORE, TOP)
    assert r.errors == [] and any(warning in w for w in r.warnings), r.warnings


def test_dockerignore_rules():
    assert check_dockerignore(IGNORE, TOP).errors == []
    r = check_dockerignore(".venv\n", TOP)
    assert any("'.env'" in e for e in r.errors) and any("'.git'" in e for e in r.errors)
    assert any("'__pycache__'" in w or "'.venv'" in w for w in check_dockerignore(".env\n.git\n", TOP | {"node_modules"}).warnings)
    assert check_dockerignore(".env\n.git\n!.env\n", TOP).errors  # re-including .env is caught


def test_ignored_matching():
    assert ignored(".env", ["**/.env"]) and ignored(".git", [".git/"]) and ignored(".env.local", [".env*"])
    assert not ignored(".env.example", [".env*", "!.env.example"]) and not ignored("app", [".env"])



def test_lines_that_arent_instructions_are_refused():
    from app.dockerfile_check import check_dockerfile
    text = 'FROM python:3.12-slim\nCMD ["python", "-m", "app"]\n\n</parameter reason="x"\n'
    errors = check_dockerfile(text, ".env\n", set()).errors
    assert any("Line 4: `</parameter reason=\"x\"` isn't a Dockerfile instruction" in e for e in errors)


def test_heredoc_lines_are_not_instructions():
    from app.dockerfile_check import check_dockerfile
    text = "FROM python:3.12-slim\nRUN <<EOF\napt-get update\npip install uv\nEOF\nCMD [\"python\", \"-m\", \"app\"]\n"
    assert not any("isn't a Dockerfile instruction" in e for e in check_dockerfile(text, ".env\n", set()).errors)
