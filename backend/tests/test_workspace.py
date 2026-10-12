import os

import pytest

from app.tools.registry import ToolError
from app.workspace import Workspace, is_secret_file, safe_project_path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "app" / "main.py").write_text('PORT = 8080\nTOKEN = "super-secret-value"\n')
    (root / ".env").write_text("# comment\nAPP_PORT=8080\nexport API_KEY=real-key-123\n")
    (root / ".env.example").write_text("APP_PORT=\nAPI_KEY=\n")
    (root / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nxyz\n-----END OPENSSH PRIVATE KEY-----")
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "node_modules" / "lib" / "index.js").write_text("PORT = 1")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]")
    (root / "logo.png").write_bytes(b"\x89PNG\0\0data")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret")
    os.symlink(outside, root / "link-out")
    return root


def test_paths_cannot_leave_the_project(project):
    ws = Workspace("proj", project)
    for bad in ["../outside.txt", "/etc/passwd", "~/.ssh/id_rsa", "app/../../outside.txt", "link-out"]:
        with pytest.raises(ToolError, match="outside the project|relative to the project"):
            ws.read(bad)
    with pytest.raises(ToolError, match="don't read"):
        ws.read(".git/config")


def test_secret_files_are_never_read(project):
    ws = Workspace("proj", project)
    env = ws.read(".env")
    assert env["kind"] == "facts" and env["facts"]["Variables"] == "APP_PORT, API_KEY"
    assert "real-key-123" not in str(env)
    key = ws.read("id_rsa")
    assert "xyz" not in str(key) and "Variables" not in key["facts"]


def test_env_templates_are_readable(project):
    assert Workspace("proj", project).read(".env.example")["kind"] == "file"


def test_is_secret_file():
    for name in [".env", ".env.local", ".env.production", "prod.env", "server.pem", "id_ed25519", "credentials.json", ".npmrc"]:
        assert is_secret_file(name), name
    for name in [".env.example", ".env.sample", "main.py", "environment.py", "keys.py"]:
        assert not is_secret_file(name), name


def test_reading_masks_secret_values(project):
    result = Workspace("proj", project).read("app/main.py")
    assert "super-secret-value" not in result["text"] and result["redacted"] == 1
    assert "PORT = 8080" in result["text"]


def test_binary_and_line_ranges(project):
    ws = Workspace("proj", project)
    with pytest.raises(ToolError, match="binary"):
        ws.read("logo.png")
    (project / "long.txt").write_text("\n".join(f"line {i}" for i in range(1, 101)))
    part = ws.read("long.txt", start_line=10, max_lines=5)
    assert part["text"].splitlines() == [f"line {i}" for i in range(10, 15)] and part["truncated"]


def test_tree_skips_dependency_folders_and_outside_links(project):
    tree, _ = Workspace("proj", project).tree()
    assert "node_modules" not in tree and ".git" not in tree and "link-out" not in tree
    assert ".env (secret, not read)" in tree and "main.py" in tree


def test_search_skips_secrets_and_dependencies(project):
    rows, _ = Workspace("proj", project).search("PORT")
    paths = {r["path"] for r in rows}
    assert "app/main.py" in paths and ".env" not in paths
    assert not any(p.startswith("node_modules") for p in paths)
    rows, _ = Workspace("proj", project).search("TOKEN")
    assert "super-secret-value" not in rows[0]["text"]


def test_invalid_regex_falls_back_to_plain_text(project):
    rows, _ = Workspace("proj", project).search("PORT = (")
    assert rows == []  # no crash


def test_git_info(project, tmp_path):
    assert Workspace("proj", project).git_info() is not None  # .git exists (not a real repo)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert Workspace("plain", plain).git_info() is None


def test_safe_project_path(tmp_path):
    from pathlib import Path

    assert safe_project_path(str(tmp_path)) == tmp_path.resolve()
    for bad in ["/", str(Path.home()), "~", "/etc", "/usr/local", str(tmp_path / "missing"), "~/.ssh"]:
        with pytest.raises(ValueError):
            safe_project_path(bad)


def test_writes_only_allowed_files_inside_the_project(project):
    ws = Workspace("proj", project)
    for bad in ["app/main.py", ".env", "../Dockerfile", "app/Dockerfile"]:
        with pytest.raises(ToolError):
            ws.plan_write(bad, "FROM x")
    plan = ws.plan_write("Dockerfile", "FROM python:3.12\nCMD x")
    assert plan["new"] and plan["hash"] is None and plan["content"].endswith("\n")
    assert plan["diff"].startswith("--- /dev/null\n+++ b/Dockerfile")


def test_write_is_atomic_and_checks_the_reviewed_version(project):
    ws = Workspace("proj", project)
    plan = ws.plan_write("Dockerfile", "FROM a\n")
    ws.write("Dockerfile", plan["content"], plan["hash"])
    assert (project / "Dockerfile").read_text() == "FROM a\n"
    replace = ws.plan_write("Dockerfile", "FROM b\n")
    assert not replace["new"] and "-FROM a" in replace["diff"] and "+FROM b" in replace["diff"]
    (project / "Dockerfile").write_text("FROM edited-by-you\n")  # changed after review
    with pytest.raises(ToolError, match="changed after you reviewed"):
        ws.write("Dockerfile", replace["content"], replace["hash"])
    assert (project / "Dockerfile").read_text() == "FROM edited-by-you\n"
    assert not [p for p in project.iterdir() if p.name.endswith(".tmp")]  # no temp files left


def test_write_refuses_symlinked_targets(project, tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("x")
    os.symlink(target, project / ".dockerignore")
    with pytest.raises(ToolError):
        Workspace("proj", project).plan_write(".dockerignore", ".env\n")


def test_lock_files_are_described_not_read(project):
    (project / "uv.lock").write_text("version = 1\n" * 5000)
    ws = Workspace("proj", project)
    note = ws.read("uv.lock")
    assert note["kind"] == "facts" and "Read pyproject.toml" in note["facts"]["Note"]
    assert "version = 1" not in str(note)
    (project / "pyproject.toml").write_text('[project]\nname = "x"\n')
    rows, _ = ws.search("version")
    assert not any(r["path"] == "uv.lock" for r in rows)


def test_key_files_pick_what_explains_the_project(project):
    (project / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (project / "app" / "__main__.py").write_text("import uvicorn\n")
    (project / "app" / "config.py").write_text("port = 8080\n")
    (project / "uv.lock").write_text("x" * 50_000)
    (project / "node_modules" / "lib" / "main.py").write_text("ignored")
    files, skipped = Workspace("proj", project).key_files()
    paths = [f["path"] for f in files]
    assert paths[0] == "pyproject.toml" and "app/__main__.py" in paths and "app/config.py" in paths
    assert ".env" not in paths and ".env.example" in paths  # template yes, secret no
    assert not any(p.startswith("node_modules") for p in paths)
    assert {"path": "uv.lock", "reason": "lock file (50 KB), not needed"} in skipped
    assert "super-secret-value" not in str(files)  # app/main.py's TOKEN is masked


def test_key_files_respect_the_budget(project):
    (project / "pyproject.toml").write_text("a" * 5000)
    (project / "app" / "config.py").write_text("b" * 5000)
    files, skipped = Workspace("proj", project).key_files(budget=6000, per_file=4000)
    assert sum(len(f["text"]) for f in files) <= 6000
    assert files[0]["truncated"] and any("budget" in s["reason"] for s in skipped)
