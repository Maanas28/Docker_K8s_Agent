import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.main as main
from app.store import Store
from app.tools import code_tools


@pytest.fixture
def client(monkeypatch):
    store = Store("sqlite://")
    monkeypatch.setattr(main, "store", store)
    code_tools.configure(store)
    return TestClient(main.app)


GOOD = {"Origin": "http://localhost:5173"}
EVIL = {"Origin": "https://evil.example"}


def test_other_websites_are_refused(client):
    assert client.get("/api/projects", headers=EVIL).status_code == 403
    assert client.post("/api/projects", json={"path": "/tmp"}, headers=EVIL).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/chat", headers=EVIL) as ws:
            ws.receive_json()


def test_the_app_and_local_scripts_are_allowed(client):
    assert client.get("/api/projects", headers=GOOD).status_code == 200
    assert client.get("/api/projects").status_code == 200  # no Origin header (curl)


def test_register_and_remove_project(client, tmp_path):
    folder = tmp_path / "My Cool App"
    folder.mkdir()
    created = client.post("/api/projects", json={"path": str(folder)}, headers=GOOD)
    assert created.status_code == 201 and created.json()["name"] == "my-cool-app"
    assert client.post("/api/projects", json={"path": str(folder)}, headers=GOOD).status_code == 409
    assert client.post("/api/projects", json={"path": str(folder), "name": "my-cool-app"}).status_code == 409
    assert client.get("/api/projects").json()[0]["reads_always_allowed"] is False
    assert client.delete("/api/projects/my-cool-app", headers=GOOD).status_code == 200
    assert client.get("/api/projects").json() == []


@pytest.mark.parametrize("path", ["/", "~", "/etc", "/does/not/exist"])
def test_unsafe_folders_are_refused(client, path):
    assert client.post("/api/projects", json={"path": path}, headers=GOOD).status_code == 400


def test_revoke_project_reads(client, tmp_path):
    client.post("/api/projects", json={"path": str(tmp_path), "name": "p"})
    main.store.allow_project_reads("p")
    assert client.delete("/api/projects/p/read-grant").status_code == 200
    assert client.delete("/api/projects/p/read-grant").status_code == 404
