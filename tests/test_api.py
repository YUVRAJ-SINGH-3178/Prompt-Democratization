from fastapi.testclient import TestClient

from app import auth_db
from app.main import app


def test_legacy_apis_require_authenticated_university_session(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_db, "DB_PATH", tmp_path / "auth.sqlite")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/auth/me").json() == {"authenticated": False}
        assert client.get("/dashboard.html", follow_redirects=False).status_code == 303
        for path in (
            "/api/overview",
            "/api/templates",
            "/api/linter",
            "/api/builder",
            "/api/routing",
            "/api/experiments",
            "/api/data",
        ):
            assert client.get(path).status_code == 401
