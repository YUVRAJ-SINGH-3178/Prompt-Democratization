from fastapi.testclient import TestClient

from app import auth_db
from app.auth_db import connect
from app.main import app
from scripts.seed_demo_users import seed


def _login(client, username, password):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    value = response.json()
    return value["csrf_token"], value


def test_role_scopes_csrf_audit_samples_and_empty_accuracy(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_db, "DB_PATH", tmp_path / "roles.sqlite")
    credentials = seed(credentials_path=tmp_path / "credentials.txt")
    passwords = {username: password for username, _, password in credentials}
    with connect() as db:
        db.execute(
            "INSERT INTO managed_records(record_id,entity_type,course_id,data_json,status) VALUES('template-ml','templates','course-ml','{\"name\":\"ML approved\",\"prompt\":\"Write #question_count questions about #topic\"}','approved')"
        )

    with TestClient(app) as client:
        # Public routes are restricted to the login page and health check.
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/admin/overview").status_code == 401
        assert client.get("/api/overview").status_code == 401

        uni_csrf, uni = _login(client, "university", passwords["university"])
        assert client.get("/admin").status_code == 200
        assert client.get("/faculty").status_code == 403
        assert client.post("/api/admin/records/departments", json={"data": {"name": "Test"}}).status_code == 403
        headers = {"X-CSRF-Token": uni_csrf}
        created = client.post(
            "/api/admin/records/departments", headers=headers, json={"record_id": "test-dept", "data": {"name": "Test"}}
        )
        assert created.status_code == 201
        assert client.delete("/api/admin/records/departments/test-dept", headers=headers).status_code == 200
        assert client.post("/api/admin/records/departments/test-dept/restore", headers=headers).status_code == 200
        accuracy = client.get("/api/admin/accuracy").json()
        assert accuracy["status"] == "awaiting annotations" and accuracy["heldout_prompts_n"] == 0
        samples = client.get("/api/samples").json()
        assert samples["n"] == 6
        assert all(x["label"] == "SAMPLE" and x["numeric_match"] for x in samples["items"])
        assert "original prompt text unavailable" in samples["provenance"]

        # A coordinator receives only assigned-course data; out-of-scope reads are denied and audited.
        client.post("/api/auth/logout", headers=headers)
        coord_csrf, _ = _login(client, "coordinator.a", passwords["coordinator.a"])
        coord_headers = {"X-CSRF-Token": coord_csrf}
        assert set(client.get("/api/coordinator/overview").json()["course_ids"]) == {"course-ml", "course-dbs"}
        assert client.get("/api/coordinator/records/courses?course_id=course-marketing").status_code == 403
        assert client.get("/api/admin/users").status_code == 403
        check = client.post(
            "/api/prompt-checker/check",
            headers=coord_headers,
            json={
                "prompt": "expert level, very easy questions",
                "template_id": "template-ml",
                "course_id": "course-marketing",
            },
        )
        assert check.status_code == 403

        client.post("/api/auth/logout", headers=coord_headers)
        fac_csrf, _ = _login(client, "faculty.a", passwords["faculty.a"])
        fac_headers = {"X-CSRF-Token": fac_csrf}
        assert client.get("/faculty").status_code == 200
        assert client.get("/coordinator").status_code == 403
        assert client.get("/api/coordinator/overview").status_code == 403
        result = client.post(
            "/api/prompt-checker/check", headers=fac_headers, json={"sample_id": samples["items"][0]["id"]}
        )
        assert result.status_code == 200
        assert result.json()["source"] == "sample"
        assert client.get("/api/faculty/history").json()["n"] == 1

        with connect() as db:
            actions = {r[0] for r in db.execute("SELECT action FROM audit_log")}
            assert {"denied", "create", "soft_delete", "restore", "login_success"} <= actions
            assert db.execute("SELECT COUNT(*) FROM audit_log WHERE details_json LIKE '%password%'").fetchone()[0] == 0
            stored = db.execute("SELECT password_hash FROM auth_users WHERE username='university'").fetchone()[0]
            assert stored.startswith("$argon2id$") and passwords["university"] not in stored
