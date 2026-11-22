import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import auth_db, database, simulated_database
from app.main import app
from scripts.generate_personas import SEED, personas


def test_personas_are_seeded_complete_and_cover_required_disciplines():
    first = personas()
    assert first == personas()
    assert len(first) == 20
    assert all(p["synthetic"] is True and p["source"] == "sim_persona" for p in first)
    assert {p["id"] for p in first} == {f"SIM-P{i:02d}" for i in range(1, 21)}
    assert {"Literature", "Engineering", "Medical", "Business", "Computing"} <= {p["discipline"] for p in first}
    assert {"Machine Learning", "Database Management Systems", "Operating Systems", "Computer Networks"} <= {
        p["course"] for p in first
    }
    assert sum(p["prompt_writing_habit"].startswith("poor_") for p in first) >= 4
    assert sum(p["prompt_writing_habit"] == "good_complete" for p in first) >= 3
    assert all(len(p["background"].split()) <= 60 for p in first)
    assert SEED == 731204


def test_simulated_rows_are_tagged_and_never_enter_real_feedback_query(monkeypatch):
    sim_path = Path(__file__).with_name("_simulated_test.sqlite")
    real_path = Path(__file__).with_name("_simulated_real_test.sqlite")
    sim_path.unlink(missing_ok=True)
    real_path.unlink(missing_ok=True)
    monkeypatch.setattr(simulated_database, "DB_PATH", sim_path)
    monkeypatch.setattr(database, "DB_PATH", real_path)
    p = personas()[0]
    with simulated_database.connect() as db:
        db.execute("INSERT INTO personas VALUES(?,?,1,'sim_persona')", (p["id"], json.dumps(p)))
        db.execute("INSERT INTO templates VALUES('SIM-T01','test','quiz','test',1,'sim_persona',?)", (p["id"],))
        db.execute(
            "INSERT INTO ratings(persona_id,template_id,repetition,rating,reason,synthetic,source) VALUES(?, 'SIM-T01',1,4,'test',1,'sim_persona')",
            (p["id"],),
        )
        assert simulated_database.audit(db)["all_simulated_rows_isolated"] is True
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO ratings(persona_id,template_id,repetition,rating,reason,synthetic,source) VALUES(?, 'SIM-T01',2,4,'bad',0,'sim_persona')",
                (p["id"],),
            )
    with database.connect() as db:
        assert "ratings" not in {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        db.execute(
            "INSERT INTO templates(id,name,task_type,prompt,status,source,synthetic) VALUES('REAL-T','real','quiz','p','approved','user',0)"
        )
        db.execute("INSERT INTO feedback(template_id,feature_json,rating,synthetic) VALUES('REAL-T','{}',3,0)")
        db.execute("INSERT INTO feedback(template_id,feature_json,rating,synthetic) VALUES('REAL-T','{}',5,1)")
    assert len(database.get_feedback()) == 1
    sim_path.unlink(missing_ok=True)
    real_path.unlink(missing_ok=True)


def test_simulated_analysis_empty_state_is_explicit():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.executescript(simulated_database.SCHEMA)
        from app.simulated_analysis import analyze

        result = analyze(db)
    assert result["label"] == "SIMULATED"
    assert result["status"] == "awaiting model calls"
    assert "no real-world faculty grounding" in result["validity_limits"][1]


def test_simulated_api_banner_and_real_score_import_are_isolated(monkeypatch):
    auth_db_path = Path(__file__).with_name("_simulated_auth_test.sqlite")
    auth_db_path.unlink(missing_ok=True)
    monkeypatch.setattr(auth_db, "DB_PATH", auth_db_path)
    db_path = Path(__file__).with_name("_simulated_api_test.sqlite")
    db_path.unlink(missing_ok=True)
    monkeypatch.setattr(simulated_database, "DB_PATH", db_path)
    persona = personas()[0]
    with simulated_database.connect() as db:
        db.execute("INSERT INTO personas VALUES(?,?,1,'sim_persona')", (persona["id"], json.dumps(persona)))
        for item_id in ("SIM-P01-P01", "SIM-P01-P02"):
            db.execute(
                "INSERT INTO annotations(persona_id,prompt_id,label,atoms_json,slots_json,synthetic,source) VALUES(?,?,'fine','[]','{}',1,'sim_persona')",
                (persona["id"], item_id),
            )
    with TestClient(app) as client:
        response = client.get("/api/simulated")
        assert response.status_code == 401
    with simulated_database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM real_faculty_scores").fetchone()[0] == 0
    db_path.unlink(missing_ok=True)
    auth_db_path.unlink(missing_ok=True)
