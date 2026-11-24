"""Unit tests for combining reader and rules, energy range, improve gate, and review queue (Task 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import auth_db
from app.auth_db import connect
from app.extractor.combine import (
    combine_rules_and_reader,
)
from app.linter.improve import improve_prompt
from app.main import app
from scripts.seed_demo_users import seed


def test_review_queue_stores_human_decisions_with_user_and_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review queue stores human decisions with user and time, and human decisions override machines."""
    test_db = tmp_path / "test_auth.sqlite"
    monkeypatch.setattr(auth_db, "DB_PATH", test_db)
    creds_file = tmp_path / "credentials.txt"
    credentials = seed(credentials_path=creds_file)
    passwords = {username: password for username, _, password in credentials}

    client = TestClient(app)
    login_resp = client.post("/api/auth/login", json={"username": "faculty.a", "password": passwords["faculty.a"]})
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    csrf = login_resp.json()["csrf_token"]
    headers = {"X-CSRF-Token": csrf}

    prompt = "Sample prompt for review queue test P999."
    # Insert open review item
    with connect() as db:
        cur = db.execute(
            "INSERT INTO manual_reviews(prompt_id, prompt_text, item_type, item_name, verdict, quote, rule_opinion, reader_opinion, status) "
            "VALUES (?, ?, 'atom', 'very_easy', 'reader_only', 'very easy', 'absent', 'present', 'open')",
            ("test_p999", prompt),
        )
        item_id = cur.lastrowid

    # 1. GET /api/reviews/queue
    resp = client.get("/api/reviews/queue", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    assert any(x["id"] == item_id for x in items)

    # 2. POST /api/reviews/{id}/decision (confirm decision)
    dec_resp = client.post(
        f"/api/reviews/{item_id}/decision",
        json={"decision": "confirm", "value": "1"},
        headers=headers,
    )
    assert dec_resp.status_code == 200
    dec_data = dec_resp.json()
    assert dec_data["status"] == "confirmed"
    assert dec_data["decided_by"] == "faculty.a"
    assert "decided_at" in dec_data

    # Verify stored in DB with user and timestamp
    with connect() as db:
        row = db.execute("SELECT * FROM manual_reviews WHERE id=?", (item_id,)).fetchone()
        assert row["status"] == "confirmed"
        assert row["decided_by"] == "faculty.a"
        assert row["decided_at"] is not None

    # Verify item is no longer in open queue
    resp2 = client.get("/api/reviews/queue", headers=headers)
    assert not any(x["id"] == item_id for x in resp2.json()["items"])

    # 3. Test human decision overrides machines in combine_rules_and_reader!
    # Even if reader and rules say absent, the human confirmed it!
    res = combine_rules_and_reader(
        prompt,
        prompt_id="test_p999",
        reader_output={"enabled": True, "status": "success", "atoms": {}, "slots": {}},
        record_reviews=False,
    )
    assert "very_easy" in res.accepted_atoms
    assert res.atom_verdicts["very_easy"]["verdict"] == "human_confirmed"
    assert res.atom_verdicts["very_easy"]["human_override"] is True
    assert res.atom_verdicts["very_easy"]["decided_by"] == "faculty.a"


def test_verdict_agree_auto_accepted() -> None:
    """When reader and rules say the same and evidence quote passed, auto-accept."""
    prompt = "Create 10 questions. All questions must be very easy. Four choices per question."
    mock_reader = {
        "enabled": True,
        "status": "success",
        "atoms": {
            "very_easy": {
                "present": True,
                "negated": False,
                "asserted": True,
                "quote_valid": True,
                "unsupported": False,
                "evidence": "very easy",
                "span": (36, 45),
            }
        },
        "slots": {
            "count": {
                "score": 1.0,
                "evidence": "10 questions",
                "quote_valid": True,
                "unsupported": False,
                "span": (7, 19),
            }
        },
    }

    res = combine_rules_and_reader(prompt, reader_output=mock_reader, record_reviews=False)
    assert "very_easy" in res.accepted_atoms
    assert res.atom_verdicts["very_easy"]["verdict"] == "agree"
    assert res.atom_verdicts["very_easy"]["accepted"] is True
    assert res.atom_verdicts["very_easy"]["needs_review"] is False
    assert res.accepted_slots["count"] == 1.0
    assert res.slot_verdicts["count"]["verdict"] == "agree"


def test_energy_range_returned_when_atom_needs_review() -> None:
    """When any atom is needs_review, return an energy range: excluded vs included."""
    # Prompt has 'very_easy' matched by rules. Reader asserts 'expert_level' with novel quote not in rule regex.
    prompt = "Create 10 questions. All questions must be very easy. Target Ph.D. dissertation level depth."
    mock_reader = {
        "enabled": True,
        "status": "success",
        "atoms": {
            "very_easy": {
                "present": True,
                "negated": False,
                "asserted": True,
                "quote_valid": True,
                "unsupported": False,
                "evidence": "very easy",
            },
            "expert_level": {
                "present": True,
                "negated": False,
                "asserted": True,
                "quote_valid": True,
                "unsupported": False,
                "evidence": "Ph.D. dissertation level depth",
            },
        },
        "slots": {},
    }

    res = combine_rules_and_reader(prompt, reader_output=mock_reader, record_reviews=False)

    # expert_level is reader_only because rules didn't find it
    assert "expert_level" in res.review_atoms
    assert res.atom_verdicts["expert_level"]["verdict"] == "reader_only"
    assert res.atom_verdicts["expert_level"]["needs_review"] is True

    # Energy range must be returned!
    # Excluded: only very_easy accepted -> energy = 0
    # Included: very_easy + expert_level (conflict!) -> energy = 1
    assert res.has_review_range is True
    assert res.energy_range == [0, 1]
    assert res.energy_min == 0
    assert res.energy_max == 1
    assert res.review_reason is not None
    assert "expert_level" in res.review_reason


def test_disagree_value_slot_routed_to_review() -> None:
    """disagree_value: both give slot score and differ. Take neither, put in review."""
    prompt = "Write questions for students."
    # Rules gives 0.5 (generic student audience), but mock reader returned 1.0 with valid quote
    mock_reader = {
        "enabled": True,
        "status": "success",
        "atoms": {},
        "slots": {
            "audience": {
                "score": 1.0,
                "evidence": "students",
                "quote_valid": True,
                "unsupported": False,
                "span": (20, 28),
            }
        },
    }

    res = combine_rules_and_reader(prompt, reader_output=mock_reader, record_reviews=False)
    # Both gave score and differ: take neither (0.0), needs_review
    assert "audience" in res.review_slots
    assert res.slot_verdicts["audience"]["verdict"] == "disagree_value"
    assert res.slot_verdicts["audience"]["needs_review"] is True
    assert res.accepted_slots["audience"] == 0.0


def test_unsupported_quote_claim_ignored() -> None:
    """unsupported: reader asserted without valid quote. Ignore claim, log it."""
    prompt = "Create 10 questions on algorithms."
    mock_reader = {
        "enabled": True,
        "status": "success",
        "atoms": {
            "hidden_answers": {
                "present": True,
                "negated": False,
                "asserted": False,
                "quote_valid": False,
                "unsupported": True,  # quote was hallucinated!
                "evidence": "keep all answers private",
            }
        },
        "slots": {},
    }

    res = combine_rules_and_reader(prompt, reader_output=mock_reader, record_reviews=False)
    assert "hidden_answers" not in res.accepted_atoms
    assert "hidden_answers" not in res.review_atoms
    assert res.atom_verdicts["hidden_answers"]["verdict"] == "unsupported"


def test_improve_returns_extractor_limited_while_item_open() -> None:
    """The improve gate returns verified only when no atom or slot is needs_review or unclassified.

    Otherwise returns extractor_limited and lists open items.
    """
    prompt = "Create 10 questions on algorithms. Target Ph.D. dissertation level depth."
    mock_reader = {
        "enabled": True,
        "status": "success",
        "atoms": {
            "expert_level": {
                "present": True,
                "negated": False,
                "asserted": True,
                "quote_valid": True,
                "unsupported": False,
                "evidence": "Ph.D. dissertation level depth",  # reader-only assertion
            },
        },
        "slots": {},
    }

    result = improve_prompt(prompt, reader_output=mock_reader)
    # Needs review because expert_level is reader_only
    assert result.verified == "extractor_limited"
    assert result.open_items is not None
    assert len(result.open_items) > 0
    # Must list the open items
    open_names = [item["item_name"] for item in result.open_items]
    assert "expert_level" in open_names or "unclassified_text" in open_names
