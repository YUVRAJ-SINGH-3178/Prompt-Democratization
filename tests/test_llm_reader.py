"""Unit tests for the LLM reader with verifiable evidence (Task 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.env import load_env
from app.extractor.llm_reader import (
    get_cache_key,
    is_negated_phrase,
    is_reader_enabled,
    is_verbatim_quote,
    read_prompt_with_llm,
    validate_and_filter_reader_output,
)


def test_reader_disabled_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reader is off unless a key is set. Without a key the system runs on rules only and says so."""
    monkeypatch.delenv("STUDY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    prompt = "Create 10 MCQs on database normalization for bachelor students."
    res = read_prompt_with_llm(prompt, use_cache=False)

    assert res["enabled"] is False
    assert res["status"] == "disabled_no_api_key"
    assert "Reader is off: no API key set. Running on rules only." in res["message"]


def test_verbatim_substring_evidence_check() -> None:
    """Every quote must be a verbatim substring of the prompt. If missing or not substring, discard."""
    prompt = "Create 10 questions on SQL joins for undergraduate students."

    # Substring check tests
    assert is_verbatim_quote("10 questions", prompt) is True
    assert is_verbatim_quote("SQL joins", prompt) is True
    assert is_verbatim_quote("SQL JOINS", prompt) is True  # case insensitive
    assert is_verbatim_quote("20 questions", prompt) is False
    assert is_verbatim_quote("master students", prompt) is False

    # Mock raw model output with hallucinated/invalid quote
    mock_raw = {
        "atoms": {
            "very_easy": {
                "present": True,
                "negated": False,
                "evidence": "questions are extremely simple for kids",  # NOT in prompt!
            },
            "expert_level": {
                "present": False,
                "negated": False,
                "evidence": "",
            },
        },
        "slots": {
            "role": {
                "score": 1.0,
                "evidence": "Act as an experienced professor",  # NOT in prompt!
            },
            "count": {
                "score": 1.0,
                "evidence": "10 questions",  # Valid!
            },
        },
    }

    filtered = validate_and_filter_reader_output(mock_raw, prompt)

    # very_easy must be discarded because quote was not in prompt
    atom_ve = filtered["atoms"]["very_easy"]
    assert atom_ve["quote_valid"] is False
    assert atom_ve["unsupported"] is True
    assert atom_ve["present"] is False
    assert atom_ve["asserted"] is False

    # role slot score must be reset to 0.0 because quote was not in prompt
    slot_role = filtered["slots"]["role"]
    assert slot_role["quote_valid"] is False
    assert slot_role["unsupported"] is True
    assert slot_role["score"] == 0.0

    # count slot score is valid and kept
    slot_count = filtered["slots"]["count"]
    assert slot_count["quote_valid"] is True
    assert slot_count["unsupported"] is False
    assert slot_count["score"] == 1.0


def test_negation_check_not_asserted() -> None:
    """An atom with negated: true (or negated phrase) is not asserted.

    Required dev-set cases: 'not easy', 'no very easy questions',
    'do not use tricky wording', 'answers should not be hidden'.
    """
    negation_phrases = [
        "not easy",
        "no very easy questions",
        "do not use tricky wording",
        "answers should not be hidden",
    ]
    for phrase in negation_phrases:
        assert is_negated_phrase(phrase) is True

    test_prompt = (
        "Create 10 questions. Questions must be not easy. Include no very easy questions. "
        "Do not use tricky wording. Answers should not be hidden."
    )

    # 1. Model returned present=True, but negated=True
    mock_raw_negated = {
        "atoms": {
            "very_easy": {
                "present": True,
                "negated": True,
                "evidence": "no very easy questions",
            },
            "confusing_language": {
                "present": True,
                "negated": True,
                "evidence": "Do not use tricky wording",
            },
            "hidden_answers": {
                "present": True,
                "negated": True,
                "evidence": "Answers should not be hidden",
            },
        },
        "slots": {},
    }

    filtered = validate_and_filter_reader_output(mock_raw_negated, test_prompt)

    for atom_name in ["very_easy", "confusing_language", "hidden_answers"]:
        atom_data = filtered["atoms"][atom_name]
        assert atom_data["quote_valid"] is True
        assert atom_data["negated"] is True
        # Must NOT be asserted when negated
        assert atom_data["asserted"] is False

    # 2. Model erroneously marked negated=False, but quote clearly contains negation
    mock_raw_unmarked_negation = {
        "atoms": {
            "very_easy": {
                "present": True,
                "negated": False,  # Model missed marking negated
                "evidence": "not easy",
            },
        },
        "slots": {},
    }
    filtered2 = validate_and_filter_reader_output(mock_raw_unmarked_negation, test_prompt)
    atom_ve2 = filtered2["atoms"]["very_easy"]
    # Our validator catches the negated phrase in code
    assert atom_ve2["negated"] is True
    assert atom_ve2["asserted"] is False


def test_caching_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache by SHA-256 of prompt, model, rubric version, schema version."""
    prompt = "Create 5 MCQs on graph traversal."
    key1 = get_cache_key(prompt, "gpt-4o-mini")
    key2 = get_cache_key(prompt, "gpt-4o-mini")
    assert key1 == key2

    # Different prompt or model gives different key
    key3 = get_cache_key(prompt + " Extra sentence.", "gpt-4o-mini")
    assert key1 != key3
    key4 = get_cache_key(prompt, "claude-3-haiku")
    assert key1 != key4


def test_groq_configuration_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """When GROQ_API_KEY is configured, reader enables and routes to Groq endpoint and model."""
    monkeypatch.delenv("STUDY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STUDY_API_BASE", raising=False)
    monkeypatch.delenv("STUDY_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testmockgroqkey12345")

    assert is_reader_enabled() is True

    captured_args = {}

    def mock_call_llm_api(prompt_text, model, key, api_base, seed=42, temperature=0.0):
        captured_args["prompt_text"] = prompt_text
        captured_args["model"] = model
        captured_args["key"] = key
        captured_args["api_base"] = api_base
        payload = {"mock": True}
        resp = {
            "choices": [{"message": {"content": '{"atoms": {}, "slots": {}}'}}],
            "usage": {"total_tokens": 10},
        }
        return payload, resp, 0.05

    monkeypatch.setattr("app.extractor.llm_reader.call_llm_api", mock_call_llm_api)

    prompt = "Create 10 MCQs on database indexing for bachelor students."
    res = read_prompt_with_llm(prompt, use_cache=False)

    assert res["enabled"] is True
    assert captured_args["key"] == "gsk_testmockgroqkey12345"
    assert captured_args["model"] == "llama-3.3-70b-versatile"
    assert captured_args["api_base"] == "https://api.groq.com/openai/v1/chat/completions"


def test_env_loader_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load variables from .env file into os.environ without overwriting existing vars."""
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "# Test comment\n"
        "\n"
        "TEST_NEW_KEY=test_value\n"
        'TEST_QUOTED_KEY="quoted_value"\n'
        "TEST_INLINE_COMMENT=raw_value # some comment\n"
        "TEST_EXISTING_KEY=new_file_value\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TEST_EXISTING_KEY", "already_present_value")
    monkeypatch.delenv("TEST_NEW_KEY", raising=False)
    monkeypatch.delenv("TEST_QUOTED_KEY", raising=False)
    monkeypatch.delenv("TEST_INLINE_COMMENT", raising=False)

    loaded = load_env(fake_env)

    assert loaded["TEST_NEW_KEY"] == "test_value"
    assert loaded["TEST_QUOTED_KEY"] == "quoted_value"
    assert loaded["TEST_INLINE_COMMENT"] == "raw_value"
    assert loaded["TEST_EXISTING_KEY"] == "new_file_value"

    import os

    # Did not overwrite already existing environment variable
    assert os.environ["TEST_EXISTING_KEY"] == "already_present_value"
    # Populated new variables
    assert os.environ["TEST_NEW_KEY"] == "test_value"
    assert os.environ["TEST_QUOTED_KEY"] == "quoted_value"
    assert os.environ["TEST_INLINE_COMMENT"] == "raw_value"
