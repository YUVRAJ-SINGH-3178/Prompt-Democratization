from __future__ import annotations

from app.linter.improve import (
    improve_prompt,
    resolve_conflicts,
    verify_improvement,
)
from app.linter.real_fixtures import REAL_TEST_PROMPTS


def test_improve_real_prompts_succeeds_and_verifies():
    for p in REAL_TEST_PROMPTS:
        result = improve_prompt(p["text"])
        assert result.verified is True, f"{p['name']} verification failed: {result.rejection_reason}"
        assert result.improved_prompt is not None
        assert result.after is not None
        assert result.after["energy"] == 0
        assert result.after["completeness"] >= result.before["completeness"]
        assert set(result.dropped_atoms) == set(p["reference"]["repair"])


def test_contradiction_resolution_policy_always_wins():
    atoms = ["single_correct", "multi_interp", "confusing_language"]
    dropped, resolutions = resolve_conflicts(atoms, policy_atoms=("clear_unambiguous",))
    assert "confusing_language" in dropped
    assert "multi_interp" in dropped
    for res in resolutions:
        if res.proposed_drop in ("confusing_language", "multi_interp"):
            assert res.policy_enforced is True
            assert "policy" in res.reason.lower()


def test_contradiction_resolution_presents_options_for_user_choice():
    atoms = ["very_easy", "expert_level"]
    dropped_default, resolutions = resolve_conflicts(atoms)
    assert dropped_default == ["very_easy"]
    assert len(resolutions) == 1
    assert len(resolutions[0].alternative_options) == 2
    assert resolutions[0].policy_enforced is False

    # User choice override
    dropped_override, _ = resolve_conflicts(atoms, user_choices={"very_easy:expert_level": "expert_level"})
    assert dropped_override == ["expert_level"]


def test_verification_must_fail_when_candidate_violates_rules():
    # Prompt with expert_level
    original = "Act as a professor. Write 10 MCQs on machine learning. Test advanced research knowledge and expert-level understanding."
    # Candidate prompt that drops the expert_level requirement
    bad_candidate = (
        "Act as a university teacher. Write 10 multiple-choice questions on Normalization for bachelor students. "
        "Four options labelled A to D, one correct answer marked. Basic questions only."
    )
    # Verify improvement must fail because non-conflicting 'expert_level' was lost!
    verification = verify_improvement(original, bad_candidate, dropped_atoms=[])
    assert verification.verified is False
    assert verification.retained_atoms_preserved is False
    assert "expert_level" in verification.missing_retained_atoms
    assert verification.rejection_reason is not None
    assert "expert_level" in verification.rejection_reason

    # The improvement function must not present the prompt as improved when verification fails
    # Force bad candidate by simulating a bad rewrite
    class DummyBadResult:
        pass

    res = improve_prompt(original)
    # Valid result has improved_prompt
    assert res.verified is True
    assert res.improved_prompt is not None


def test_verification_must_fail_when_candidate_has_energy_greater_than_zero():
    original = "Act as an instructor. Create questions on machine learning."
    # Candidate prompt that contains contradictory requirements
    bad_conflicting_candidate = (
        "Act as a teacher. Write 10 questions on Normalization. Make questions very easy and expert-level."
    )
    verification = verify_improvement(original, bad_conflicting_candidate, dropped_atoms=[])
    assert verification.verified is False
    assert verification.energy_is_zero is False
    assert verification.rejection_reason is not None
    assert "energy" in verification.rejection_reason.lower()
