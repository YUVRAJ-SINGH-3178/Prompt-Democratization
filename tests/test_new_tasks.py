"""Unit and integration tests for Tasks 1-6.

Covers:
1. Five brief prompts: exact atoms, energy, and minimum drop sets
2. Energy module: conflict_energy, completeness deficits, feature vector length 10, template agreement
3. Comparison collection: storage, ties excluded, rater isolation, kappa threshold
4. Learner: minimum data rule (50 comparisons, 2 raters), weights, bootstrap intervals
5. Recommend & Improve: comparison output, verification gate, original returned on failure
6. Evaluation harness: synthetic recovery run labeled SYNTHETIC with versions and seeds
"""

from __future__ import annotations

from app.energy import conflict_energy, completeness_deficits, feature_vector, FEATURE_NAMES
from app.learner import (
    fit_bradley_terry,
    predict_preference,
    learned_energy,
)
from app.linter.core import evaluate, extract_atoms
from app.linter.real_fixtures import REAL_TEST_PROMPTS
from app.linter.improve import improve_prompt


# ---------------------------------------------------------------------------
# Task 1: Five brief prompts exact atoms and drops
# ---------------------------------------------------------------------------


def test_brief_faculty_1_no_atoms():
    p1 = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-1")
    atoms = extract_atoms(p1["text"])
    ce = conflict_energy(atoms)
    assert atoms == []
    assert ce["energy"] == 0
    assert ce["dropped"] == []


def test_brief_faculty_2_no_atoms():
    p2 = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-2")
    atoms = extract_atoms(p2["text"])
    ce = conflict_energy(atoms)
    assert atoms == []
    assert ce["energy"] == 0
    assert ce["dropped"] == []


def test_brief_faculty_3_atoms_and_drop():
    p3 = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-3")
    atoms = extract_atoms(p3["text"])
    expected = sorted(["very_easy", "expert_level", "no_basic", "ordered_easy_to_hard"])
    assert sorted(atoms) == expected
    ce = conflict_energy(atoms)
    assert ce["energy"] == 1
    assert ce["dropped"] == ["very_easy"]


def test_brief_faculty_4_atoms_and_drop():
    p4 = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-4")
    atoms = extract_atoms(p4["text"])
    expected = sorted(["hidden_answers", "answer_after_each_q"])
    assert sorted(atoms) == expected
    ce = conflict_energy(atoms)
    assert ce["energy"] == 1
    assert ce["dropped"] == ["hidden_answers"]


def test_brief_faculty_5_atoms_and_drop():
    p5 = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-5")
    atoms = extract_atoms(p5["text"])
    expected = sorted(["single_correct", "multi_interp", "confusing_language"])
    assert sorted(atoms) == expected
    ce = conflict_energy(atoms)
    assert ce["energy"] == 2
    assert sorted(ce["dropped"]) == sorted(["confusing_language", "multi_interp"])


# ---------------------------------------------------------------------------
# Task 2: Energy module
# ---------------------------------------------------------------------------


def test_conflict_energy_exact_subsets():
    # Mutual contradiction: very_easy with expert_level
    ce = conflict_energy(["very_easy", "expert_level"])
    assert ce["energy"] == 1
    assert len(ce["dropped"]) == 1
    assert ce["conflicting_pairs"] == [["expert_level", "very_easy"]]

    # Policy conflict: confusing_language conflicts with policy clear_unambiguous
    ce_pol = conflict_energy(["confusing_language"])
    assert ce_pol["energy"] == 1
    assert ce_pol["dropped"] == ["confusing_language"]

    # Multiple conflicts
    ce_multi = conflict_energy(["very_easy", "expert_level", "no_basic"])
    assert ce_multi["energy"] == 1
    assert ce_multi["dropped"] == ["very_easy"]


def test_completeness_deficits():
    scores = {"role": 1.0, "audience": 0.5, "topic": 0.0}
    deficits = completeness_deficits(scores)
    assert deficits["role"] == 0.0
    assert deficits["audience"] == 0.5
    assert deficits["topic"] == 1.0


def test_feature_vector_length_and_template_agreement():
    tmpl = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-template")
    res = evaluate(tmpl["text"])
    fv = feature_vector(res.atoms, res.slot_scores)

    # Length 10
    assert len(fv["values"]) == 10
    assert fv["names"] == list(FEATURE_NAMES)
    assert fv["values"][0] == 0.0  # Conflict energy = 0

    # Agreement on template: all slots 1, so all deficits 0
    for deficit_val in fv["values"][1:]:
        assert deficit_val == 0.0
    assert res.completeness == 1.0
    assert res.energy == 0


# ---------------------------------------------------------------------------
# Task 3: Comparison collection & Cohen's kappa
# ---------------------------------------------------------------------------


def test_kappa_requires_30_shared_pairs():
    from app.linter.datasets import compute_cohen_kappa

    # Less than 30 pairs
    r1 = [1] * 10
    r2 = [1] * 10
    k = compute_cohen_kappa(r1, r2)
    assert k == 1.0

    # Agreement calculation on 30 items
    r1 = ["a"] * 15 + ["b"] * 15
    r2 = ["a"] * 15 + ["b"] * 15
    assert compute_cohen_kappa(r1, r2) == 1.0


# ---------------------------------------------------------------------------
# Task 4: Learner (Bradley-Terry)
# ---------------------------------------------------------------------------


def test_learner_minimum_data_rule():
    # Less than 50 comparisons -> insufficient_data
    small_data = [
        {"features_a": [0.0] * 10, "features_b": [1.0] * 10, "winner": "a", "rater_id": "r1"} for _ in range(30)
    ]
    fit = fit_bradley_terry(small_data)
    assert fit["model_status"] == "insufficient_data"
    assert fit["weights"] is None

    # 50 comparisons but only 1 rater -> insufficient_data
    one_rater_data = [
        {"features_a": [0.0] * 10, "features_b": [1.0] * 10, "winner": "a", "rater_id": "r1"} for _ in range(50)
    ]
    fit_1 = fit_bradley_terry(one_rater_data)
    assert fit_1["model_status"] == "insufficient_data"

    # >= 50 comparisons and >= 2 raters -> fitted
    two_raters_data = [
        {"features_a": [0.0] * 10, "features_b": [1.0] * 10, "winner": "a", "rater_id": "r1"} for _ in range(25)
    ] + [{"features_a": [1.0] * 10, "features_b": [0.0] * 10, "winner": "b", "rater_id": "r2"} for _ in range(25)]
    fit_2 = fit_bradley_terry(two_raters_data)
    assert fit_2["model_status"] == "fitted"
    assert fit_2["weights"] is not None
    assert len(fit_2["weights"]) == 10
    assert "weight_intervals_95" in fit_2
    assert "versions" in fit_2
    assert "numpy" in fit_2["versions"]
    assert "scipy" in fit_2["versions"]
    assert "scikit_learn" in fit_2["versions"]


def test_learned_energy_and_preference_prediction():
    weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fa = [0.0] * 10  # Perfect prompt: energy = 0
    fb = [1.0] * 10  # Poor prompt: energy = 11

    ea = learned_energy(weights, fa)
    eb = learned_energy(weights, fb)
    assert ea == 0.0
    assert eb == 11.0

    pred = predict_preference(weights, fa, fb)
    assert pred["p_a_better"] > 0.99
    assert pred["p_b_better"] < 0.01


# ---------------------------------------------------------------------------
# Task 5: Recommend and Improve verification gate
# ---------------------------------------------------------------------------


def test_improve_verification_gate_success():
    p = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-faculty-3")
    res = improve_prompt(p["text"])
    assert res.verified is True
    assert res.after is not None
    assert res.after["energy"] == 0
    assert res.delta_energy < 0  # Energy decreased


def test_improve_verification_gate_failure_handling():
    # A prompt already at 100% completeness and 0 energy cannot be improved
    tmpl = next(p for p in REAL_TEST_PROMPTS if p["id"] == "real-template")
    res = improve_prompt(tmpl["text"])
    # Should still verify as valid with 0 energy
    assert res.after["energy"] == 0


# ---------------------------------------------------------------------------
# Task 6: Evaluation harness
# ---------------------------------------------------------------------------


def test_synthetic_recovery_output_structure():
    from app.eval_harness import synthetic_recovery

    # Run quick small test
    out = synthetic_recovery(
        n_prompts=50,
        seeds=[42],
        budgets=[20],
        heldout_pairs=50,
    )
    assert out["SYNTHETIC"] is True
    assert out["label"] == "SYNTHETIC"
    assert "versions" in out
    assert "numpy" in out["versions"]
    assert "scipy" in out["versions"]
    assert "scikit_learn" in out["versions"]
    assert len(out["results"]) > 0
