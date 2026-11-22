from itertools import combinations
from app.linter.core import ATOMS, CONFLICTS, evaluate, extract_atoms


def test_hyphens_spacing_and_negative_variants():
    assert "expert_level" in extract_atoms("Create expert-level questions")
    assert "hidden_answers" in extract_atoms("Keep the answers hidden")
    assert "very_easy" not in extract_atoms("Do not make questions very easy")
    assert "confusing_language" not in extract_atoms("Avoid confusing language")
    assert "multi_interp" not in extract_atoms("Do not allow multiple interpretations")
    assert "answer_after_each_q" not in extract_atoms("Do not show answers after each question")
    assert "expert_level" not in extract_atoms("Never use an expert level")


def test_reference_conflict_repairs_from_annotated_atoms():
    cases = [
        ([], 0, []),
        (["very_easy", "expert_level"], 1, ["very_easy"]),
        (["hidden_answers", "answer_after_each_q"], 1, ["hidden_answers"]),
        (["confusing_language", "multi_interp"], 2, ["confusing_language", "multi_interp"]),
    ]
    for atoms, energy, repair in cases:
        result = evaluate("", atoms)
        assert (result.energy, result.repair) == (energy, repair)


def test_faculty_fixture_rule_phrases_are_recognized():
    f3 = evaluate("Make very easy expert-level questions for an advanced course.")
    f4 = evaluate("Keep the answers hidden and show the answer after each question.")
    f5 = evaluate("The question wording is confusing and allows multiple interpretations.")
    assert {"very_easy", "expert_level"} <= set(f3.atoms)
    assert (f3.energy, f3.repair) == (1, ["very_easy"])
    assert {"hidden_answers", "answer_after_each_q"} <= set(f4.atoms)
    assert (f4.energy, f4.repair) == (1, ["hidden_answers"])
    assert {"confusing_language", "multi_interp"} <= set(f5.atoms)
    assert (f5.energy, f5.repair) == (2, ["confusing_language", "multi_interp"])


def test_completeness_slots_have_nine_values():
    result = evaluate(
        "Write 5 questions for bachelor students about biology, with four options A to D, one correct answer and explanations ordered by difficulty."
    )
    assert len(result.slot_scores) == 9
    assert result.completeness == sum(result.slot_scores.values()) / 9


def test_energy_zero_means_full_atom_set_is_consistent():
    for size in range(len(ATOMS) + 1):
        for subset in combinations(ATOMS, size):
            result = evaluate("", list(subset))
            if result.energy == 0:
                kept = set(result.atoms) | {"clear_unambiguous"}
                assert not any(set(edge) <= kept for edge in CONFLICTS)
                assert result.consistent_atoms == sorted(kept)
