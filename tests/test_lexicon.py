from __future__ import annotations

import re
import pytest
from app.linter.lexicon import (
    ATOM_REGISTRY,
    SLOT_REGISTRY,
    extract_atoms_from_text,
    has_clause_negation,
    score_slots_from_text,
    split_clauses,
)


@pytest.mark.parametrize("atom_name,spec", list(ATOM_REGISTRY.items()))
def test_atom_patterns_positive_and_negative(atom_name: str, spec):
    """Every lexicon pattern must have at least one verified positive and one negative test sentence."""
    pos_atoms = extract_atoms_from_text(spec.positive_example)
    assert atom_name in pos_atoms, f"Positive example for {atom_name} failed: '{spec.positive_example}'"

    neg_atoms = extract_atoms_from_text(spec.negative_example)
    assert atom_name not in neg_atoms, f"Negative example for {atom_name} falsely triggered: '{spec.negative_example}'"


@pytest.mark.parametrize("slot_name,spec", list(SLOT_REGISTRY.items()))
def test_slot_patterns_positive_and_negative(slot_name: str, spec):
    """Every slot pattern must score 1.0 on positive example and < 1.0 (or 0.0) on negative example."""
    pos_slots = score_slots_from_text(spec.positive_example)
    assert pos_slots[slot_name] == 1.0, f"Positive example for {slot_name} did not score 1.0: '{spec.positive_example}'"

    neg_slots = score_slots_from_text(spec.negative_example)
    assert neg_slots[slot_name] < 1.0, f"Negative example for {slot_name} falsely scored 1.0: '{spec.negative_example}'"


def test_clause_level_negation_scope_does_not_spill():
    """Negation in one clause must not cancel a requirement in a subsequent contrasting clause."""
    mixed = "Do not make the questions very easy, but ensure they test expert-level understanding."
    atoms = extract_atoms_from_text(mixed)
    assert "very_easy" not in atoms
    assert "expert_level" in atoms


def test_expert_alone_does_not_trigger_expert_level():
    """'Act as an expert professor' is a role persona, not a difficulty requirement."""
    text = "Act as an expert professor and prepare MCQs on computer science."
    atoms = extract_atoms_from_text(text)
    assert "expert_level" not in atoms
    slots = score_slots_from_text(text)
    assert slots["role"] == 1.0


def test_count_variations_recognized():
    assert score_slots_from_text("Create 20 MCQs on machine learning.")["count"] == 1.0
    assert score_slots_from_text("Generate 15 MCQs for students.")["count"] == 1.0
    assert score_slots_from_text("Create 25 MCQs on operating systems.")["count"] == 1.0
    assert score_slots_from_text("Create 10 multiple-choice questions on Normalization.")["count"] == 1.0
    assert score_slots_from_text("Create some multiple-choice questions.")["count"] == 0.0


def test_n_options_score_full_credit():
    assert score_slots_from_text("Give four options for each question.")["n_options"] == 1.0
    assert score_slots_from_text("Provide four choices for every question.")["n_options"] == 1.0
    assert score_slots_from_text("Each question must have four answer choices.")["n_options"] == 1.0
    assert score_slots_from_text("Provide multiple choices for each question.")["n_options"] == 0.5


def test_labeling_variations_recognized():
    assert score_slots_from_text("Label each answer choice (A, B, C, D).")["labeling"] == 1.0
    assert score_slots_from_text("Each question should have four options labelled A, B, C, and D.")["labeling"] == 1.0
    assert score_slots_from_text("Provide questions without letter labels.")["labeling"] == 0.0


def test_marking_variations_recognized():
    assert score_slots_from_text("Give four options and provide the answers.")["marking"] == 1.0
    assert score_slots_from_text("Provide four choices and mark the right answer.")["marking"] == 1.0
    assert score_slots_from_text("Highlight the correct answer in bold.")["marking"] == 1.0
    assert score_slots_from_text("Give the correct answer and explain why other options are wrong.")["marking"] == 1.0


def test_ordering_variations_recognized():
    assert score_slots_from_text("Arrange all questions from the easiest to the hardest.")["ordering"] == 1.0
    assert score_slots_from_text("Organize the questions in order of increasing difficulty.")["ordering"] == 1.0
    assert score_slots_from_text("Arrange them properly.")["ordering"] == 0.5
    assert score_slots_from_text("No ordering instruction provided.")["ordering"] == 0.0


def test_clause_splitting_handles_semicolons_and_conjunctions():
    s = "Highlight the correct answer in bold, but do not reveal the correct answer to students."
    clauses = split_clauses(s)
    assert len(clauses) >= 2
    match = re.search(r"do\s+not\s+reveal", clauses[1])
    assert match is not None
    assert has_clause_negation(clauses[1], match.start()) is False  # 'do not' starts at 0, no prior negation
