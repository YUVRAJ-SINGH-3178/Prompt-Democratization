from __future__ import annotations

import itertools
from dataclasses import dataclass

from app.linter.lexicon import (
    ATOMS,
    CONFLICTS,
    SLOTS,
    extract_atoms_from_text,
    score_slots_from_text,
)


@dataclass
class LintResult:
    atoms: list[str]
    completeness: float
    slot_scores: dict[str, float]
    energy: int
    repair: list[str]
    consistent_atoms: list[str]


def extract_atoms(prompt: str) -> list[str]:
    """Extract normalized requirement atoms using clause-level negation and lexical rules."""
    return extract_atoms_from_text(prompt)


def score_slots(prompt: str) -> dict[str, float]:
    """Score the nine governed slots using lexical full- and partial-credit patterns."""
    return score_slots_from_text(prompt)


def evaluate(
    prompt: str,
    atoms: list[str] | None = None,
    conflicts: tuple[tuple[str, str], ...] | None = None,
    policy_atoms: tuple[str, ...] | None = None,
) -> LintResult:
    found = sorted(set(atoms if atoms is not None else extract_atoms(prompt)))
    if any(a not in ATOMS for a in found):
        raise ValueError("Unknown requirement atom")
    # Repository policy includes clear_unambiguous unless an operator supplies a policy set.
    fixed_policy = set(policy_atoms if policy_atoms is not None else ("clear_unambiguous",))
    if fixed_policy - set(ATOMS):
        raise ValueError("Unknown policy atom")
    active = sorted(set(found) | fixed_policy)
    edges = {frozenset(e) for e in (conflicts if conflicts is not None else CONFLICTS)}
    best: tuple[int, list[str]] | None = None
    # The repository policy atom is fixed and cannot be removed as a repair.
    droppable = [a for a in active if a not in fixed_policy]
    repair_priority = {
        "very_easy": 0,
        "hidden_answers": 1,
        "confusing_language": 2,
        "multi_interp": 3,
        "ordered_easy_to_hard": 4,
        "no_basic": 5,
        "single_correct": 6,
        "expert_level": 7,
    }
    # Exact minimum drop set for the small atom vocabulary.
    for k in range(len(droppable) + 1):
        for dropped in itertools.combinations(droppable, k):
            kept = set(active) - set(dropped)
            if not any(edge <= kept for edge in edges):
                ordered_drop = sorted(dropped, key=lambda a: (repair_priority.get(a, 10), a))
                candidate = (k, ordered_drop)
                if best is None or tuple(repair_priority.get(a, 10) for a in candidate[1]) < tuple(
                    repair_priority.get(a, 10) for a in best[1]
                ):
                    best = candidate
        if best is not None:
            break
    assert best is not None
    slots = score_slots(prompt)
    return LintResult(
        found, sum(slots.values()) / len(SLOTS), slots, best[0], best[1], sorted(set(active) - set(best[1]))
    )
