"""Energy module: conflict energy, completeness deficits, and feature vectors.

Conflict energy is the exact minimum number of atoms to drop so that no
conflict pair and no policy conflict remains.  For nine atoms this is
done by enumerating subsets in increasing size.
"""

from __future__ import annotations

import itertools
from typing import Any

from app.linter.lexicon import CONFLICTS, SLOTS

POLICY_ATOMS: tuple[str, ...] = ("clear_unambiguous",)

FEATURE_NAMES: tuple[str, ...] = (
    "conflict_energy",
    "deficit_role",
    "deficit_audience",
    "deficit_topic",
    "deficit_count",
    "deficit_n_options",
    "deficit_labeling",
    "deficit_marking",
    "deficit_explanation",
    "deficit_ordering",
)


def conflict_energy(
    atoms: list[str],
    policy_atoms: tuple[str, ...] = POLICY_ATOMS,
    conflicts: tuple[tuple[str, str], ...] = CONFLICTS,
) -> dict[str, Any]:
    """Return the minimum number of atoms to drop for a conflict-free set.

    Returns dict with keys: energy, dropped, kept, conflicting_pairs.
    """
    active = sorted(set(atoms) | set(policy_atoms))
    edges = [frozenset(e) for e in conflicts]
    fixed_policy = set(policy_atoms)

    # Identify which pairs actually conflict in the active set
    conflicting_pairs: list[list[str]] = []
    for a, b in conflicts:
        if a in active and b in active:
            conflicting_pairs.append(sorted([a, b]))

    if not conflicting_pairs:
        return {
            "energy": 0,
            "dropped": [],
            "kept": active,
            "conflicting_pairs": [],
        }

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
        "answer_after_each_q": 8,
    }

    # Enumerate subsets in increasing size to find the minimum drop set
    best: tuple[int, list[str]] | None = None
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

    if best is not None:
        return {
            "energy": best[0],
            "dropped": best[1],
            "kept": sorted(set(active) - set(best[1])),
            "conflicting_pairs": conflicting_pairs,
        }

    return {
        "energy": len(droppable),
        "dropped": sorted(droppable),
        "kept": sorted(fixed_policy),
        "conflicting_pairs": conflicting_pairs,
    }


def completeness_deficits(slot_scores: dict[str, float]) -> dict[str, float]:
    """Return per-slot deficit: 1 minus the score."""
    return {slot: round(1.0 - slot_scores.get(slot, 0.0), 4) for slot in SLOTS}


def feature_vector(
    atoms: list[str],
    slot_scores: dict[str, float],
    policy_atoms: tuple[str, ...] = POLICY_ATOMS,
) -> dict[str, Any]:
    """Return the length-10 feature vector: conflict energy + 9 deficits."""
    ce = conflict_energy(atoms, policy_atoms)
    deficits = completeness_deficits(slot_scores)
    values = [float(ce["energy"])] + [deficits[slot] for slot in SLOTS]
    named = dict(zip(FEATURE_NAMES, values))
    return {
        "values": values,
        "names": list(FEATURE_NAMES),
        "named": named,
        "conflict_energy_detail": ce,
    }


def feature_vector_from_text(prompt: str) -> dict[str, Any]:
    """Extract atoms and slots from text, then return the feature vector."""
    from app.linter.core import evaluate

    result = evaluate(prompt)
    return feature_vector(result.atoms, result.slot_scores)
