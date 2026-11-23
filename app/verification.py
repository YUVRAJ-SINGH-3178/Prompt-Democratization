"""Independent verification helpers; intentionally do not call production solvers."""

from __future__ import annotations

import itertools

VERIFY_ATOMS = (
    "very_easy",
    "expert_level",
    "no_basic",
    "hidden_answers",
    "answer_after_each_q",
    "single_correct",
    "multi_interp",
    "clear_unambiguous",
    "confusing_language",
    "ordered_easy_to_hard",
)
VERIFY_EDGES = (
    frozenset(("very_easy", "expert_level")),
    frozenset(("very_easy", "no_basic")),
    frozenset(("hidden_answers", "answer_after_each_q")),
    frozenset(("single_correct", "multi_interp")),
    frozenset(("clear_unambiguous", "confusing_language")),
    frozenset(("clear_unambiguous", "multi_interp")),
)


def direct_energy(atoms: list[str]) -> int:
    active = set(atoms) | {"clear_unambiguous"}
    ordered = tuple(sorted(active - {"clear_unambiguous"}))
    for remove_count in range(len(ordered) + 1):
        for removed in itertools.combinations(ordered, remove_count):
            kept = active.difference(removed)
            if all(not edge.issubset(kept) for edge in VERIFY_EDGES):
                return remove_count
    raise AssertionError("The full removal set must be consistent")


def direct_slots(prompt: str) -> dict[str, float]:
    from app.linter.lexicon import score_slots_from_text

    return score_slots_from_text(prompt)


def independent_metrics(prompt: str, atoms: list[str]) -> dict:
    slots = direct_slots(prompt)
    return {"slots": slots, "completeness": sum(slots.values()) / len(slots), "energy": direct_energy(atoms)}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - margin), min(1.0, centre + margin)]
