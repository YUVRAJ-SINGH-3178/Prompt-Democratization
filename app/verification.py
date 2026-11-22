"""Independent verification helpers; intentionally do not call production solvers."""

from __future__ import annotations

import itertools
import re

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
)
VERIFY_EDGES = (
    frozenset(("very_easy", "expert_level")),
    frozenset(("very_easy", "no_basic")),
    frozenset(("hidden_answers", "answer_after_each_q")),
    frozenset(("single_correct", "multi_interp")),
    frozenset(("clear_unambiguous", "confusing_language")),
    frozenset(("clear_unambiguous", "multi_interp")),
)
VERIFY_SLOTS = {
    "role": (r"\b(teacher|instructor|professor|faculty)\b",),
    "audience": (r"\b(bachelor|master|undergraduate|graduate|students?)\b",),
    "topic": (r"\b(course|topic|about|on)\b",),
    "count": (r"\b\d+\s+(questions?|items?)\b",),
    "n_options": (r"\b\d+\s+(options?|choices?)\b", r"\bfour options?\b"),
    "labeling": (r"\b(a\s*(,|to|-|through)\s*d|a\)\s*b\)\s*c\)\s*d\))",),
    "marking": (r"\b(correct answer|mark(ed|ing)? as correct|answer key)\b",),
    "explanation": (r"\b(explanation|explain why|rationale)\b",),
    "ordering": (r"\b(order(ed)?|progress(ively)?|difficulty|bloom)\b",),
}


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
    lowered = prompt.lower()
    result = {}
    for name, patterns in VERIFY_SLOTS.items():
        count = sum(bool(re.search(pattern, lowered)) for pattern in patterns)
        result[name] = 1.0 if count == len(patterns) else 0.5 if count else 0.0
    if re.search(r"\b(course|topic|about|on)\s*[:=]?\s*[#\w][^,.;\n]{2,}", lowered):
        result["topic"] = 1.0
    return result


def independent_metrics(prompt: str, atoms: list[str]) -> dict:
    slots = direct_slots(prompt)
    return {"slots": slots, "completeness": sum(slots.values()) / len(VERIFY_SLOTS), "energy": direct_energy(atoms)}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - margin), min(1.0, centre + margin)]
