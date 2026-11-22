from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

ATOMS = (
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
CONFLICTS = (
    ("very_easy", "expert_level"),
    ("very_easy", "no_basic"),
    ("hidden_answers", "answer_after_each_q"),
    ("single_correct", "multi_interp"),
    ("clear_unambiguous", "confusing_language"),
    ("clear_unambiguous", "multi_interp"),
)
SLOTS = ("role", "audience", "topic", "count", "n_options", "labeling", "marking", "explanation", "ordering")


@dataclass
class LintResult:
    atoms: list[str]
    completeness: float
    slot_scores: dict[str, float]
    energy: int
    repair: list[str]
    consistent_atoms: list[str]


def extract_atoms(prompt: str) -> list[str]:
    """Extract normalized requirements with negation and hyphen/spacing support."""
    t = re.sub(r"[-_]+", " ", prompt.lower())
    t = re.sub(r"\s+", " ", t)
    atoms: set[str] = set()

    def has(pattern: str) -> bool:
        return bool(re.search(pattern, t))

    if has(r"\b(very easy|beginner level|basic level)\b"):
        atoms.add("very_easy")
    if has(r"\b(expert level|advanced level|expert)\b"):
        atoms.add("expert_level")
    if has(r"\b(no basic|without basics|skip the basics|no basic questions)\b"):
        atoms.add("no_basic")
    if has(
        r"\b(hidden answers?|hide answers?|answers? (hidden|at the end|in an answer key)|do not show answers? until the end)\b"
    ):
        atoms.add("hidden_answers")
    if has(
        r"\b(answer after each question|answers? after every question|show( the)? answers? after each question|show the answer immediately)\b"
    ):
        atoms.add("answer_after_each_q")
    if has(r"\b(exactly one|single) (correct )?(answer|option)\b|\bone correct answer\b"):
        atoms.add("single_correct")
    if has(r"\b(multiple interpretations?|multiple correct|more than one correct|ambiguous)\b"):
        atoms.add("multi_interp")
    if has(r"\b(clear|unambiguous|precise)\b") and has(r"\b(question|wording|language|instruction)\b"):
        atoms.add("clear_unambiguous")
    if has(r"\b(confusing|ambiguous|unclear|vague)\b") and has(r"\b(language|wording|question|instruction)\b"):
        atoms.add("confusing_language")
    # A negated requirement is not present; no_basic itself intentionally encodes a negation.
    negated = {
        "very_easy": r"very easy|beginner level|basic level",
        "expert_level": r"expert level|advanced level|expert",
        "hidden_answers": r"hidden answers?|hide answers?|answers? at the end",
        "answer_after_each_q": r"answer after each question|answers? after every question|show( the)? answers? after each question",
        "single_correct": r"exactly one correct|single correct|one correct answer",
        "multi_interp": r"multiple interpretations?|multiple correct|more than one correct|ambiguous",
        "clear_unambiguous": r"clear|unambiguous|precise",
        "confusing_language": r"confusing|unclear|vague|ambiguous",
    }
    for atom, pattern in negated.items():
        for match in re.finditer(rf"\b(?:{pattern})\b", t):
            prefix = t[max(0, match.start() - 55) : match.start()]
            clause = re.split(r"[.!?;]", prefix)[-1]
            if re.search(r"\b(no|not|never|avoid|without|do not|don't)\b", clause) and not re.search(
                r"\bbut\b", clause
            ):
                atoms.discard(atom)
    return sorted(atoms)


def score_slots(prompt: str) -> dict[str, float]:
    t = prompt.lower()
    checks = {
        "role": (r"\b(teacher|instructor|professor|faculty)\b",),
        "audience": (r"\b(bachelor|master|undergraduate|graduate|students?)\b",),
        "topic": (r"\b(topic|course|about|on)\b",),
        "count": (r"\b\d+\s+(questions?|items?)\b",),
        "n_options": (r"\b\d+\s+(options?|choices?)\b", r"\bfour options?\b"),
        "labeling": (r"\b(a\s*(,|to|-|through)\s*d|a\)\s*b\)\s*c\)\s*d\))",),
        "marking": (r"\b(correct answer|mark(ed|ing)? as correct|answer key)\b",),
        "explanation": (r"\b(explanation|explain why|rationale)\b",),
        "ordering": (r"\b(order(ed)?|progress(ively)?|difficulty|bloom)\b",),
    }
    out: dict[str, float] = {}
    for slot, patterns in checks.items():
        n = sum(bool(re.search(p, t)) for p in patterns)
        out[slot] = 1.0 if n == len(patterns) else (0.5 if n else 0.0)
    # A topic slot requires an identified subject phrase, not just the word 'course'.
    if re.search(r"\b(course|topic|about|on)\s*[:=]?\s*[#\w][^,.;\n]{2,}", t):
        out["topic"] = 1.0
    return out


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
    repair_priority = {"very_easy": 0, "hidden_answers": 1, "confusing_language": 2, "multi_interp": 3}
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
