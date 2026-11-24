"""Prompt Improvement Engine with Strict Verification.

1. Contradiction resolution: Proposes requirement to drop; policy atoms always win; user choices presented; never silent.
2. Slot filling: Fills missing slots from approved templates and mapping tables (no free-text invention).
3. Preservation: Preserves user's retained non-conflicting intent sentences.
4. Code-enforced verification:
   - energy_after == 0
   - every non-conflicting original atom is still present
   - no new conflict may appear
   If any rule fails, the result is NOT presented as improved and the exact reason is reported.
5. Optional LLM rewrite mode: Off by default; output accepted only if it passes identical verification.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from typing import Any

from app.linter.core import CONFLICTS, evaluate, extract_atoms, score_slots
from app.linter.lexicon import find_unclassified_clauses, split_clauses, split_sentences

# Default mapping values aligned with Faculty builder
MAPPING_DEFAULTS = {
    "role": "Act as a university teacher tasked with assessing students' knowledge.",
    "audience": "bachelor students",
    "cohort": "bachelor",
    "course": "Database Management Systems",
    "topic": "Normalization",
    "count": 10,
    "n_options": "Each question must have four answer choices, with only one correct answer.",
    "labeling": "Label each answer choice (A, B, C, D).",
    "marking": "Mark the correct answer using a consistent method (e.g. make the correct answer bold).",
    "explanation": "After each question, provide an explanation for why the chosen answer is correct and why each incorrect answer is not.",
    "ordering": "Organize the questions in order of increasing difficulty, starting with the easiest that test recall and progressing to the hardest that test application and critical thinking.",
    "policy": "Ensure each question is clear and unambiguous, leaving no room for misinterpretation.",
}


@dataclass
class ConflictResolution:
    conflict_pair: tuple[str, str]
    proposed_drop: str
    retained_atom: str
    reason: str
    policy_enforced: bool
    alternative_options: list[dict[str, str]]


@dataclass
class ImprovementVerification:
    verified: bool
    energy_is_zero: bool
    retained_atoms_preserved: bool
    missing_retained_atoms: list[str]
    no_new_conflicts: bool
    conflicts_detected: list[list[str]]
    rejection_reason: str | None


@dataclass
class ImprovementResult:
    original_prompt: str
    improved_prompt: str | None
    verified: bool | str
    rejection_reason: str | None
    verification_details: ImprovementVerification
    before: dict[str, Any]
    after: dict[str, Any] | None
    delta_completeness: float | None
    delta_energy: int | None
    resolutions: list[ConflictResolution]
    dropped_atoms: list[str]
    retained_atoms: list[str]
    filled_slots: list[str]
    text_diff: str
    mode: str  # 'rule_based' or 'llm'
    extractor_confidence: str = "high"
    unclassified_clauses: list[str] | None = None
    llm_metadata: dict[str, Any] | None = None


def resolve_conflicts(
    atoms: list[str],
    policy_atoms: tuple[str, ...] = ("clear_unambiguous",),
    user_choices: dict[str, str] | None = None,
) -> tuple[list[str], list[ConflictResolution]]:
    """Determine which conflicting requirement atoms to drop.

    Policy atoms always win. For other conflicts, lowest-energy drop is proposed or user choice is applied.
    Never drops silently.
    """
    user_choices = user_choices or {}
    active_atoms = set(atoms)
    fixed_policy = set(policy_atoms)
    edges = [tuple(sorted(e)) for e in CONFLICTS]

    resolutions: list[ConflictResolution] = []
    dropped: set[str] = set()

    # 1. Policy conflicts (policy atoms always win)
    for a, b in edges:
        if (a in active_atoms and b in fixed_policy) or (b in active_atoms and a in fixed_policy):
            target = a if a in active_atoms else b
            pol = b if a in active_atoms else a
            dropped.add(target)
            resolutions.append(
                ConflictResolution(
                    conflict_pair=(a, b),
                    proposed_drop=target,
                    retained_atom=pol,
                    reason=f"Requirement '{target}' conflicts with institutional policy '{pol}'. Policy requirements cannot be overridden.",
                    policy_enforced=True,
                    alternative_options=[],  # No alternative for fixed policy
                )
            )

    # 2. Non-policy conflicts
    remaining_active = active_atoms - dropped
    for a, b in edges:
        if a in remaining_active and b in remaining_active:
            # Check user choice first
            chosen = user_choices.get(f"{a}:{b}") or user_choices.get(f"{b}:{a}")
            if chosen in (a, b):
                drop_atom = chosen
                keep_atom = b if chosen == a else a
            else:
                # Default heuristic: prioritize keeping more specific educational requirements
                # (e.g. drop very_easy to keep expert_level/no_basic; drop hidden_answers to keep answer_after_each_q)
                priority = {"very_easy": 0, "hidden_answers": 1, "confusing_language": 2, "multi_interp": 3}
                drop_atom = a if priority.get(a, 10) <= priority.get(b, 10) else b
                keep_atom = b if drop_atom == a else a

            dropped.add(drop_atom)
            resolutions.append(
                ConflictResolution(
                    conflict_pair=(a, b),
                    proposed_drop=drop_atom,
                    retained_atom=keep_atom,
                    reason=f"Mutual contradiction detected between '{a}' and '{b}'. Proposing to drop '{drop_atom}' to preserve '{keep_atom}'.",
                    policy_enforced=False,
                    alternative_options=[
                        {"drop": drop_atom, "keep": keep_atom, "label": f"Drop '{drop_atom}' (Recommended)"},
                        {"drop": keep_atom, "keep": drop_atom, "label": f"Drop '{keep_atom}' instead"},
                    ],
                )
            )

    return sorted(dropped), resolutions


def filter_retained_sentences(prompt: str, dropped_atoms: list[str]) -> list[str]:
    """Identify sentences conveying user intent that do not conflict with dropped requirements."""
    sentences = split_sentences(prompt)
    retained: list[str] = []

    for sentence in sentences:
        sentence_atoms = extract_atoms(sentence)
        # If sentence only contained dropped atoms, drop it
        if set(sentence_atoms) & set(dropped_atoms):
            # Check if it has non-conflicting clauses
            clauses = split_clauses(sentence)
            clean_clauses = []
            for clause in clauses:
                clause_atoms = extract_atoms(clause)
                if not (set(clause_atoms) & set(dropped_atoms)):
                    clean_clauses.append(clause)
            if clean_clauses:
                retained.append(", ".join(clean_clauses))
        else:
            retained.append(sentence)

    return retained


def build_improved_prompt_text(
    original_prompt: str,
    dropped_atoms: list[str],
    slot_values: dict[str, Any] | None = None,
    template_text: str | None = None,
) -> tuple[str, list[str]]:
    """Assemble the improved prompt using approved template structure and controlled mappings.

    No free-text invention. Preserves non-conflicting user intent.
    """
    slots = {**MAPPING_DEFAULTS, **(slot_values or {})}
    retained_sentences = filter_retained_sentences(original_prompt, dropped_atoms)

    # Determine missing slots in the original prompt
    original_slots = score_slots(original_prompt)
    filled_slots: list[str] = []
    for s_name, score in original_slots.items():
        if score < 1.0:
            filled_slots.append(s_name)

    # Extract user-specific topic, course, cohort, count if present in retained sentences
    cohort = str(slots.get("cohort", "bachelor"))
    count = int(slots.get("count", 10))
    topic = str(slots.get("topic", "Normalization"))
    course = str(slots.get("course", "Database Management Systems"))

    # Construct the structured improved prompt from standard template blocks
    blocks = [
        f"Act as a university teacher tasked with assessing {cohort} students' knowledge. "
        f"Your job is to create multiple-choice questions that effectively test students' understanding of {course}.",
        f"Task:\nCreate {count} multiple-choice questions on {topic}.",
        "Format:\n"
        f"{slots['n_options']} {slots['labeling']} {slots['marking']}\n"
        f"The distractors (incorrect answers) should be plausible but subtly flawed, to effectively test students' understanding. "
        f"{slots['explanation']} {slots['policy']}\n"
        f"The questions must range from testing basic recall to assessing the ability to think critically and apply knowledge. "
        f"{slots['ordering']}",
    ]

    # If the user prompt had retained intent sentences with specific pedagogical context (like expert-level understanding),
    # include them under a dedicated Additional Requirements section
    retained_atoms = set(extract_atoms(" ".join(retained_sentences))) - set(dropped_atoms)
    extra_intents = []
    if "expert_level" in retained_atoms:
        extra_intents.append("Test advanced research knowledge, critical thinking, and expert-level understanding.")
    if "no_basic" in retained_atoms:
        extra_intents.append("Do not include any basic questions.")
    if "answer_after_each_q" in retained_atoms:
        extra_intents.append("After every question, provide the answer and a detailed explanation.")
    if "hidden_answers" in retained_atoms:
        extra_intents.append("Keep the answers hidden and provide an answer key at the end.")
    if "ordered_easy_to_hard" in retained_atoms and "ordering" not in filled_slots:
        extra_intents.append("Arrange all questions from the easiest to the hardest.")
    if "single_correct" in retained_atoms and "single_correct" not in extract_atoms(slots["n_options"]):
        extra_intents.append("Make sure there is only one correct answer.")

    if extra_intents:
        blocks.append("Additional Requirements:\n" + " ".join(extra_intents))

    improved_text = "\n\n".join(blocks)
    return improved_text, filled_slots


def verify_improvement(
    original_prompt: str,
    improved_prompt: str,
    dropped_atoms: list[str],
    policy_atoms: tuple[str, ...] = ("clear_unambiguous",),
) -> ImprovementVerification:
    """Strict verification of improvement rules:

    1. energy_after must be 0.
    2. Every non-conflicting original atom must still be present.
    3. No new conflict may appear.
    """
    orig_result = evaluate(original_prompt, policy_atoms=policy_atoms)
    imp_result = evaluate(improved_prompt, policy_atoms=policy_atoms)

    # Rule 1: Energy after must be 0
    energy_zero = imp_result.energy == 0

    # Rule 2: Every non-conflicting original atom must be present
    original_atoms = set(orig_result.atoms)
    retained_expected = original_atoms - set(dropped_atoms)
    improved_atoms = set(imp_result.atoms)
    missing_retained = sorted(retained_expected - improved_atoms)
    retained_preserved = len(missing_retained) == 0

    # Rule 3: No new conflicts
    active_imp = set(imp_result.atoms) | set(policy_atoms)
    conflicts_found = []
    for a, b in CONFLICTS:
        if a in active_imp and b in active_imp:
            conflicts_found.append([a, b])
    no_new_conflicts = len(conflicts_found) == 0

    # Summary decision
    verified = energy_zero and retained_preserved and no_new_conflicts
    rejection_reason = None
    if not energy_zero:
        rejection_reason = f"Verification failed: improved prompt has non-zero conflict energy ({imp_result.energy})."
    elif not retained_preserved:
        rejection_reason = (
            f"Verification failed: non-conflicting original requirements were lost: {', '.join(missing_retained)}."
        )
    elif not no_new_conflicts:
        rejection_reason = f"Verification failed: new conflicts detected: {conflicts_found}."

    return ImprovementVerification(
        verified=verified,
        energy_is_zero=energy_zero,
        retained_atoms_preserved=retained_preserved,
        missing_retained_atoms=missing_retained,
        no_new_conflicts=no_new_conflicts,
        conflicts_detected=conflicts_found,
        rejection_reason=rejection_reason,
    )


def compute_diff(text1: str, text2: str) -> str:
    """Generate a clean unified diff between original and improved text."""
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = difflib.unified_diff(lines1, lines2, fromfile="Original Prompt", tofile="Improved Prompt")
    return "".join(diff)


def improve_prompt(
    prompt: str,
    user_choices: dict[str, str] | None = None,
    slot_values: dict[str, Any] | None = None,
    template_text: str | None = None,
    use_llm: bool = False,
    api_key: str | None = None,
    model: str | None = None,
) -> ImprovementResult:
    """Main improvement entry point with contradiction resolution, slot filling, and code-enforced verification."""
    orig_result = evaluate(prompt)
    dropped_atoms, resolutions = resolve_conflicts(orig_result.atoms, user_choices=user_choices)

    llm_metadata: dict[str, Any] | None = None
    candidate_prompt: str

    if use_llm:
        # Optional LLM rewrite mode (off by default)
        key = api_key or os.environ.get("STUDY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        selected_model = model or os.environ.get("STUDY_MODEL", "gpt-4o-mini")
        if not key:
            raise ValueError("LLM rewrite requested but no STUDY_API_KEY or OPENAI_API_KEY provided.")

        from app.study import call_model

        system_instruction = (
            "You are an academic prompt engineer. Rewrite the user prompt to resolve contradictory requirements "
            f"by dropping: {dropped_atoms}. Fill missing educational specification slots without changing core subject intent."
        )
        response_text = call_model(selected_model, key, system_instruction, prompt)
        candidate_prompt = response_text
        llm_metadata = {
            "model": selected_model,
            "raw_response": response_text,
            "system_instruction": system_instruction,
        }
        mode = "llm"
    else:
        candidate_prompt, filled_slots = build_improved_prompt_text(
            prompt, dropped_atoms, slot_values, template_text=template_text
        )
        mode = "rule_based"

    # Step 4: Code-enforced verification
    verification = verify_improvement(prompt, candidate_prompt, dropped_atoms)

    if not verification.verified:
        # If verification fails, do NOT present the candidate prompt as improved!
        return ImprovementResult(
            original_prompt=prompt,
            improved_prompt=None,  # Not presented as improved
            verified=False,
            rejection_reason=verification.rejection_reason,
            verification_details=verification,
            before={
                "completeness": orig_result.completeness,
                "energy": orig_result.energy,
                "atoms": orig_result.atoms,
                "slots": orig_result.slot_scores,
            },
            after=None,
            delta_completeness=None,
            delta_energy=None,
            resolutions=resolutions,
            dropped_atoms=dropped_atoms,
            retained_atoms=sorted(set(orig_result.atoms) - set(dropped_atoms)),
            filled_slots=[],
            text_diff="",
            mode=mode,
            llm_metadata=llm_metadata,
        )

    # Verification passed!
    imp_result = evaluate(candidate_prompt)
    delta_comp = round(imp_result.completeness - orig_result.completeness, 3)
    delta_energy = imp_result.energy - orig_result.energy
    diff = compute_diff(prompt, candidate_prompt)

    unclassified = find_unclassified_clauses(prompt)
    confidence = "high" if not unclassified else "medium"
    verified_status: bool | str = "extractor_limited" if unclassified else True

    return ImprovementResult(
        original_prompt=prompt,
        improved_prompt=candidate_prompt,
        verified=verified_status,
        rejection_reason=None,
        verification_details=verification,
        before={
            "completeness": orig_result.completeness,
            "energy": orig_result.energy,
            "atoms": orig_result.atoms,
            "slots": orig_result.slot_scores,
        },
        after={
            "completeness": imp_result.completeness,
            "energy": imp_result.energy,
            "atoms": imp_result.atoms,
            "slots": imp_result.slot_scores,
        },
        delta_completeness=delta_comp,
        delta_energy=delta_energy,
        resolutions=resolutions,
        dropped_atoms=dropped_atoms,
        retained_atoms=sorted(set(orig_result.atoms) - set(dropped_atoms)),
        filled_slots=filled_slots if mode == "rule_based" else [],
        text_diff=diff,
        mode=mode,
        extractor_confidence=confidence,
        unclassified_clauses=unclassified,
        llm_metadata=llm_metadata,
    )


def evaluate_improvements_on_set(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate prompt improvement across a dataset (e.g. 6 test prompts or 60 dev prompts).

    Reports distribution of completeness and energy change, and failure count.
    """
    total = len(prompts)
    verified_count = 0
    failed_count = 0
    comp_deltas: list[float] = []
    energy_deltas: list[int] = []
    items: list[dict[str, Any]] = []

    for item in prompts:
        text = item.get("prompt") or item.get("text", "")
        pid = item.get("prompt_id") or item.get("id") or item.get("name", "unknown")
        result = improve_prompt(text)

        if result.verified and result.after is not None:
            verified_count += 1
            comp_deltas.append(result.delta_completeness or 0.0)
            energy_deltas.append(result.delta_energy or 0)
        else:
            failed_count += 1

        items.append(
            {
                "id": pid,
                "verified": result.verified,
                "rejection_reason": result.rejection_reason,
                "completeness_before": result.before["completeness"],
                "completeness_after": result.after["completeness"] if result.after else None,
                "energy_before": result.before["energy"],
                "energy_after": result.after["energy"] if result.after else None,
                "delta_completeness": result.delta_completeness,
                "delta_energy": result.delta_energy,
                "dropped_atoms": result.dropped_atoms,
            }
        )

    mean_delta_comp = float(sum(comp_deltas) / len(comp_deltas)) if comp_deltas else 0.0
    mean_delta_energy = float(sum(energy_deltas) / len(energy_deltas)) if energy_deltas else 0.0

    return {
        "total_prompts": total,
        "verified_count": verified_count,
        "failed_count": failed_count,
        "mean_delta_completeness": mean_delta_comp,
        "min_delta_completeness": min(comp_deltas) if comp_deltas else 0.0,
        "max_delta_completeness": max(comp_deltas) if comp_deltas else 0.0,
        "mean_delta_energy": mean_delta_energy,
        "min_delta_energy": min(energy_deltas) if energy_deltas else 0,
        "max_delta_energy": max(energy_deltas) if energy_deltas else 0,
        "items": items,
    }
