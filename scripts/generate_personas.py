"""Deterministically generate the 20 synthetic professor cards."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 731_204
COURSES = [
    ("Computing", "Machine Learning"),
    ("Computing", "Database Management Systems"),
    ("Computing", "Operating Systems"),
    ("Computing", "Computer Networks"),
    ("Literature", "Introduction to Literary Analysis"),
    ("Literature", "Modern Poetry"),
    ("Literature", "World Literature"),
    ("Literature", "Creative Writing"),
    ("Engineering", "Electrical Circuits"),
    ("Engineering", "Thermodynamics"),
    ("Engineering", "Civil Engineering Materials"),
    ("Engineering", "Control Systems"),
    ("Medical", "Human Anatomy"),
    ("Medical", "Pharmacology"),
    ("Medical", "Clinical Microbiology"),
    ("Medical", "Epidemiology"),
    ("Business", "Financial Accounting"),
    ("Business", "Marketing Principles"),
    ("Business", "Operations Management"),
    ("Business", "Corporate Finance"),
]
HABITS = [
    ("poor_vague_underspecified", "Often omits audience, scope, and what counts as a good answer."),
    ("poor_vague_underspecified", "Uses shorthand and leaves key task details implicit."),
    ("poor_contradictory", "Combines requirements that cannot all be satisfied at once."),
    ("poor_contradictory", "Adds conflicting constraints, sometimes without noticing."),
    ("poor_adversarial_confusing", "Intentionally asks for confusing wording or ambiguous items."),
    ("good_complete", "Usually specifies audience, topic, count, format, answer key, and rationale."),
    ("good_complete", "Writes structured prompts with clear constraints and output checks."),
    ("good_complete", "Provides complete context and defines what a usable assessment looks like."),
    ("good_complete", "Carefully states scope, difficulty progression, and marking expectations."),
    ("good_complete", "Uses concise but complete instructions and explicit answer formatting."),
    ("concise_practical", "Prefers short instructions and adds detail only when needed."),
    ("recall_first", "Usually starts with recall questions before applying concepts."),
    ("application_first", "Favors scenarios and transfer over definition recall."),
    ("mixed_style", "Balances recall, interpretation, and application in one assessment."),
    ("iterative_editor", "Drafts quickly, then edits constraints after seeing a first version."),
    ("format_focused", "Pays close attention to labels, answer keys, and consistent formatting."),
    ("minimalist", "Keeps prompts brief and expects the model to infer routine details."),
    ("context_rich", "Adds course-specific context and examples before stating the task."),
    ("exam_oriented", "Emphasizes coverage, time pressure, and clear marking criteria."),
    ("discussion_oriented", "Values interpretive range and nuanced distractors."),
]


def personas() -> list[dict]:
    rng = random.Random(SEED)
    shuffled_habits = HABITS.copy()
    rng.shuffle(shuffled_habits)
    styles = ["recall", "application", "mixed"]
    tolerances = ["low", "medium", "high"]
    seniorities = ["early-career", "mid-career", "senior", "principal"]
    out = []
    for index, ((discipline, course), (habit, habit_text)) in enumerate(zip(COURSES, shuffled_habits), 1):
        years = rng.randint(2, 38)
        style = styles[(index + rng.randrange(3)) % len(styles)]
        tolerance = tolerances[(index + rng.randrange(3)) % len(tolerances)]
        seniority = "early-career" if years < 7 else "mid-career" if years < 16 else seniorities[2 + rng.randrange(2)]
        out.append(
            {
                "id": f"SIM-P{index:02d}",
                "synthetic": True,
                "source": "sim_persona",
                "discipline": discipline,
                "course": course,
                "years_teaching": years,
                "seniority": seniority,
                "assessment_style": style,
                "ambiguity_tolerance": tolerance,
                "prompt_writing_habit": habit,
                "habit_description": habit_text,
                "background": (
                    f"A synthetic {seniority} instructor in {course} with {years} years of teaching. "
                    f"They favor {style} assessment and have {tolerance} tolerance for ambiguity. "
                    f"Their prompt habit is {habit.replace('_', ' ')}."
                ),
            }
        )
    return out


def main() -> None:
    values = personas()
    path = ROOT / "data" / "simulated" / "personas.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a YAML 1.2 subset and avoids nondeterministic third-party serialization.
    path.write_text(
        json.dumps({"seed": SEED, "personas": values}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"seed={SEED}; personas={len(values)}; file={path.relative_to(ROOT)}")
    print("id       discipline  course                              years seniority      style       ambiguity habit")
    for p in values:
        print(
            f"{p['id']:<8} {p['discipline']:<11} {p['course']:<36} {p['years_teaching']:>2} "
            f"{p['seniority']:<14} {p['assessment_style']:<11} {p['ambiguity_tolerance']:<8} {p['prompt_writing_habit']}"
        )


if __name__ == "__main__":
    main()
