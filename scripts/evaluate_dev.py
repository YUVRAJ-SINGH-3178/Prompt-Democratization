"""Evaluate extractor against data/gold/dev_authored/dev_prompts.json."""

from __future__ import annotations

import json
from pathlib import Path

from app.linter.core import extract_atoms, score_slots

DEV_PATH = Path("data/gold/dev_authored/dev_prompts.json")


def main():
    prompts = json.loads(DEV_PATH.read_text(encoding="utf-8"))

    total_slot_comparisons = 0
    correct_slots = 0

    atom_names = [
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
    ]

    per_atom = {a: {"tp": 0, "fp": 0, "fn": 0} for a in atom_names}

    failures = []

    for item in prompts:
        pid = item["prompt_id"]
        text = item["prompt_text"]
        gold_atoms = set(item["atoms"])
        pred_atoms = set(extract_atoms(text))

        gold_slots = item["slots"]
        pred_slots = score_slots(text)

        for a in atom_names:
            in_gold = a in gold_atoms
            in_pred = a in pred_atoms
            if in_gold and in_pred:
                per_atom[a]["tp"] += 1
            elif in_pred and not in_gold:
                per_atom[a]["fp"] += 1
            elif in_gold and not in_pred:
                per_atom[a]["fn"] += 1

        for s, g_val in gold_slots.items():
            p_val = pred_slots.get(s, 0.0)
            total_slot_comparisons += 1
            if p_val == g_val:
                correct_slots += 1
            else:
                failures.append(f"{pid} slot {s}: expected {g_val}, got {p_val}")

        if gold_atoms != pred_atoms:
            failures.append(f"{pid} atoms: expected {sorted(gold_atoms)}, got {sorted(pred_atoms)}")

    print(f"Total dev prompts: {len(prompts)}")
    tp_sum = sum(v["tp"] for v in per_atom.values())
    fp_sum = sum(v["fp"] for v in per_atom.values())
    fn_sum = sum(v["fn"] for v in per_atom.values())

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    slot_acc = correct_slots / total_slot_comparisons if total_slot_comparisons else 0.0

    print(f"Overall Atom Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"Slot Agreement: {slot_acc:.3f} ({correct_slots}/{total_slot_comparisons})")
    print(f"Failures count: {len(failures)}")
    print("\nSample failures:")
    for f in failures[:20]:
        print(" -", f)


if __name__ == "__main__":
    main()
