"""Run single evaluation on the frozen test set data/gold/synthetic/authored_prompts_SYNTHETIC.csv.

Performs one measurement run, reports all results honestly, and saves
to data/eval/extractor_frozen_report.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.energy import conflict_energy
from app.linter.core import extract_atoms, score_slots
from app.linter.datasets import CURRENT_EXTRACTOR_VERSION

ROOT = Path(__file__).resolve().parent.parent
FROZEN_PATH = ROOT / "data" / "gold" / "synthetic" / "authored_prompts_SYNTHETIC.csv"
HASH_PATH = ROOT / "data" / "gold" / "FROZEN_SHA256.txt"
OUT_PATH = ROOT / "data" / "eval" / "extractor_frozen_report.json"

ATOM_NAMES = [
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

SLOT_NAMES = [
    "role",
    "audience",
    "topic",
    "count",
    "n_options",
    "labeling",
    "marking",
    "explanation",
    "ordering",
]


def run_frozen_measurement() -> dict:
    assert FROZEN_PATH.exists(), f"Missing frozen test set: {FROZEN_PATH}"
    assert HASH_PATH.exists(), f"Missing hash file: {HASH_PATH}"

    expected_hash = HASH_PATH.read_text(encoding="utf-8").strip()
    actual_hash = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"Frozen set hash mismatch! Expected {expected_hash}, got {actual_hash}"

    with open(FROZEN_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    per_atom = {a: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for a in ATOM_NAMES}

    total_slot_slots = 0
    correct_slot_slots = 0

    wrong_conflict_energy_prompts = []
    per_prompt_results = []

    for r in rows:
        pid = r.get("prompt_id") or r.get("id", "unknown")
        text = r.get("prompt_text") or r.get("prompt", "")

        gold_atoms = [a for a in ATOM_NAMES if r.get(a) in ("1", 1, True, "true")]
        pred_atoms = extract_atoms(text)

        gold_slots = {s: float(r.get(s, 0.0)) for s in SLOT_NAMES}
        pred_slots = score_slots(text)

        for a in ATOM_NAMES:
            g = a in gold_atoms
            p = a in pred_atoms
            if g and p:
                per_atom[a]["tp"] += 1
            elif not g and p:
                per_atom[a]["fp"] += 1
            elif g and not p:
                per_atom[a]["fn"] += 1
            else:
                per_atom[a]["tn"] += 1

        for s in SLOT_NAMES:
            total_slot_slots += 1
            if pred_slots.get(s, 0.0) == gold_slots.get(s, 0.0):
                correct_slot_slots += 1

        gold_ce = conflict_energy(gold_atoms)
        pred_ce = conflict_energy(pred_atoms)

        if gold_ce["energy"] != pred_ce["energy"]:
            wrong_conflict_energy_prompts.append(
                {
                    "prompt_id": pid,
                    "gold_energy": gold_ce["energy"],
                    "pred_energy": pred_ce["energy"],
                    "gold_atoms": gold_atoms,
                    "pred_atoms": pred_atoms,
                    "gold_conflicts": gold_ce["conflicting_pairs"],
                    "pred_conflicts": pred_ce["conflicting_pairs"],
                }
            )

        per_prompt_results.append(
            {
                "prompt_id": pid,
                "gold_atoms": gold_atoms,
                "pred_atoms": pred_atoms,
                "gold_energy": gold_ce["energy"],
                "pred_energy": pred_ce["energy"],
                "slot_matches": {s: pred_slots.get(s, 0.0) == gold_slots.get(s, 0.0) for s in SLOT_NAMES},
            }
        )

    # Compute atom precision, recall, F1
    atom_metrics = {}
    for a in ATOM_NAMES:
        tp = per_atom[a]["tp"]
        fp = per_atom[a]["fp"]
        fn = per_atom[a]["fn"]
        tn = per_atom[a]["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        atom_metrics[a] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "gold_positive_n": tp + fn,
        }

    slot_agreement = correct_slot_slots / total_slot_slots if total_slot_slots > 0 else 0.0

    report = {
        "status": "computed",
        "frozen_set_file": "data/gold/synthetic/authored_prompts_SYNTHETIC.csv",
        "frozen_set_sha256": actual_hash,
        "extractor_version": CURRENT_EXTRACTOR_VERSION,
        "date": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "single_measurement_run": True,
        "overall_atom_summary": {
            "tp_total": sum(v["tp"] for v in per_atom.values()),
            "fp_total": sum(v["fp"] for v in per_atom.values()),
            "fn_total": sum(v["fn"] for v in per_atom.values()),
        },
        "atom_metrics": atom_metrics,
        "slot_agreement": round(slot_agreement, 4),
        "total_slot_comparisons": total_slot_slots,
        "correct_slot_comparisons": correct_slot_slots,
        "wrong_conflict_energy_prompts": wrong_conflict_energy_prompts,
        "per_prompt_results": per_prompt_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run_frozen_measurement()
    print("=== FROZEN TEST SET MEASUREMENT REPORT ===")
    print(f"File: {report['frozen_set_file']}")
    print(f"SHA256: {report['frozen_set_sha256']}")
    print(f"Date: {report['date']}")
    print(f"Sample size n: {report['n']}")
    print(
        f"Slot Agreement: {report['slot_agreement'] * 100:.2f}% ({report['correct_slot_comparisons']}/{report['total_slot_comparisons']})"
    )
    print(f"Wrong Conflict Energy Prompts: {len(report['wrong_conflict_energy_prompts'])}")
    for p in report["wrong_conflict_energy_prompts"]:
        print(f" - {p['prompt_id']}: gold_energy={p['gold_energy']} vs pred_energy={p['pred_energy']}")
    print("\nPer-atom metrics:")
    for a, m in report["atom_metrics"].items():
        prec_str = f"{m['precision']:.3f}" if m["precision"] is not None else "N/A"
        rec_str = f"{m['recall']:.3f}" if m["recall"] is not None else "N/A"
        f1_str = f"{m['f1']:.3f}" if m["f1"] is not None else "N/A"
        print(
            f" - {a:20s}: P={prec_str:5s} R={rec_str:5s} F1={f1_str:5s} (gold positive n={m['gold_positive_n']}, tp={m['tp']}, fp={m['fp']}, fn={m['fn']})"
        )
