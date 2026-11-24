"""Real Data Readiness Checklist.

Inspects data/gold/inbox/ and the database to report:
- Counts of real prompts, annotators, shared prompts, raters, comparisons.
- Readiness status and requirements for each analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.auth_db import connect

INBOX_DIR = Path(__file__).resolve().parent.parent / "data" / "gold" / "inbox"
PROMPTS_CSV = INBOX_DIR / "prompts.csv"


def get_checklist_data() -> dict[str, Any]:
    # 1. Real prompts in inbox
    real_prompts = []
    if PROMPTS_CSV.exists():
        try:
            with open(PROMPTS_CSV, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                real_prompts = [row for row in reader if row.get("consent", "").strip().lower() == "yes"]
        except Exception:
            pass

    # 2. Annotator files and shared prompts
    annotator_files = list(INBOX_DIR.glob("annotations_*.csv"))
    annotator_files = [f for f in annotator_files if not f.name.endswith("TEMPLATE.csv")]

    prompt_annotator_map: dict[str, set[str]] = {}
    for af in annotator_files:
        annotator_id = af.stem.replace("annotations_", "")
        try:
            with open(af, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("prompt_id")
                    if pid:
                        prompt_annotator_map.setdefault(pid, set()).add(annotator_id)
        except Exception:
            pass

    shared_annotated_prompts = [pid for pid, annots in prompt_annotator_map.items() if len(annots) >= 2]

    # 3. Database comparisons and raters
    total_comparisons = 0
    nontie_comparisons = 0
    raters: set[str] = set()
    pair_raters: dict[tuple[str, str], set[str]] = {}

    try:
        with connect() as db:
            rows = db.execute("SELECT rater_id, prompt_a_id, prompt_b_id, winner FROM comparisons").fetchall()
            total_comparisons = len(rows)
            for r in rows:
                if r["rater_id"]:
                    raters.add(r["rater_id"])
                if r["winner"] != "tie":
                    nontie_comparisons += 1
                    pair_key = tuple(sorted([r["prompt_a_id"], r["prompt_b_id"]]))
                    pair_raters.setdefault(pair_key, set()).add(r["rater_id"])
    except Exception:
        pass

    shared_pairs_count = sum(1 for raters_for_pair in pair_raters.values() if len(raters_for_pair) >= 2)

    return {
        "n_real_prompts": len(real_prompts),
        "n_annotators": len(annotator_files),
        "n_shared_annotated_prompts": len(shared_annotated_prompts),
        "n_raters": len(raters),
        "n_total_comparisons": total_comparisons,
        "n_nontie_comparisons": nontie_comparisons,
        "n_shared_comparison_pairs": shared_pairs_count,
    }


def print_checklist() -> None:
    data = get_checklist_data()

    print("=" * 70)
    print("REAL-DATA READINESS CHECKLIST")
    print("=" * 70)
    print(f"1. Real prompts in inbox (consent=yes):      {data['n_real_prompts']}")
    print(f"2. Annotator files in inbox:                {data['n_annotators']}")
    print(f"3. Shared prompts (annotated by >= 2):      {data['n_shared_annotated_prompts']}")
    print(f"4. Total raters in database:                {data['n_raters']}")
    print(f"5. Total comparisons in database:           {data['n_total_comparisons']}")
    print(f"6. Non-tie comparisons in database:         {data['n_nontie_comparisons']}")
    print(f"7. Shared comparison pairs (rated by >= 2): {data['n_shared_comparison_pairs']}")
    print("-" * 70)
    print("ANALYSIS READINESS STATUS:")
    print("-" * 70)

    # Analysis 1: Extractor Report
    ext_ready = (data["n_real_prompts"] >= 1) and (data["n_annotators"] >= 2)
    print("[Analysis: Linter Extractor Report (make extractor-report)]")
    print(f"  Status: {'READY' if ext_ready else 'BLOCKED'}")
    print(
        f"  Requires: >= 1 real prompt (current: {data['n_real_prompts']}), >= 2 annotator files (current: {data['n_annotators']})"
    )
    if not ext_ready:
        print("  Missing: verified real prompts with consent and dual annotator files")

    # Analysis 2: Inter-rater Reliability (Cohen's Kappa)
    kappa_ready = (data["n_raters"] >= 2) and (data["n_shared_comparison_pairs"] >= 30)
    print("\n[Analysis: Inter-rater Reliability / Cohen's Kappa]")
    print(f"  Status: {'READY' if kappa_ready else 'BLOCKED'}")
    print(
        f"  Requires: >= 2 raters (current: {data['n_raters']}), >= 30 shared pairs (current: {data['n_shared_comparison_pairs']})"
    )
    if not kappa_ready:
        print("  Missing: >= 30 shared prompt pairs rated by the same pair of raters")

    # Analysis 3: Bradley-Terry Learner Training
    learner_ready = (data["n_nontie_comparisons"] >= 50) and (data["n_raters"] >= 2)
    print("\n[Analysis: Bradley-Terry Learner Training (fit_bradley_terry)]")
    print(f"  Status: {'READY' if learner_ready else 'BLOCKED'}")
    print(
        f"  Requires: >= 50 non-tie comparisons (current: {data['n_nontie_comparisons']}), >= 2 raters (current: {data['n_raters']})"
    )
    if not learner_ready:
        print("  Missing: >= 50 non-tie comparisons from >= 2 raters")

    # Analysis 4: Real-Data Evaluation Pipeline
    eval_ready = (data["n_real_prompts"] >= 1) and (data["n_nontie_comparisons"] >= 50) and (data["n_raters"] >= 2)
    print("\n[Analysis: Real Evaluation Pipeline (make real-eval)]")
    print(f"  Status: {'READY' if eval_ready else 'BLOCKED'}")
    print("  Requires: >= 1 real prompt, >= 50 non-tie comparisons from >= 2 raters")
    print("  Split rule: Prompt-pair disjoint (no prompt in both train and test)")
    if not eval_ready:
        print("  Missing: human comparison dataset and verified real prompts")

    print("=" * 70)


if __name__ == "__main__":
    print_checklist()
