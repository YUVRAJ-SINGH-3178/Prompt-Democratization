"""Evaluation module for honest development and frozen evaluations (Tasks 3 and 4).

Generates side-by-side evaluation reports for:
1. Rules only
2. Reader only (verified quotes only)
3. Combined system

Strict rules:
- No invented numbers or fake LLM responses in UI or reports.
- If no API key is available, evaluates rules only and prints 'reader not run: no API key'.
- Reports state task, data provenance, sample size n, confidence intervals, and review rates.
- No em dashes in user-facing text.
- Never write 'best', 'perfect', or 'state of the art'.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from app.extractor.combine import combine_rules_and_reader
from app.extractor.llm_reader import (
    ATOM_NAMES,
    SLOT_NAMES,
    is_reader_enabled,
    read_prompt_with_llm,
)
from app.linter.core import evaluate, extract_atoms, score_slots
from app.linter.rubrics import RUBRIC_VERSION, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "data" / "eval"
DEV_JSON_PATH = ROOT / "data" / "gold" / "dev_authored" / "dev_prompts.json"
DEV_OLD_FROZEN_PATH = ROOT / "data" / "gold" / "dev_authored" / "authored_prompts_SYNTHETIC.csv"
FROZEN_V2_DIR = ROOT / "data" / "gold" / "frozen_v2"
FROZEN_V2_HASH_FILE = ROOT / "data" / "gold" / "FROZEN_V2_SHA256.txt"


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = float(successes) / float(total)
    denom = 1.0 + (z * z) / float(total)
    center = (p + (z * z) / (2.0 * float(total))) / denom
    variance = (p * (1.0 - p) / float(total)) + (z * z) / (4.0 * float(total * total))
    spread = (z / denom) * math.sqrt(max(0.0, variance))
    return round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4)


def load_dev_prompts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load both dev prompt splits: 68 authored prompts and 12 legacy synthetic prompts."""
    # Split 1: 68 dev prompts
    with open(DEV_JSON_PATH, encoding="utf-8") as f:
        dev_68_raw = json.load(f)

    dev_68 = []
    for item in dev_68_raw:
        dev_68.append(
            {
                "prompt_id": item["prompt_id"],
                "prompt_text": item["prompt_text"],
                "gold_atoms": sorted(item.get("atoms", [])),
                "gold_slots": {s: float(item.get("slots", {}).get(s, 0.0)) for s in SLOT_NAMES},
                "source": item.get("source", "authored_by_builder"),
                "synthetic": item.get("synthetic", 1),
            }
        )

    # Split 2: 12 legacy synthetic prompts
    dev_12 = []
    with open(DEV_OLD_FROZEN_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            active_atoms = [a for a in ATOM_NAMES if row.get(a) in ("1", "True", "true", 1)]
            slot_scores = {s: float(row.get(s, 0.0)) for s in SLOT_NAMES}
            dev_12.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_text": row["prompt_text"],
                    "gold_atoms": sorted(active_atoms),
                    "gold_slots": slot_scores,
                    "source": row.get("source", "authored_by_builder"),
                    "synthetic": int(row.get("synthetic", 1)),
                }
            )

    return dev_68, dev_12


def evaluate_system_on_dataset(
    dataset: list[dict[str, Any]],
    system_name: str,  # 'rules_only', 'reader_only', 'combined'
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Run evaluation of one system across a given prompt dataset."""
    n = len(dataset)
    if n == 0:
        return {"n": 0, "status": "empty_dataset"}

    atom_tp = {a: 0 for a in ATOM_NAMES}
    atom_fp = {a: 0 for a in ATOM_NAMES}
    atom_fn = {a: 0 for a in ATOM_NAMES}
    atom_gold_count = {a: 0 for a in ATOM_NAMES}
    atom_pred_count = {a: 0 for a in ATOM_NAMES}

    slot_matches = 0
    total_slots = n * len(SLOT_NAMES)

    total_review_items = 0
    total_eval_items = n * (len(ATOM_NAMES) + len(SLOT_NAMES))

    wrong_energy_strict: list[dict[str, Any]] = []
    wrong_energy_range: list[dict[str, Any]] = []

    for item in dataset:
        p_id = item["prompt_id"]
        text = item["prompt_text"]
        gold_atoms = set(item["gold_atoms"])
        gold_slots = item["gold_slots"]
        true_energy = evaluate(text, atoms=list(gold_atoms)).energy

        if system_name == "rules_only":
            pred_atoms = set(extract_atoms(text))
            pred_slots = score_slots(text)
            pred_energy = evaluate(text, atoms=list(pred_atoms)).energy
            energy_range = [pred_energy, pred_energy]
            review_items = 0

        elif system_name == "reader_only":
            reader_out = read_prompt_with_llm(text, model=model, api_key=api_key)
            if not reader_out.get("enabled"):
                return {"status": "not_run_no_api_key", "n": n}
            # Verified quotes only
            pred_atoms = {
                a for a, v in reader_out.get("atoms", {}).items() if v.get("asserted") and v.get("quote_valid")
            }
            pred_slots = {
                s: float(v.get("score", 0.0)) for s, v in reader_out.get("slots", {}).items() if v.get("quote_valid")
            }
            pred_energy = evaluate(text, atoms=list(pred_atoms)).energy
            energy_range = [pred_energy, pred_energy]
            review_items = 0

        elif system_name == "combined":
            comb = combine_rules_and_reader(
                text,
                prompt_id=p_id,
                api_key=api_key,
                model=model,
                record_reviews=False,
            )
            pred_atoms = set(comb.accepted_atoms)
            pred_slots = comb.accepted_slots
            pred_energy = comb.energy
            energy_range = comb.energy_range
            review_items = len(comb.review_atoms) + len(comb.review_slots)

        else:
            raise ValueError(f"Unknown system name: {system_name}")

        total_review_items += review_items

        # Atom evaluation
        for a in ATOM_NAMES:
            in_gold = a in gold_atoms
            in_pred = a in pred_atoms
            if in_gold:
                atom_gold_count[a] += 1
            if in_pred:
                atom_pred_count[a] += 1
            if in_gold and in_pred:
                atom_tp[a] += 1
            elif (not in_gold) and in_pred:
                atom_fp[a] += 1
            elif in_gold and (not in_pred):
                atom_fn[a] += 1

        # Slot evaluation
        for s in SLOT_NAMES:
            if round(float(pred_slots.get(s, 0.0)), 2) == round(float(gold_slots.get(s, 0.0)), 2):
                slot_matches += 1

        # Energy evaluation
        strict_correct = pred_energy == true_energy
        range_correct = energy_range[0] <= true_energy <= energy_range[1]

        if not strict_correct:
            wrong_energy_strict.append(
                {
                    "prompt_id": p_id,
                    "true_energy": true_energy,
                    "predicted_energy": pred_energy,
                    "energy_range": energy_range,
                }
            )
        if not range_correct:
            wrong_energy_range.append(
                {
                    "prompt_id": p_id,
                    "true_energy": true_energy,
                    "predicted_energy": pred_energy,
                    "energy_range": energy_range,
                }
            )

    # Aggregate atom metrics
    per_atom_metrics: dict[str, dict[str, Any]] = {}
    total_tp = sum(atom_tp.values())
    total_fp = sum(atom_fp.values())
    total_fn = sum(atom_fn.values())

    for a in ATOM_NAMES:
        tp = atom_tp[a]
        fp = atom_fp[a]
        fn = atom_fn[a]
        prec = round(float(tp) / float(tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(float(tp) / float(tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2.0 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        prec_ci = wilson_interval(tp, tp + fp)
        rec_ci = wilson_interval(tp, tp + fn)
        per_atom_metrics[a] = {
            "gold_count": atom_gold_count[a],
            "pred_count": atom_pred_count[a],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "precision_ci_95": prec_ci,
            "recall": rec,
            "recall_ci_95": rec_ci,
            "f1": f1,
        }

    micro_prec = round(float(total_tp) / float(total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = round(float(total_tp) / float(total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        round(2.0 * micro_prec * micro_rec / (micro_prec + micro_rec), 4) if (micro_prec + micro_rec) > 0 else 0.0
    )
    micro_prec_ci = wilson_interval(total_tp, total_tp + total_fp)
    micro_rec_ci = wilson_interval(total_tp, total_tp + total_fn)

    slot_agreement = round(float(slot_matches) / float(total_slots), 4)
    slot_agreement_ci = wilson_interval(slot_matches, total_slots)
    review_rate = round(float(total_review_items) / float(total_eval_items), 4)

    return {
        "status": "success",
        "n_prompts": n,
        "review_rate": review_rate,
        "total_review_items": total_review_items,
        "total_eval_items": total_eval_items,
        "slot_agreement": slot_agreement,
        "slot_agreement_ci_95": slot_agreement_ci,
        "slot_matches": slot_matches,
        "total_slots": total_slots,
        "micro_precision": micro_prec,
        "micro_precision_ci_95": micro_prec_ci,
        "micro_recall": micro_rec,
        "micro_recall_ci_95": micro_rec_ci,
        "micro_f1": micro_f1,
        "per_atom": per_atom_metrics,
        "wrong_energy_strict_count": len(wrong_energy_strict),
        "wrong_energy_strict_prompts": wrong_energy_strict,
        "wrong_energy_range_count": len(wrong_energy_range),
        "wrong_energy_range_prompts": wrong_energy_range,
    }


def generate_dev_eval_markdown(results: dict[str, Any]) -> str:
    """Generate Markdown report dynamically from numbers without typed hard-coded strings."""
    lines: list[str] = []
    lines.append("# Development Dataset Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {results['metadata']['date']}")
    lines.append(f"**Model:** {results['metadata']['model']}")
    lines.append(f"**Rubric Version:** {results['metadata']['rubric_version']}")
    lines.append(f"**Schema Version:** {results['metadata']['schema_version']}")
    v = results["metadata"]["versions"]
    lines.append(
        f"**Environment:** Python {v['python']}, NumPy {v['numpy']}, SciPy {v['scipy']}, scikit-learn {v['scikit_learn']}"
    )
    lines.append("")
    lines.append("> **DATA PROVENANCE STATEMENT**:")
    lines.append(
        f"> Total prompts: {results['provenance']['total_prompts']} (Split 1: {results['provenance']['dev_68_n']} authored prompts, Split 2: {results['provenance']['dev_12_n']} legacy synthetic prompts)."
    )
    lines.append(
        f"> Source label: {results['provenance']['source']}. Synthetic label: {results['provenance']['synthetic']}."
    )
    lines.append(
        "> These prompts were authored by the builder for tuning and development. They are not real faculty data."
    )
    lines.append("")

    for split_key, split_title in [
        ("dev_68", "Split 1: Authored Development Prompts (n = 68)"),
        ("dev_12", "Split 2: Legacy Authored Prompts (n = 12)"),
        ("combined_dev", "Combined Development Pool (n = 80)"),
    ]:
        lines.append(f"## {split_title}")
        lines.append("")
        split_data = results["splits"].get(split_key, {})
        lines.append("| Metric | Rules Only | Reader Only | Combined System |")
        lines.append("|---|---|---|---|")

        r_only = split_data.get("rules_only", {})
        rd_only = split_data.get("reader_only", {})
        comb = split_data.get("combined", {})

        def _fmt_pct(d: dict[str, Any], key: str) -> str:
            if d.get("status") != "success":
                return "Not run (no API key)" if d.get("status") == "not_run_no_api_key" else "N/A"
            val = d.get(key)
            return f"{round(val * 100, 1)}%" if val is not None else "N/A"

        def _fmt_ci(d: dict[str, Any], key: str, ci_key: str) -> str:
            if d.get("status") != "success":
                return "Not run (no API key)" if d.get("status") == "not_run_no_api_key" else "N/A"
            val = d.get(key)
            ci = d.get(ci_key)
            if val is None or ci is None:
                return "N/A"
            return f"{round(val * 100, 1)}% [{round(ci[0] * 100, 1)}%, {round(ci[1] * 100, 1)}%]"

        def _fmt_int(d: dict[str, Any], key: str) -> str:
            if d.get("status") != "success":
                return "Not run (no API key)" if d.get("status") == "not_run_no_api_key" else "N/A"
            val = d.get(key)
            return str(val) if val is not None else "N/A"

        lines.append(
            f"| **Review Rate** | {_fmt_pct(r_only, 'review_rate')} | {_fmt_pct(rd_only, 'review_rate')} | {_fmt_pct(comb, 'review_rate')} |"
        )
        lines.append(
            f"| **Slot Agreement** | {_fmt_ci(r_only, 'slot_agreement', 'slot_agreement_ci_95')} | {_fmt_ci(rd_only, 'slot_agreement', 'slot_agreement_ci_95')} | {_fmt_ci(comb, 'slot_agreement', 'slot_agreement_ci_95')} |"
        )
        lines.append(
            f"| **Micro Precision** | {_fmt_ci(r_only, 'micro_precision', 'micro_precision_ci_95')} | {_fmt_ci(rd_only, 'micro_precision', 'micro_precision_ci_95')} | {_fmt_ci(comb, 'micro_precision', 'micro_precision_ci_95')} |"
        )
        lines.append(
            f"| **Micro Recall** | {_fmt_ci(r_only, 'micro_recall', 'micro_recall_ci_95')} | {_fmt_ci(rd_only, 'micro_recall', 'micro_recall_ci_95')} | {_fmt_ci(comb, 'micro_recall', 'micro_recall_ci_95')} |"
        )
        lines.append(
            f"| **Micro F1** | {_fmt_pct(r_only, 'micro_f1')} | {_fmt_pct(rd_only, 'micro_f1')} | {_fmt_pct(comb, 'micro_f1')} |"
        )
        lines.append(
            f"| **Wrong Energy Prompts (Strict)** | {_fmt_int(r_only, 'wrong_energy_strict_count')} | {_fmt_int(rd_only, 'wrong_energy_strict_count')} | {_fmt_int(comb, 'wrong_energy_strict_count')} |"
        )
        lines.append(
            f"| **Wrong Energy Prompts (Range-based)** | {_fmt_int(r_only, 'wrong_energy_range_count')} | {_fmt_int(rd_only, 'wrong_energy_range_count')} | {_fmt_int(comb, 'wrong_energy_range_count')} |"
        )
        lines.append("")

        lines.append("### Per-Atom Breakdown (Rules Only)")
        lines.append("| Atom | Gold Count | Pred Count | Precision (95% CI) | Recall (95% CI) | F1 |")
        lines.append("|---|---|---|---|---|---|")
        if r_only.get("status") == "success":
            for a in ATOM_NAMES:
                am = r_only["per_atom"][a]
                p_str = f"{round(am['precision'] * 100, 1)}% [{round(am['precision_ci_95'][0] * 100, 1)}%, {round(am['precision_ci_95'][1] * 100, 1)}%]"
                r_str = f"{round(am['recall'] * 100, 1)}% [{round(am['recall_ci_95'][0] * 100, 1)}%, {round(am['recall_ci_95'][1] * 100, 1)}%]"
                lines.append(
                    f"| `{a}` | {am['gold_count']} | {am['pred_count']} | {p_str} | {r_str} | {round(am['f1'] * 100, 1)}% |"
                )
        else:
            lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
        lines.append("")

    return "\n".join(lines)


def run_dev_eval() -> dict[str, Any]:
    """Execute dev evaluation on all 68 + 12 dev prompts and write json + md reports."""
    dev_68, dev_12 = load_dev_prompts()
    combined_all = dev_68 + dev_12

    reader_active = is_reader_enabled()
    if not reader_active:
        print("reader not run: no API key")

    key = os.environ.get("STUDY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("STUDY_MODEL", "gpt-4o-mini")

    systems = ["rules_only"]
    if reader_active:
        systems.extend(["reader_only", "combined"])

    splits_data: dict[str, dict[str, Any]] = {
        "dev_68": {},
        "dev_12": {},
        "combined_dev": {},
    }

    # Evaluate each system on each split
    for sys_name in ["rules_only", "reader_only", "combined"]:
        if sys_name != "rules_only" and not reader_active:
            splits_data["dev_68"][sys_name] = {"status": "not_run_no_api_key"}
            splits_data["dev_12"][sys_name] = {"status": "not_run_no_api_key"}
            splits_data["combined_dev"][sys_name] = {"status": "not_run_no_api_key"}
            continue

        splits_data["dev_68"][sys_name] = evaluate_system_on_dataset(dev_68, sys_name, api_key=key, model=model)
        splits_data["dev_12"][sys_name] = evaluate_system_on_dataset(dev_12, sys_name, api_key=key, model=model)
        splits_data["combined_dev"][sys_name] = evaluate_system_on_dataset(
            combined_all, sys_name, api_key=key, model=model
        )

    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "model": model if reader_active else "none",
            "reader_enabled": reader_active,
            "rubric_version": RUBRIC_VERSION,
            "schema_version": SCHEMA_VERSION,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "provenance": {
            "source": "authored_by_builder",
            "synthetic": 1,
            "total_prompts": len(combined_all),
            "dev_68_n": len(dev_68),
            "dev_12_n": len(dev_12),
        },
        "splits": splits_data,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "dev_evaluation_report.json"
    md_path = EVAL_DIR / "dev_evaluation_report.md"

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_content = generate_dev_eval_markdown(output)
    md_path.write_text(md_content, encoding="utf-8")

    return output


def compute_normalized_file_sha256(path: Path) -> str:
    """Compute SHA-256 with CRLF to LF line ending normalization."""
    import hashlib

    content = path.read_bytes()
    normalized = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def run_frozen_eval(force: bool = False) -> dict[str, Any]:
    """Execute one-command frozen evaluation (Task 4).

    Strict rules:
    - Aborts if data/gold/frozen_v2/ directory is missing.
    - Verifies line-ending normalised hash against FROZEN_V2_SHA256.txt.
    - Runs once; refuses second run unless force=True.
    - Records forced runs in report.
    - Reports strict and range-based conflict handling.
    """
    if not FROZEN_V2_DIR.exists():
        raise FileNotFoundError(
            f"frozen_v2 directory not found at {FROZEN_V2_DIR}. "
            "Please provide data/gold/frozen_v2/ to run frozen evaluation."
        )

    # Locate frozen file (CSV or JSON)
    frozen_files = list(FROZEN_V2_DIR.glob("*.csv")) + list(FROZEN_V2_DIR.glob("*.json"))
    if not frozen_files:
        raise FileNotFoundError(f"No CSV or JSON files found in {FROZEN_V2_DIR}.")

    target_file = frozen_files[0]

    # Verify SHA-256 against FROZEN_V2_SHA256.txt if exists
    hash_candidate = FROZEN_V2_DIR / "FROZEN_V2_SHA256.txt"
    hash_file = hash_candidate if hash_candidate.exists() else FROZEN_V2_HASH_FILE
    if hash_file.exists():
        expected_hash = hash_file.read_text(encoding="utf-8").strip()
        actual_hash = compute_normalized_file_sha256(target_file)
        if actual_hash.casefold() != expected_hash.casefold():
            raise ValueError(
                f"Frozen set hash mismatch for {target_file.name}. Expected: {expected_hash}, Actual: {actual_hash}"
            )

    report_json_path = EVAL_DIR / "frozen_v2_report.json"
    report_md_path = EVAL_DIR / "frozen_v2_report.md"

    # Enforce run once
    prior_forced_count = 0
    if report_json_path.exists():
        if not force:
            raise RuntimeError(
                f"Frozen evaluation report already exists at {report_json_path}. "
                "Refusing to run a second time unless --force is passed."
            )
        try:
            prior_data = json.loads(report_json_path.read_text(encoding="utf-8"))
            prior_forced_count = int(prior_data.get("execution", {}).get("forced_run_count", 0)) + 1
        except Exception:
            prior_forced_count = 1

    # Load frozen dataset
    frozen_items: list[dict[str, Any]] = []
    if target_file.suffix == ".json":
        raw_items = json.loads(target_file.read_text(encoding="utf-8"))
        for item in raw_items:
            frozen_items.append(
                {
                    "prompt_id": item["prompt_id"],
                    "prompt_text": item["prompt_text"],
                    "gold_atoms": sorted(item.get("atoms", [])),
                    "gold_slots": {s: float(item.get("slots", {}).get(s, 0.0)) for s in SLOT_NAMES},
                    "source": "authored_by_builder",
                    "synthetic": 1,
                }
            )
    else:
        with open(target_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                active_atoms = [a for a in ATOM_NAMES if row.get(a) in ("1", "True", "true", 1)]
                slot_scores = {s: float(row.get(s, 0.0)) for s in SLOT_NAMES}
                frozen_items.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "prompt_text": row["prompt_text"],
                        "gold_atoms": sorted(active_atoms),
                        "gold_slots": slot_scores,
                        "source": "authored_by_builder",
                        "synthetic": 1,
                    }
                )

    reader_active = is_reader_enabled()
    key = os.environ.get("STUDY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("STUDY_MODEL", "gpt-4o-mini")

    results_rules = evaluate_system_on_dataset(frozen_items, "rules_only")
    results_reader = (
        evaluate_system_on_dataset(frozen_items, "reader_only", api_key=key, model=model)
        if reader_active
        else {"status": "not_run_no_api_key"}
    )
    results_comb = (
        evaluate_system_on_dataset(frozen_items, "combined", api_key=key, model=model)
        if reader_active
        else {"status": "not_run_no_api_key"}
    )

    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "model": model if reader_active else "none",
            "reader_enabled": reader_active,
            "rubric_version": RUBRIC_VERSION,
            "schema_version": SCHEMA_VERSION,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "provenance": {
            "dataset_file": target_file.name,
            "source": "authored_by_builder",
            "synthetic": 1,
            "statement": "The frozen set is authored, not real faculty data.",
            "n_prompts": len(frozen_items),
        },
        "execution": {
            "forced": force,
            "forced_run_count": prior_forced_count,
        },
        "systems": {
            "rules_only": results_rules,
            "reader_only": results_reader,
            "combined": results_comb,
        },
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Generate Markdown report
    meta_dict = dict(output["metadata"])
    prov_dict = dict(output["provenance"])
    md_lines = [
        "# Frozen Test Set Evaluation Report (v2)",
        "",
        f"**Date:** {meta_dict['date']}",
        f"**Model:** {meta_dict['model']}",
        f"**Rubric Version:** {meta_dict['rubric_version']}",
        f"**Schema Version:** {meta_dict['schema_version']}",
        f"**Execution:** {'Forced run #' + str(prior_forced_count) if force else 'Initial single run'}",
        "",
        "> **DATA PROVENANCE STATEMENT**:",
        f"> {prov_dict['statement']}",
        f"> Sample size: n = {prov_dict['n_prompts']} prompts.",
        "",
        "## Performance Summary",
        "| Metric | Rules Only | Reader Only | Combined System |",
        "|---|---|---|---|",
    ]

    def _fmt(d: dict[str, Any], k: str) -> str:
        if d.get("status") != "success":
            return "Not run (no API key)" if d.get("status") == "not_run_no_api_key" else "N/A"
        val = d.get(k)
        return f"{round(val * 100, 1)}%" if val is not None else "N/A"

    def _fmt_ci(d: dict[str, Any], k: str, cik: str) -> str:
        if d.get("status") != "success":
            return "Not run (no API key)" if d.get("status") == "not_run_no_api_key" else "N/A"
        val = d.get(k)
        ci = d.get(cik)
        if val is None or ci is None:
            return "N/A"
        return f"{round(val * 100, 1)}% [{round(ci[0] * 100, 1)}%, {round(ci[1] * 100, 1)}%]"

    md_lines.append(
        f"| **Review Rate** | {_fmt(results_rules, 'review_rate')} | {_fmt(results_reader, 'review_rate')} | {_fmt(results_comb, 'review_rate')} |"
    )
    md_lines.append(
        f"| **Slot Agreement** | {_fmt_ci(results_rules, 'slot_agreement', 'slot_agreement_ci_95')} | {_fmt_ci(results_reader, 'slot_agreement', 'slot_agreement_ci_95')} | {_fmt_ci(results_comb, 'slot_agreement', 'slot_agreement_ci_95')} |"
    )
    md_lines.append(
        f"| **Micro Precision** | {_fmt_ci(results_rules, 'micro_precision', 'micro_precision_ci_95')} | {_fmt_ci(results_reader, 'micro_precision', 'micro_precision_ci_95')} | {_fmt_ci(results_comb, 'micro_precision', 'micro_precision_ci_95')} |"
    )
    md_lines.append(
        f"| **Micro Recall** | {_fmt_ci(results_rules, 'micro_recall', 'micro_recall_ci_95')} | {_fmt_ci(results_reader, 'micro_recall', 'micro_recall_ci_95')} | {_fmt_ci(results_comb, 'micro_recall', 'micro_recall_ci_95')} |"
    )
    md_lines.append(
        f"| **Micro F1** | {_fmt(results_rules, 'micro_f1')} | {_fmt(results_reader, 'micro_f1')} | {_fmt(results_comb, 'micro_f1')} |"
    )
    md_lines.append(
        f"| **Wrong Energy (Strict)** | {results_rules.get('wrong_energy_strict_count', 'N/A')} | {results_reader.get('wrong_energy_strict_count', 'N/A')} | {results_comb.get('wrong_energy_strict_count', 'N/A')} |"
    )
    md_lines.append(
        f"| **Wrong Energy (Range-based)** | {results_rules.get('wrong_energy_range_count', 'N/A')} | {results_reader.get('wrong_energy_range_count', 'N/A')} | {results_comb.get('wrong_energy_range_count', 'N/A')} |"
    )

    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return output
