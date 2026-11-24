"""Evaluation module for honest development and frozen evaluations.

Generates side-by-side evaluation reports for:
1. Rules only
2. Reader only (verified quotes only)
3. Combined system

Strict rules:
- No invented numbers or fake LLM responses in UI or reports.
- If no API key is available, evaluates rules only and prints 'reader not run: no key'.
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

import app.env  # noqa: F401
from app.extractor.combine import combine_rules_and_reader, find_unrecognised_cue_flags
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
    is_dev: bool = False,
    quote_mode: str = "strict",
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

    # Diagnostics accumulation
    calls_made_total = 0
    calls_returned_total = 0
    malformed_total = 0
    retries_total = 0
    reader_failed_total = 0
    claims_asserted_total = 0
    claims_discarded_quote_total = 0
    claims_discarded_negated_total = 0
    claims_accepted_total = 0
    strict_accepted_total = 0
    norm_accepted_total = 0
    latency_total = 0.0
    tokens_total = 0

    per_atom_diag: dict[str, dict[str, int]] = {
        a: {
            "claims_asserted": 0,
            "claims_discarded_quote": 0,
            "claims_discarded_negated": 0,
            "claims_accepted": 0,
            "strict_accepted": 0,
            "norm_accepted": 0,
        }
        for a in ATOM_NAMES
    }
    per_slot_diag: dict[str, dict[str, int]] = {
        s: {
            "claims_asserted": 0,
            "claims_discarded_quote": 0,
            "claims_accepted": 0,
            "strict_accepted": 0,
            "norm_accepted": 0,
        }
        for s in SLOT_NAMES
    }

    # Task 2: Cue flag cost metrics
    missed_atoms_total = 0
    missed_atoms_in_flagged_prompts = 0
    zero_gold_prompts_count = 0
    zero_gold_prompts_flagged = 0
    total_cue_flags_count = 0

    # Task 4: Reader contribution tracking
    reader_added_total = 0
    reader_added_correct_total = 0

    for item in dataset:
        p_id = item["prompt_id"]
        text = item["prompt_text"]
        gold_atoms = set(item["gold_atoms"])
        gold_slots = item["gold_slots"]
        true_energy = evaluate(text, atoms=list(gold_atoms)).energy

        if len(gold_atoms) == 0:
            zero_gold_prompts_count += 1

        if system_name == "rules_only":
            pred_atoms = set(extract_atoms(text))
            pred_slots = score_slots(text)
            pred_energy = evaluate(text, atoms=list(pred_atoms)).energy
            energy_range = [pred_energy, pred_energy]
            review_items = 0

            # Reader contribution tracking: atoms reader added that rules missed
            rule_atoms_for_prompt = set(extract_atoms(text))
            reader_added = pred_atoms - rule_atoms_for_prompt
            reader_added_total += len(reader_added)
            reader_added_correct_total += len(reader_added & gold_atoms)

            # Cue flags for rules only
            flags, _ = find_unrecognised_cue_flags(text, list(pred_atoms), [])
            prompt_cue_flags = flags

        elif system_name == "reader_only":
            reader_out = read_prompt_with_llm(
                text,
                model=model,
                api_key=api_key,
                prompt_id=p_id,
                is_dev=is_dev,
                quote_mode=quote_mode,
            )
            if not reader_out.get("enabled"):
                return {"status": "not_run_no_api_key", "n": n}

            d_diag = reader_out.get("diagnostics", {})
            calls_made_total += d_diag.get("calls_made", 0)
            calls_returned_total += d_diag.get("calls_returned", 0)
            malformed_total += d_diag.get("malformed_responses", 0)
            retries_total += d_diag.get("retries", 0)
            reader_failed_total += d_diag.get("reader_failed", 0)
            claims_asserted_total += d_diag.get("claims_asserted_total", 0)
            claims_discarded_quote_total += d_diag.get("claims_discarded_quote_total", 0)
            claims_discarded_negated_total += d_diag.get("claims_discarded_negated_total", 0)
            claims_accepted_total += d_diag.get("claims_accepted_total", 0)
            strict_accepted_total += d_diag.get("strict_accepted_total", 0)
            norm_accepted_total += d_diag.get("norm_accepted_total", 0)
            latency_total += float(reader_out.get("latency_seconds", 0.0))
            tokens_total += int(reader_out.get("token_usage", {}).get("total_tokens", 0))

            for a in ATOM_NAMES:
                am = d_diag.get("per_atom", {}).get(a, {})
                for k, v in am.items():
                    per_atom_diag[a][k] += v
            for s in SLOT_NAMES:
                sm = d_diag.get("per_slot", {}).get(s, {})
                for k, v in sm.items():
                    per_slot_diag[s][k] += v

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

            # Reader contribution tracking: atoms reader added that rules missed
            rule_atoms_for_prompt = set(extract_atoms(text))
            reader_added = pred_atoms - rule_atoms_for_prompt
            reader_added_total += len(reader_added)
            reader_added_correct_total += len(reader_added & gold_atoms)

            # Cue flags
            highlight_spans = reader_out.get("accepted_spans", [])
            flags, _ = find_unrecognised_cue_flags(text, list(pred_atoms), highlight_spans)
            prompt_cue_flags = flags

        elif system_name == "combined":
            comb = combine_rules_and_reader(
                text,
                prompt_id=p_id,
                api_key=api_key,
                model=model,
                record_reviews=False,
                is_dev=is_dev,
                quote_mode=quote_mode,
            )
            pred_atoms = set(comb.accepted_atoms)
            pred_slots = comb.accepted_slots
            pred_energy = comb.energy
            energy_range = comb.energy_range
            review_items = len(comb.review_atoms) + len(comb.review_slots)
            prompt_cue_flags = comb.cue_flags or []

        else:
            raise ValueError(f"Unknown system name: {system_name}")

        total_review_items += review_items

        # Cue flag dev cost tracking
        total_cue_flags_count += len(prompt_cue_flags)
        if len(gold_atoms) == 0 and len(prompt_cue_flags) > 0:
            zero_gold_prompts_flagged += 1

        missed_in_prompt = gold_atoms - pred_atoms
        missed_atoms_total += len(missed_in_prompt)
        if len(missed_in_prompt) > 0 and len(prompt_cue_flags) > 0:
            missed_atoms_in_flagged_prompts += len(missed_in_prompt)

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

    # Cue cost metrics
    pct_missed_caught = (
        round(float(missed_atoms_in_flagged_prompts) / float(missed_atoms_total), 4) if missed_atoms_total > 0 else 0.0
    )
    false_flag_rate = (
        round(float(zero_gold_prompts_flagged) / float(zero_gold_prompts_count), 4)
        if zero_gold_prompts_count > 0
        else 0.0
    )
    review_load_per_prompt = round(float(total_cue_flags_count) / float(n), 2) if n > 0 else 0.0

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
        "strict_energy_correctness": round(float(n - len(wrong_energy_strict)) / float(n), 4) if n > 0 else 0.0,
        "range_energy_correctness": round(float(n - len(wrong_energy_range)) / float(n), 4) if n > 0 else 0.0,
        "diagnostics": {
            "calls_made": calls_made_total,
            "calls_returned": calls_returned_total,
            "malformed_responses": malformed_total,
            "retries": retries_total,
            "reader_failed": reader_failed_total,
            "claims_asserted_total": claims_asserted_total,
            "claims_discarded_quote_total": claims_discarded_quote_total,
            "claims_discarded_negated_total": claims_discarded_negated_total,
            "claims_accepted_total": claims_accepted_total,
            "strict_accepted_total": strict_accepted_total,
            "norm_accepted_total": norm_accepted_total,
            "latency_seconds_total": round(latency_total, 2),
            "tokens_total": tokens_total,
            "per_atom": per_atom_diag,
            "per_slot": per_slot_diag,
        },
        "cue_flags_cost": {
            "missed_atoms_total": missed_atoms_total,
            "missed_atoms_caught_by_cues": missed_atoms_in_flagged_prompts,
            "pct_missed_caught": pct_missed_caught,
            "zero_gold_prompts_count": zero_gold_prompts_count,
            "zero_gold_prompts_flagged": zero_gold_prompts_flagged,
            "false_flag_rate_on_zero_gold": false_flag_rate,
            "total_cue_flags": total_cue_flags_count,
            "review_load_per_prompt": review_load_per_prompt,
        },
        "reader_contribution": {
            "reader_added_atoms_total": reader_added_total,
            "reader_added_atoms_correct": reader_added_correct_total,
            "reader_added_accuracy": round(float(reader_added_correct_total) / float(reader_added_total), 4)
            if reader_added_total > 0
            else 0.0,
        },
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

    # Reader Diagnostics Section
    comb_diag = results["splits"].get("combined_dev", {}).get("reader_only", {}).get("diagnostics", {})
    if comb_diag:
        lines.append("## Reader Diagnostics")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|---|---|")
        lines.append(f"| Calls Made | {comb_diag.get('calls_made', 0)} |")
        lines.append(f"| Calls Returned | {comb_diag.get('calls_returned', 0)} |")
        lines.append(f"| Malformed Responses | {comb_diag.get('malformed_responses', 0)} |")
        lines.append(f"| Retries | {comb_diag.get('retries', 0)} |")
        lines.append(f"| Reader Failed Prompts | {comb_diag.get('reader_failed', 0)} |")
        lines.append(f"| Claims Asserted | {comb_diag.get('claims_asserted_total', 0)} |")
        lines.append(f"| Claims Discarded (Quote not verbatim) | {comb_diag.get('claims_discarded_quote_total', 0)} |")
        lines.append(f"| Claims Discarded (Negated) | {comb_diag.get('claims_discarded_negated_total', 0)} |")
        lines.append(f"| Claims Accepted | {comb_diag.get('claims_accepted_total', 0)} |")
        lines.append(f"| Total Latency (s) | {comb_diag.get('latency_seconds_total', 0)} |")
        lines.append(f"| Total Tokens | {comb_diag.get('tokens_total', 0)} |")
        lines.append("")

    # Strict vs Normalised Quote Check Comparison
    quote_comp = results.get("quote_mode_comparison")
    if quote_comp:
        lines.append("## Strict vs Normalised Quote Check Comparison")
        lines.append("")
        lines.append("| Quote Mode | Accepted Claims | Precision (95% CI) | Recall (95% CI) | F1 |")
        lines.append("|---|---|---|---|---|")
        st = quote_comp.get("strict", {})
        nm = quote_comp.get("normalized", {})
        st_p = f"{round(st.get('micro_precision', 0) * 100, 1)}% [{round(st.get('micro_precision_ci_95', [0, 0])[0] * 100, 1)}%, {round(st.get('micro_precision_ci_95', [0, 0])[1] * 100, 1)}%]"
        st_r = f"{round(st.get('micro_recall', 0) * 100, 1)}% [{round(st.get('micro_recall_ci_95', [0, 0])[0] * 100, 1)}%, {round(st.get('micro_recall_ci_95', [0, 0])[1] * 100, 1)}%]"
        nm_p = f"{round(nm.get('micro_precision', 0) * 100, 1)}% [{round(nm.get('micro_precision_ci_95', [0, 0])[0] * 100, 1)}%, {round(nm.get('micro_precision_ci_95', [0, 0])[1] * 100, 1)}%]"
        nm_r = f"{round(nm.get('micro_recall', 0) * 100, 1)}% [{round(nm.get('micro_recall_ci_95', [0, 0])[0] * 100, 1)}%, {round(nm.get('micro_recall_ci_95', [0, 0])[1] * 100, 1)}%]"
        lines.append(
            f"| Strict | {st.get('claims_accepted', 0)} | {st_p} | {st_r} | {round(st.get('micro_f1', 0) * 100, 1)}% |"
        )
        lines.append(
            f"| Normalised | {nm.get('claims_accepted', 0)} | {nm_p} | {nm_r} | {round(nm.get('micro_f1', 0) * 100, 1)}% |"
        )
        lines.append("")
        lines.append(
            f"**Policy Rule Verdict:** {quote_comp.get('verdict', 'Normalised check retained only if it adds accepted claims without reducing precision.')}"
        )
        lines.append("")

    # Cue Flags Cost Analysis
    cue_cost = results["splits"].get("combined_dev", {}).get("combined", {}).get("cue_flags_cost")
    if cue_cost:
        lines.append("## Unrecognised-Cue Flags Cost Analysis on Development Pool")
        lines.append("")
        lines.append("| Metric | Measured Value |")
        lines.append("|---|---|")
        lines.append(
            f"| Missed gold requirements caught by cue flags | {round(cue_cost['pct_missed_caught'] * 100, 1)}% ({cue_cost['missed_atoms_caught_by_cues']} / {cue_cost['missed_atoms_total']}) |"
        )
        lines.append(
            f"| False-flag rate on zero-gold prompts | {round(cue_cost['false_flag_rate_on_zero_gold'] * 100, 1)}% ({cue_cost['zero_gold_prompts_flagged']} / {cue_cost['zero_gold_prompts_count']}) |"
        )
        lines.append(
            f"| Review load per prompt | {cue_cost['review_load_per_prompt']} flags/prompt (total {cue_cost['total_cue_flags']} flags across {results['provenance']['total_prompts']} prompts) |"
        )
        lines.append("")

    # Per-Atom Breakdown
    r_only_comb = results["splits"].get("combined_dev", {}).get("rules_only", {})
    lines.append("### Per-Atom Breakdown (Rules Only on Combined Dev Pool)")
    lines.append("| Atom | Gold Count | Pred Count | Precision (95% CI) | Recall (95% CI) | F1 |")
    lines.append("|---|---|---|---|---|---|")
    if r_only_comb.get("status") == "success":
        for a in ATOM_NAMES:
            am = r_only_comb["per_atom"][a]
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
        print("reader not run: no key")

    key = (
        (os.environ.get("STUDY_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    is_groq = bool((os.environ.get("GROQ_API_KEY") or "").strip()) and not os.environ.get("STUDY_API_BASE")
    default_model = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"
    model = os.environ.get("STUDY_MODEL") or default_model

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

        splits_data["dev_68"][sys_name] = evaluate_system_on_dataset(
            dev_68, sys_name, api_key=key, model=model, is_dev=True, quote_mode="strict"
        )
        splits_data["dev_12"][sys_name] = evaluate_system_on_dataset(
            dev_12, sys_name, api_key=key, model=model, is_dev=True, quote_mode="strict"
        )
        splits_data["combined_dev"][sys_name] = evaluate_system_on_dataset(
            combined_all, sys_name, api_key=key, model=model, is_dev=True, quote_mode="strict"
        )

    # Task 1: Strict vs Normalised quote check comparison on dev data
    quote_mode_comparison: dict[str, Any] | None = None
    if reader_active:
        strict_eval = splits_data["combined_dev"]["reader_only"]
        norm_eval = evaluate_system_on_dataset(
            combined_all, "reader_only", api_key=key, model=model, is_dev=True, quote_mode="normalized"
        )
        st_claims = strict_eval.get("diagnostics", {}).get("claims_accepted_total", 0)
        nm_claims = norm_eval.get("diagnostics", {}).get("claims_accepted_total", 0)
        st_prec = strict_eval.get("micro_precision", 0.0)
        nm_prec = norm_eval.get("micro_precision", 0.0)

        adds_claims = nm_claims > st_claims
        preserves_prec = nm_prec >= st_prec
        adopted = adds_claims and preserves_prec

        quote_mode_comparison = {
            "strict": {
                "claims_accepted": st_claims,
                "micro_precision": st_prec,
                "micro_precision_ci_95": strict_eval.get("micro_precision_ci_95"),
                "micro_recall": strict_eval.get("micro_recall", 0.0),
                "micro_recall_ci_95": strict_eval.get("micro_recall_ci_95"),
                "micro_f1": strict_eval.get("micro_f1", 0.0),
            },
            "normalized": {
                "claims_accepted": nm_claims,
                "micro_precision": nm_prec,
                "micro_precision_ci_95": norm_eval.get("micro_precision_ci_95"),
                "micro_recall": norm_eval.get("micro_recall", 0.0),
                "micro_recall_ci_95": norm_eval.get("micro_recall_ci_95"),
                "micro_f1": norm_eval.get("micro_f1", 0.0),
            },
            "adds_claims": adds_claims,
            "preserves_precision": preserves_prec,
            "adopted": adopted,
            "verdict": (
                "Normalised check adds accepted claims without reducing precision. Adopted."
                if adopted
                else "Normalised check does not add claims or reduces precision. Strict check retained."
            ),
        }

    output: dict[str, Any] = {
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
        "quote_mode_comparison": quote_mode_comparison,
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


def run_frozen_eval(force: bool = False, set_name: str | None = None) -> dict[str, Any]:
    """Execute frozen evaluation (Task 4 & Task 5).

    Strict rules:
    - If set_name is None, check legacy v2 folder.
    - If target directory is missing:
      - For v3: prints that it does not exist and does nothing without raising unhandled exception.
      - For v2: raises FileNotFoundError (legacy test suite expectation).
    - Verifies line-ending normalised hash.
    - Runs once; refuses second run unless force=True.
    - Records forced runs in report.
    - Never write prompt text of frozen prompts to logs or diagnostics.
    """
    if set_name is None:
        target_set = "v2"
        target_dir = FROZEN_V2_DIR
        hash_file = FROZEN_V2_HASH_FILE
        if not target_dir.exists():
            raise FileNotFoundError(
                f"frozen_v2 directory not found at {FROZEN_V2_DIR}. "
                "Please provide data/gold/frozen_v2/ to run frozen evaluation."
            )
    else:
        target_set = set_name.lower().replace("frozen_", "").strip()
        target_dir = ROOT / "data" / "gold" / f"frozen_{target_set}"
        hash_file = ROOT / "data" / "gold" / f"FROZEN_{target_set.upper()}_SHA256.txt"
        if not target_dir.exists():
            msg = f"data/gold/frozen_{target_set}/ does not exist. Skipping frozen evaluation."
            print(msg)
            return {"status": "folder_absent", "message": msg}

    frozen_files = list(target_dir.glob("*.csv")) + list(target_dir.glob("*.json"))
    if not frozen_files:
        raise FileNotFoundError(f"No CSV or JSON files found in {target_dir}.")

    target_file = frozen_files[0]

    # Verify SHA-256 against hash file if exists
    dir_hash = target_dir / f"FROZEN_{target_set.upper()}_SHA256.txt"
    active_hash_file = dir_hash if dir_hash.exists() else hash_file
    if active_hash_file.exists():
        expected_hash = active_hash_file.read_text(encoding="utf-8").strip()
        actual_hash = compute_normalized_file_sha256(target_file)
        if actual_hash.casefold() != expected_hash.casefold():
            raise ValueError(
                f"Frozen set hash mismatch for {target_file.name}. Expected: {expected_hash}, Actual: {actual_hash}"
            )

    report_json_path = EVAL_DIR / f"frozen_{target_set}_report.json"
    report_md_path = EVAL_DIR / f"frozen_{target_set}_report.md"

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
    key = (
        (os.environ.get("STUDY_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    is_groq = bool((os.environ.get("GROQ_API_KEY") or "").strip()) and not os.environ.get("STUDY_API_BASE")
    default_model = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"
    model = os.environ.get("STUDY_MODEL") or default_model

    # Frozen evaluation: is_dev=False (strictly no discard logging on frozen data)
    results_rules = evaluate_system_on_dataset(frozen_items, "rules_only", is_dev=False)
    results_reader = (
        evaluate_system_on_dataset(frozen_items, "reader_only", api_key=key, model=model, is_dev=False)
        if reader_active
        else {"status": "not_run_no_api_key"}
    )
    results_comb = (
        evaluate_system_on_dataset(frozen_items, "combined", api_key=key, model=model, is_dev=False)
        if reader_active
        else {"status": "not_run_no_api_key"}
    )

    output: dict[str, Any] = {
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
            "statement": f"The frozen set ({target_set}) is authored, not real faculty data.",
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
    md_lines = [
        f"# Frozen Test Set Evaluation Report ({target_set})",
        "",
        f"**Date:** {output['metadata']['date']}",
        f"**Model:** {output['metadata']['model']}",
        f"**Rubric Version:** {output['metadata']['rubric_version']}",
        f"**Schema Version:** {output['metadata']['schema_version']}",
        f"**Execution:** {'Forced run #' + str(prior_forced_count) if force else 'Initial single run'}",
        "",
        "> **DATA PROVENANCE STATEMENT**:",
        f"> {output['provenance']['statement']}",
        f"> Sample size: n = {output['provenance']['n_prompts']} prompts.",
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
        f"| **Strict Energy Correctness** | {_fmt(results_rules, 'strict_energy_correctness')} | {_fmt(results_reader, 'strict_energy_correctness')} | {_fmt(results_comb, 'strict_energy_correctness')} |"
    )
    md_lines.append(
        f"| **Range Energy Correctness** | {_fmt(results_rules, 'range_energy_correctness')} | {_fmt(results_reader, 'range_energy_correctness')} | {_fmt(results_comb, 'range_energy_correctness')} |"
    )
    md_lines.append(
        f"| **Wrong Energy (Strict)** | {results_rules.get('wrong_energy_strict_count', 'N/A')} | {results_reader.get('wrong_energy_strict_count', 'N/A')} | {results_comb.get('wrong_energy_strict_count', 'N/A')} |"
    )
    md_lines.append(
        f"| **Wrong Energy (Range-based)** | {results_rules.get('wrong_energy_range_count', 'N/A')} | {results_reader.get('wrong_energy_range_count', 'N/A')} | {results_comb.get('wrong_energy_range_count', 'N/A')} |"
    )
    md_lines.append("")

    # Reader Diagnostics on Frozen Set
    rd_diag = results_reader.get("diagnostics", {})
    if rd_diag:
        md_lines.append("## Reader Diagnostics on Frozen Set")
        md_lines.append("")
        md_lines.append("| Metric | Count |")
        md_lines.append("|---|---|")
        md_lines.append(f"| Calls Made | {rd_diag.get('calls_made', 0)} |")
        md_lines.append(f"| Calls Returned | {rd_diag.get('calls_returned', 0)} |")
        md_lines.append(f"| Malformed Responses | {rd_diag.get('malformed_responses', 0)} |")
        md_lines.append(f"| Retries | {rd_diag.get('retries', 0)} |")
        md_lines.append(f"| Reader Failed Prompts | {rd_diag.get('reader_failed', 0)} |")
        md_lines.append(f"| Claims Asserted | {rd_diag.get('claims_asserted_total', 0)} |")
        md_lines.append(f"| Claims Discarded (Quote not verbatim) | {rd_diag.get('claims_discarded_quote_total', 0)} |")
        md_lines.append(f"| Claims Discarded (Negated) | {rd_diag.get('claims_discarded_negated_total', 0)} |")
        md_lines.append(f"| Claims Accepted | {rd_diag.get('claims_accepted_total', 0)} |")
        md_lines.append(f"| Total Latency (s) | {rd_diag.get('latency_seconds_total', 0)} |")
        md_lines.append(f"| Total Tokens | {rd_diag.get('tokens_total', 0)} |")
        md_lines.append("")

    # Cue Flags on Frozen Set
    cue_cost = results_comb.get("cue_flags_cost", {})
    if cue_cost:
        md_lines.append("## Unrecognised-Cue Flags on Frozen Set")
        md_lines.append("")
        md_lines.append("| Metric | Count |")
        md_lines.append("|---|---|")
        md_lines.append(
            f"| Missed gold requirements caught | {round(cue_cost.get('pct_missed_caught', 0) * 100, 1)}% ({cue_cost.get('missed_atoms_caught_by_cues', 0)} / {cue_cost.get('missed_atoms_total', 0)}) |"
        )
        md_lines.append(
            f"| False-flag rate on zero-gold prompts | {round(cue_cost.get('false_flag_rate_on_zero_gold', 0) * 100, 1)}% ({cue_cost.get('zero_gold_prompts_flagged', 0)} / {cue_cost.get('zero_gold_prompts_count', 0)}) |"
        )
        md_lines.append(f"| Total Cue Flags | {cue_cost.get('total_cue_flags', 0)} |")
        md_lines.append(f"| Review load per prompt | {cue_cost.get('review_load_per_prompt', 0)} |")
        md_lines.append("")

    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return output


def run_model_sweep(models: list[str]) -> dict[str, Any]:
    """Execute model sweep across specified models on dev set only (Task 4).

    Strict rules:
    - Must refuse to run on any frozen folder.
    - If no API key is set, prints 'reader not run: no key' and exits cleanly.
    - Evaluates call counts, malformed counts, retry counts, discard counts by reason.
    - Evaluates precision, recall, F1 against dev labels, latency, token usage, review rate.
    - Writes data/eval/model_sweep_dev.md.
    """
    # 1. Check for frozen targets
    for m in models:
        if "frozen" in m.lower():
            raise ValueError(f"Model sweep refused: sweep runs strictly on development data, not frozen data ({m}).")

    # 2. Check API key
    if not is_reader_enabled():
        print("reader not run: no key")
        return {"status": "not_run_no_key", "message": "reader not run: no key"}

    dev_68, dev_12 = load_dev_prompts()
    dev_all = dev_68 + dev_12
    n = len(dev_all)

    key = (
        (os.environ.get("STUDY_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )

    clean_models = [m.strip() for m in models if m.strip()]
    if not clean_models:
        print("No models provided for sweep.")
        return {"status": "no_models"}

    sweep_results: dict[str, Any] = {}
    for mod_name in clean_models:
        rd_res = evaluate_system_on_dataset(dev_all, "reader_only", api_key=key, model=mod_name, is_dev=True)
        comb_res = evaluate_system_on_dataset(dev_all, "combined", api_key=key, model=mod_name, is_dev=True)
        sweep_results[mod_name] = {
            "reader_only": rd_res,
            "combined": comb_res,
        }

    # Generate Markdown report data/eval/model_sweep_dev.md
    md_lines: list[str] = [
        "# Model Sweep Evaluation Report (Development Dataset)",
        "",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Dataset:** Combined Development Pool (n = {n} prompts: 68 authored + 12 legacy synthetic)",
        f"**Evaluated Models:** {', '.join(clean_models)}",
        "",
        "> **DATA PROVENANCE STATEMENT**:",
        "> This model sweep was conducted strictly on the development dataset pool.",
        "> No frozen test data was accessed or evaluated.",
        "",
        "## Performance Comparison (Reader Only vs Combined)",
        "| Model | Reader F1 [95% CI] | Combined F1 [95% CI] | Slot Agreement (Combined) | Review Rate |",
        "|---|---|---|---|---|",
    ]

    for mod_name in clean_models:
        mdata = sweep_results[mod_name]
        rd = mdata["reader_only"]
        cb = mdata["combined"]
        rd_f1 = f"{round(rd.get('micro_f1', 0) * 100, 1)}% [{round(rd.get('micro_recall_ci_95', [0, 0])[0] * 100, 1)}%, {round(rd.get('micro_precision_ci_95', [0, 0])[1] * 100, 1)}%]"
        cb_f1 = f"{round(cb.get('micro_f1', 0) * 100, 1)}% [{round(cb.get('micro_recall_ci_95', [0, 0])[0] * 100, 1)}%, {round(cb.get('micro_precision_ci_95', [0, 0])[1] * 100, 1)}%]"
        cb_slot = f"{round(cb.get('slot_agreement', 0) * 100, 1)}% [{round(cb.get('slot_agreement_ci_95', [0, 0])[0] * 100, 1)}%, {round(cb.get('slot_agreement_ci_95', [0, 0])[1] * 100, 1)}%]"
        rev_rate = f"{round(cb.get('review_rate', 0) * 100, 1)}%"
        md_lines.append(f"| `{mod_name}` | {rd_f1} | {cb_f1} | {cb_slot} | {rev_rate} |")

    md_lines.extend(
        [
            "",
            "## Reader Execution and Diagnostics Comparison",
            "| Model | Calls Made | Returned | Malformed | Retries | Failed Prompts | Latency (s) | Tokens |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for mod_name in clean_models:
        rd = sweep_results[mod_name]["reader_only"]
        diag = rd.get("diagnostics", {})
        md_lines.append(
            f"| `{mod_name}` | {diag.get('calls_made', 0)} | {diag.get('calls_returned', 0)} | {diag.get('malformed_responses', 0)} | {diag.get('retries', 0)} | {diag.get('reader_failed', 0)} | {diag.get('latency_seconds_total', 0)} | {diag.get('tokens_total', 0)} |"
        )

    md_lines.extend(
        [
            "",
            "## Claim Discards by Reason",
            "| Model | Claims Asserted | Discarded (Quote not verbatim) | Discarded (Negated) | Claims Accepted |",
            "|---|---|---|---|---|",
        ]
    )

    for mod_name in clean_models:
        rd = sweep_results[mod_name]["reader_only"]
        diag = rd.get("diagnostics", {})
        md_lines.append(
            f"| `{mod_name}` | {diag.get('claims_asserted_total', 0)} | {diag.get('claims_discarded_quote_total', 0)} | {diag.get('claims_discarded_negated_total', 0)} | {diag.get('claims_accepted_total', 0)} |"
        )

    md_lines.extend(
        [
            "",
            "## Contribution to Combined System",
            "| Model | Atoms Added (Rules Missed) | Added Atoms Correct | Reader Precision on Added Atoms |",
            "|---|---|---|---|",
        ]
    )

    for mod_name in clean_models:
        rd = sweep_results[mod_name]["reader_only"]
        contrib = rd.get("reader_contribution", {})
        added = contrib.get("reader_added_atoms_total", 0)
        corr = contrib.get("reader_added_atoms_correct", 0)
        acc_str = f"{round(contrib.get('reader_added_accuracy', 0.0) * 100, 1)}%" if added > 0 else "N/A"
        md_lines.append(f"| `{mod_name}` | {added} | {corr} | {acc_str} |")

    md_lines.append("")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sweep_md_path = EVAL_DIR / "model_sweep_dev.md"
    sweep_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Model sweep report generated: {sweep_md_path}")

    return sweep_results
