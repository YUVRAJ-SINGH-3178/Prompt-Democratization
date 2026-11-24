# Development Dataset Evaluation Report

**Date:** 2026-11-24T13:21:08.278424+00:00
**Model:** openai/gpt-oss-20b
**Rubric Version:** 1.0.0
**Schema Version:** 1.0.0
**Environment:** Python 3.13.5, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1

> **DATA PROVENANCE STATEMENT**:
> Total prompts: 80 (Split 1: 68 authored prompts, Split 2: 12 legacy synthetic prompts).
> Source label: authored_by_builder. Synthetic label: 1.
> These prompts were authored by the builder for tuning and development. They are not real faculty data.

## Split 1: Authored Development Prompts (n = 68)

| Metric | Rules Only | Reader Only | Combined System |
|---|---|---|---|
| **Review Rate** | 0.0% | 0.0% | 1.5% |
| **Slot Agreement** | 99.8% [99.1%, 100.0%] | 58.3% [54.4%, 62.2%] | 98.4% [97.0%, 99.1%] |
| **Micro Precision** | 100.0% [94.9%, 100.0%] | 82.8% [65.5%, 92.4%] | 100.0% [94.7%, 100.0%] |
| **Micro Recall** | 100.0% [94.9%, 100.0%] | 33.3% [23.5%, 44.8%] | 94.4% [86.6%, 97.8%] |
| **Micro F1** | 100.0% | 47.5% | 97.1% |
| **Wrong Energy Prompts (Strict)** | 0 | 11 | 0 |
| **Wrong Energy Prompts (Range-based)** | 0 | 11 | 0 |

## Split 2: Legacy Authored Prompts (n = 12)

| Metric | Rules Only | Reader Only | Combined System |
|---|---|---|---|
| **Review Rate** | 0.0% | 0.0% | 6.5% |
| **Slot Agreement** | 57.4% [48.0%, 66.3%] | 41.7% [32.8%, 51.1%] | 58.3% [48.9%, 67.2%] |
| **Micro Precision** | 100.0% [67.6%, 100.0%] | 80.0% [37.5%, 96.4%] | 100.0% [67.6%, 100.0%] |
| **Micro Recall** | 34.8% [18.8%, 55.1%] | 17.4% [7.0%, 37.1%] | 34.8% [18.8%, 55.1%] |
| **Micro F1** | 51.6% | 28.6% | 51.6% |
| **Wrong Energy Prompts (Strict)** | 2 | 4 | 2 |
| **Wrong Energy Prompts (Range-based)** | 2 | 4 | 2 |

## Combined Development Pool (n = 80)

| Metric | Rules Only | Reader Only | Combined System |
|---|---|---|---|
| **Review Rate** | 0.0% | 0.0% | 2.2% |
| **Slot Agreement** | 93.5% [91.4%, 95.1%] | 55.8% [52.2%, 59.4%] | 92.4% [90.2%, 94.1%] |
| **Micro Precision** | 100.0% [95.4%, 100.0%] | 82.3% [66.5%, 91.6%] | 100.0% [95.2%, 100.0%] |
| **Micro Recall** | 84.2% [75.6%, 90.2%] | 29.5% [21.2%, 39.3%] | 80.0% [70.9%, 86.8%] |
| **Micro F1** | 91.4% | 43.4% | 88.9% |
| **Wrong Energy Prompts (Strict)** | 2 | 15 | 2 |
| **Wrong Energy Prompts (Range-based)** | 2 | 15 | 2 |

## Reader Diagnostics

| Metric | Count |
|---|---|
| Calls Made | 100 |
| Calls Returned | 0 |
| Malformed Responses | 0 |
| Retries | 50 |
| Reader Failed Prompts | 50 |
| Claims Asserted | 210 |
| Claims Discarded (Quote not verbatim) | 0 |
| Claims Discarded (Negated) | 4 |
| Claims Accepted | 206 |
| Total Latency (s) | 94.35 |
| Total Tokens | 178134 |

## Strict vs Normalised Quote Check Comparison

| Quote Mode | Accepted Claims | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|
| Strict | 206 | 82.3% [66.5%, 91.6%] | 29.5% [21.2%, 39.3%] | 43.4% |
| Normalised | 206 | 82.3% [66.5%, 91.6%] | 29.5% [21.2%, 39.3%] | 43.4% |

**Policy Rule Verdict:** Normalised check does not add claims or reduces precision. Strict check retained.

## Unrecognised-Cue Flags Cost Analysis on Development Pool

| Metric | Measured Value |
|---|---|
| Missed gold requirements caught by cue flags | 68.0% (17 / 25) |
| False-flag rate on zero-gold prompts | 33.3% (6 / 18) |
| Review load per prompt | 0.38 flags/prompt (total 30 flags across 80 prompts) |

### Per-Atom Breakdown (Rules Only on Combined Dev Pool)
| Atom | Gold Count | Pred Count | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|
| `very_easy` | 11 | 11 | 100.0% [74.1%, 100.0%] | 100.0% [74.1%, 100.0%] | 100.0% |
| `expert_level` | 8 | 8 | 100.0% [67.6%, 100.0%] | 100.0% [67.6%, 100.0%] | 100.0% |
| `no_basic` | 10 | 9 | 100.0% [70.1%, 100.0%] | 90.0% [59.6%, 98.2%] | 94.7% |
| `ordered_easy_to_hard` | 14 | 11 | 100.0% [74.1%, 100.0%] | 78.6% [52.4%, 92.4%] | 88.0% |
| `hidden_answers` | 14 | 11 | 100.0% [74.1%, 100.0%] | 78.6% [52.4%, 92.4%] | 88.0% |
| `answer_after_each_q` | 10 | 8 | 100.0% [67.6%, 100.0%] | 80.0% [49.0%, 94.3%] | 88.9% |
| `single_correct` | 15 | 10 | 100.0% [72.2%, 100.0%] | 66.7% [41.7%, 84.8%] | 80.0% |
| `multi_interp` | 7 | 6 | 100.0% [61.0%, 100.0%] | 85.7% [48.7%, 97.4%] | 92.3% |
| `confusing_language` | 6 | 6 | 100.0% [61.0%, 100.0%] | 100.0% [61.0%, 100.0%] | 100.0% |
