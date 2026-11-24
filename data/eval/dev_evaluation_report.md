# Development Dataset Evaluation Report

**Date:** 2026-11-24T11:04:43.840204+00:00
**Model:** none
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
| **Review Rate** | 0.0% | Not run (no API key) | Not run (no API key) |
| **Slot Agreement** | 99.8% [99.1%, 100.0%] | Not run (no API key) | Not run (no API key) |
| **Micro Precision** | 100.0% [94.9%, 100.0%] | Not run (no API key) | Not run (no API key) |
| **Micro Recall** | 100.0% [94.9%, 100.0%] | Not run (no API key) | Not run (no API key) |
| **Micro F1** | 100.0% | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Strict)** | 0 | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Range-based)** | 0 | Not run (no API key) | Not run (no API key) |

### Per-Atom Breakdown (Rules Only)
| Atom | Gold Count | Pred Count | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|
| `very_easy` | 8 | 8 | 100.0% [67.6%, 100.0%] | 100.0% [67.6%, 100.0%] | 100.0% |
| `expert_level` | 7 | 7 | 100.0% [64.6%, 100.0%] | 100.0% [64.6%, 100.0%] | 100.0% |
| `no_basic` | 8 | 8 | 100.0% [67.6%, 100.0%] | 100.0% [67.6%, 100.0%] | 100.0% |
| `ordered_easy_to_hard` | 10 | 10 | 100.0% [72.2%, 100.0%] | 100.0% [72.2%, 100.0%] | 100.0% |
| `hidden_answers` | 11 | 11 | 100.0% [74.1%, 100.0%] | 100.0% [74.1%, 100.0%] | 100.0% |
| `answer_after_each_q` | 8 | 8 | 100.0% [67.6%, 100.0%] | 100.0% [67.6%, 100.0%] | 100.0% |
| `single_correct` | 9 | 9 | 100.0% [70.1%, 100.0%] | 100.0% [70.1%, 100.0%] | 100.0% |
| `multi_interp` | 6 | 6 | 100.0% [61.0%, 100.0%] | 100.0% [61.0%, 100.0%] | 100.0% |
| `confusing_language` | 5 | 5 | 100.0% [56.5%, 100.0%] | 100.0% [56.5%, 100.0%] | 100.0% |

## Split 2: Legacy Authored Prompts (n = 12)

| Metric | Rules Only | Reader Only | Combined System |
|---|---|---|---|
| **Review Rate** | 0.0% | Not run (no API key) | Not run (no API key) |
| **Slot Agreement** | 57.4% [48.0%, 66.3%] | Not run (no API key) | Not run (no API key) |
| **Micro Precision** | 100.0% [67.6%, 100.0%] | Not run (no API key) | Not run (no API key) |
| **Micro Recall** | 34.8% [18.8%, 55.1%] | Not run (no API key) | Not run (no API key) |
| **Micro F1** | 51.6% | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Strict)** | 2 | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Range-based)** | 2 | Not run (no API key) | Not run (no API key) |

### Per-Atom Breakdown (Rules Only)
| Atom | Gold Count | Pred Count | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|
| `very_easy` | 3 | 3 | 100.0% [43.9%, 100.0%] | 100.0% [43.9%, 100.0%] | 100.0% |
| `expert_level` | 1 | 1 | 100.0% [20.6%, 100.0%] | 100.0% [20.6%, 100.0%] | 100.0% |
| `no_basic` | 2 | 1 | 100.0% [20.6%, 100.0%] | 50.0% [9.4%, 90.5%] | 66.7% |
| `ordered_easy_to_hard` | 4 | 1 | 100.0% [20.6%, 100.0%] | 25.0% [4.6%, 69.9%] | 40.0% |
| `hidden_answers` | 3 | 0 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 56.1%] | 0.0% |
| `answer_after_each_q` | 2 | 0 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 65.8%] | 0.0% |
| `single_correct` | 6 | 1 | 100.0% [20.6%, 100.0%] | 16.7% [3.0%, 56.4%] | 28.6% |
| `multi_interp` | 1 | 0 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 79.3%] | 0.0% |
| `confusing_language` | 1 | 1 | 100.0% [20.6%, 100.0%] | 100.0% [20.6%, 100.0%] | 100.0% |

## Combined Development Pool (n = 80)

| Metric | Rules Only | Reader Only | Combined System |
|---|---|---|---|
| **Review Rate** | 0.0% | Not run (no API key) | Not run (no API key) |
| **Slot Agreement** | 93.5% [91.4%, 95.1%] | Not run (no API key) | Not run (no API key) |
| **Micro Precision** | 100.0% [95.4%, 100.0%] | Not run (no API key) | Not run (no API key) |
| **Micro Recall** | 84.2% [75.6%, 90.2%] | Not run (no API key) | Not run (no API key) |
| **Micro F1** | 91.4% | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Strict)** | 2 | Not run (no API key) | Not run (no API key) |
| **Wrong Energy Prompts (Range-based)** | 2 | Not run (no API key) | Not run (no API key) |

### Per-Atom Breakdown (Rules Only)
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
