# Model Sweep Evaluation Report (Development Dataset)

**Date:** 2026-11-24T13:12:42.116356+00:00
**Dataset:** Combined Development Pool (n = 80 prompts: 68 authored + 12 legacy synthetic)
**Evaluated Models:** openai/gpt-oss-20b, openai/gpt-oss-120b

> **DATA PROVENANCE STATEMENT**:
> This model sweep was conducted strictly on the development dataset pool.
> No frozen test data was accessed or evaluated.

## Performance Comparison (Reader Only vs Combined)
| Model | Reader F1 [95% CI] | Combined F1 [95% CI] | Slot Agreement (Combined) | Review Rate |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | 42.9% [20.3%, 94.9%] | 88.9% [70.9%, 100.0%] | 92.4% [90.2%, 94.1%] | 2.1% |
| `openai/gpt-oss-120b` | 0.0% [0.0%, 0.0%] | 91.4% [75.6%, 100.0%] | 93.5% [91.4%, 95.1%] | 0.0% |

## Reader Execution and Diagnostics Comparison
| Model | Calls Made | Returned | Malformed | Retries | Failed Prompts | Latency (s) | Tokens |
|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | 102 | 0 | 0 | 51 | 51 | 91.73 | 172055 |
| `openai/gpt-oss-120b` | 160 | 0 | 0 | 80 | 80 | 0.0 | 0 |

## Claim Discards by Reason
| Model | Claims Asserted | Discarded (Quote not verbatim) | Discarded (Negated) | Claims Accepted |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | 198 | 0 | 4 | 194 |
| `openai/gpt-oss-120b` | 0 | 0 | 0 | 0 |

## Contribution to Combined System
| Model | Atoms Added (Rules Missed) | Added Atoms Correct | Reader Precision on Added Atoms |
|---|---|---|---|
| `openai/gpt-oss-20b` | 6 | 2 | 33.3% |
| `openai/gpt-oss-120b` | 0 | 0 | N/A |
