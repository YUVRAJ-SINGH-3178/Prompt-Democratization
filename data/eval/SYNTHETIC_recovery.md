# SYNTHETIC Bradley-Terry Recovery Report

> **LABEL: SYNTHETIC DATA**
> This report evaluates the recovery of a known linear preference model from simulated comparisons.
> No human comparison data was used. Synthetic results stay labelled and separate from real data.

## Experimental Setup

- Prompts generated: 3000
- Random seeds: [42, 43, 44, 45, 46]
- Comparison budgets: [20, 50, 100, 200, 400]
- Held-out evaluation pairs: 5000
- Bradley-Terry inverse temperature beta: 1.0
- True feature weights: `[2.0, 1.0, 0.8, 1.2, 0.5, 0.3, 0.6, 0.9, 1.1, 0.7]`

## Environment and Library Versions

- Python: 3.13.5
- numpy: 2.2.6
- scipy: 1.15.3
- scikit-learn: 1.6.1

## Pairwise Accuracy on Noise-Free Ordering (Linear Case)

| Budget | Learned Model | Unweighted Sum | Completeness Only | Conflict Only | Random Guess |
|--------|---------------|----------------|-------------------|---------------|--------------|
| 20 | 80.96% (+/- 2.48%) | 84.08% (+/- 0.34%) | 69.01% (+/- 0.66%) | 76.16% (+/- 0.27%) | 49.54% (+/- 0.27%) |
| 50 | 82.84% (+/- 4.76%) | 84.08% (+/- 0.34%) | 69.01% (+/- 0.66%) | 76.16% (+/- 0.27%) | 49.96% (+/- 1.21%) |
| 100 | 88.06% (+/- 1.65%) | 84.08% (+/- 0.34%) | 69.01% (+/- 0.66%) | 76.16% (+/- 0.27%) | 49.82% (+/- 0.64%) |
| 200 | 91.94% (+/- 2.08%) | 84.08% (+/- 0.34%) | 69.01% (+/- 0.66%) | 76.16% (+/- 0.27%) | 49.50% (+/- 0.69%) |
| 400 | 94.79% (+/- 1.08%) | 84.08% (+/- 0.34%) | 69.01% (+/- 0.66%) | 76.16% (+/- 0.27%) | 49.86% (+/- 0.30%) |

## Observations

1. At small budgets (N=20), the learned model achieves mean accuracy 80.96% (+/- 2.48%) across seeds.
2. As budget increases to N=400, recovery accuracy exceeds all baselines, reaching 94.79% (+/- 1.08%).
3. Baselines at N=400: Random 49.86% (+/- 0.30%), Completeness Only 69.01% (+/- 0.66%), Conflict Only 76.16% (+/- 0.27%), Unweighted Sum 84.08% (+/- 0.34%).

## Model Misspecification: Non-linear Interaction Case

> **NOTE:** This checks the estimator under model misspecification and says nothing about real faculty.

| Budget | Learned Model | Unweighted Sum | Completeness Only | Conflict Only | Random Guess |
|--------|---------------|----------------|-------------------|---------------|--------------|
| 20 | 77.28% (+/- 4.26%) | 84.59% (+/- 0.39%) | 69.87% (+/- 0.69%) | 75.36% (+/- 0.29%) | 50.28% (+/- 0.76%) |
| 50 | 82.89% (+/- 3.74%) | 84.59% (+/- 0.39%) | 69.87% (+/- 0.69%) | 75.36% (+/- 0.29%) | 50.04% (+/- 0.49%) |
| 100 | 88.79% (+/- 1.39%) | 84.59% (+/- 0.39%) | 69.87% (+/- 0.69%) | 75.36% (+/- 0.29%) | 50.70% (+/- 0.38%) |
| 200 | 91.38% (+/- 2.50%) | 84.59% (+/- 0.39%) | 69.87% (+/- 0.69%) | 75.36% (+/- 0.29%) | 49.69% (+/- 0.60%) |
| 400 | 94.65% (+/- 0.79%) | 84.59% (+/- 0.39%) | 69.87% (+/- 0.69%) | 75.36% (+/- 0.29%) | 49.28% (+/- 1.27%) |

## Claims Verification (eval/claims.yaml)

- Decision rule: `learned_model_exceeds_all_baselines_after_holm`
- Verdict: **WIN** (synthetic: not a real-data result)
- Alpha threshold: 0.05
  - random: p_raw = 0.0000e+00, threshold = 1.2500e-02 -> PASS
  - completeness_only: p_raw = 0.0000e+00, threshold = 1.6667e-02 -> PASS
  - conflict_only: p_raw = 0.0000e+00, threshold = 2.5000e-02 -> PASS
  - repository_comparator: p_raw = 0.0000e+00, threshold = 5.0000e-02 -> PASS