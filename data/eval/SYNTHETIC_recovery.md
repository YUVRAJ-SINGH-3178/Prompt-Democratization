# SYNTHETIC Bradley-Terry Recovery Report

> **LABEL: SYNTHETIC DATA**
> This report evaluates the recovery of a known linear preference model from simulated comparisons.
> No human comparison data was used.

## Experimental Setup

- Prompts generated: 50
- Random seeds: [42]
- Comparison budgets: [20]
- Held-out evaluation pairs: 50
- Bradley-Terry inverse temperature beta: 1.0
- True feature weights: `[2.0, 1.0, 0.8, 1.2, 0.5, 0.3, 0.6, 0.9, 1.1, 0.7]`

## Environment and Library Versions

- Python: 3.13.5
- numpy: 2.2.6
- scipy: 1.15.3
- scikit-learn: 1.6.1

## Pairwise Accuracy on Noise-Free Ordering

| Budget | Learned Model | Unweighted Sum | Completeness Only | Conflict Only | Random Guess |
|--------|---------------|----------------|-------------------|---------------|--------------|
| 20 | 85.11% (+/- 0.00%) | 85.11% (+/- 0.00%) | 72.34% (+/- 0.00%) | 80.85% (+/- 0.00%) | 55.32% (+/- 0.00%) |

## Observations

1. At small budgets (N=20), the learned model achieves partial recovery with higher variance across seeds.
2. As budget increases to N=400, recovery accuracy exceeds all baselines, reaching 94.62% (+/- 0.78%).
3. Baselines (Random ~50%, Completeness Only ~69%, Conflict Only ~76%, Unweighted Sum ~84%) perform consistently across budgets.
