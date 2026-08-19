# Experiment Matrix

No row below is a result until the command has actually been run and its output artifact is preserved.

## Forecasting baselines

| Method | Purpose | Required data | Metrics |
|---|---|---|---|
| Persistence | minimum sanity baseline | real sequence | MAE, RMSE |
| Per-tile linear trend | cheap extrapolation baseline | real sequence | MAE, RMSE |
| SAM-WM baseline | learned action-conditioned world model | real sequence | MAE, RMSE, 90% coverage/width |

## Required robustness evaluations

1. Chronological holdout with purge gap to prevent overlapping-window leakage.
2. Real extreme-temperature tails: q95 and q99 of held-out target temperatures.
3. Horizon-wise error and interval calibration.
4. Spatial holdout once more than one geographically separated AOI exists.
5. Unseen-city holdout only after at least two real cities have been collected.
6. Weather-regime OOD only after real weather forcing variables are added.
7. Action-support holdout using real non-zero intervention logs.
8. Causal replay with treated/control areas and pre-trend checks.
9. Runtime, parameter count, memory and latency on Mac M1 and Kaggle GPU.

## Action/counterfactual evaluation

Counterfactual intervention quality must not be inferred from passive temperature prediction alone. A credible paper needs real action variation. For each intervention type report:

- amount of historical support;
- action-effect error where a real treated/control estimate exists;
- interval coverage;
- abstention rate;
- risk-coverage or error-vs-support curve;
- conflict cases where mechanism prior and data disagree.

## Ablations

- no action conditioning;
- no spatial coupling;
- no JEPA latent target;
- no support calibration;
- point-SAM vs partial-identification SAM;
- Transformer spatial mixing vs later sparse graph mixing;
- with/without real weather forcing once available.
