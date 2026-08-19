# SAM-WM Research Status — v0.6

This file is intentionally conservative. It records what exists and what is still missing.

## Implemented and locally testable

- Real-only FortyGuard evidence ingestion with provenance hashing.
- Fixed-grid timeline reconstruction from recorded real TCM heatmaps.
- Action-conditioned JEPA-style spatiotemporal world-model baseline.
- Explicit model-readiness gate; no checkpoint means `MODEL_NOT_READY`.
- Support-aware counterfactual gate and conformal interval artifact interface.
- Baseline-vs-intervention future rendering, only after a validated checkpoint exists.
- Cheap forecasting baselines: persistence and per-tile linear trend.
- Real extreme-temperature-tail evaluation (`q95`, `q99`) for a trained checkpoint.
- Runtime/parameter benchmark on an actual real sequence.
- Browser pipeline-readiness panel and 3D camera/basemap corrections.

## NOT completed yet

- No real training run has been performed for this local project state.
- No Kaggle training/evaluation artifact is bundled.
- No empirical result may be called SOTA or better than any published method.
- No real non-zero intervention dataset is bundled for shade/tree/cool-pavement actions.
- Therefore causal intervention effects are not established yet.
- No unseen-city or unseen-climate OOD result exists yet.
- No severe-weather forcing variables are present in the current minimal sequence schema.
- No validated global deployment exists. The architecture is designed to be extensible, not globally validated.
- No physical actuator has been deployed by this software. Physical cooling requires a real-world intervention.

## Immediate empirical gates

1. Collect a repeated, non-empty real FortyGuard TCM sequence on one U.S. AOI.
2. Build the sequence bundle and verify hashes/grid signature.
3. Run persistence + linear-trend baselines.
4. Train the current compact world-model baseline on Kaggle.
5. Evaluate chronological holdout + q95/q99 temperature tails.
6. Calibrate uncertainty/support on held-out real data.
7. Add a real intervention study (initially cool pavement is the most defensible candidate).
8. Only then enable counterfactual results in the application.
9. After the baseline exists, compare a sparse graph transition model against the Transformer baseline for quality/latency/compute.

The project should remain fail-closed at every stage.
