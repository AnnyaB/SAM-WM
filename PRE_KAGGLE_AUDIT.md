# SAM-WM v0.6.1 — Pre-Kaggle audit

## Ready in code

- real-only evidence store and SHA-256 provenance
- FortyGuard asynchronous submit -> activity_id -> status polling
- heatmap, environmental parameters, satellite, street-view client paths
- Heat Intelligence streamed-PDF retrieval + evidence persistence
- fixed-grid sequence ETL
- optional real intervention action log
- purged chronological development/calibration/test splitting
- action-conditioned JEPA/GRU/Transformer world-model baseline
- held-out evaluation entry point
- persistence + linear-trend baselines
- q95/q99 extreme-temperature-tail evaluation
- support/conformal calibration entry point
- runtime/parameter benchmark
- model/evidence readiness gate
- fail-closed counterfactual API

## Not evidence-complete yet

These require real data/runs and therefore cannot be marked complete in source code alone:

1. non-empty real FortyGuard time series for a fixed AOI
2. real non-zero intervention training/evaluation records
3. development/calibration/test artifacts produced from those records
4. trained multi-seed checkpoints
5. final held-out metrics and confidence intervals
6. unseen-city and weather-regime OOD datasets
7. causal replay with pre-trend/control diagnostics
8. architecture comparison proving the lowest-cost/best-performing model
9. live browser integration with validated trained artifacts
10. external SOTA comparison on a directly comparable task/dataset

## API surface note

The public documentation states that the Enterprise API has six POST analysis endpoints. The current public pages clearly expose Heatmap, Environmental Parameters, Satellite Segmentation, Street View Segmentation, and Heat Intelligence. The Temperature Property documentation page does not expose a usable request schema in the currently retrievable documentation snapshot. SAM-WM must not invent that schema. Add it only after the official developer guide/webinar provides the exact contract.

## Scientific boundary

FortyGuard LTMs provide temperature intelligence. SAM-WM's research contribution is the real-evidence, action-conditioned, support-aware counterfactual layer. No unsupported cooling delta is a result.
