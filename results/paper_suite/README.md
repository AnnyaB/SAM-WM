# SAM-WM matched paper-suite evidence

This directory is the reviewable result package from the final five-seed Kaggle experiment: **40/40 completed fits** (8 model/ablation families × 5 seeds).

## Files

- `paper_suite_results.json` — protocol/index linking the complete split result files.
- `training_histories.json` — all 40 Freiburg training/validation histories used by the learning-dynamics figure.
- `raw/*.json` — every per-seed Freiburg/Novi Sad evaluation record split by model/variant, including horizon metrics, coverage, latency, parameter counts and representative traces.
- `summary/*.json` — aggregate five-seed records split by model/variant.
- `FINAL_MANIFEST.json` — completion/provenance manifest.
- `figures/*.svg` — tracked editable vector figures used by the README. `scripts/plot_paper_elite.py` regenerates SVG, vector PDF and 600-dpi PNG exports from the machine-readable evidence.

The original Kaggle archive also retains all 40 `best.pt` research checkpoints. Those per-run binaries are intentionally not duplicated in Git. The promoted model actually used by CoolWorld is committed separately at `artifacts/deployment/best.pt`.

## Protocol

- Freiburg train only;
- Freiburg validation only for checkpoint selection;
- Freiburg validation residuals only for conformal calibration;
- Freiburg held-out ID test;
- Novi Sad zero-shot OOD test;
- no target fine-tuning;
- no target recalibration;
- 48 h context → +1…+6 h;
- seeds `17, 29, 42, 73, 101`.

The external baselines are independent matched adapters inspired by the peer-reviewed iTransformer and TimeMixer architectures. They are not the authors' official implementations.

FAIRUrbTemp/Turku was not rerun for every matched model during the deadline window because the public dataset host became unavailable. The earlier frozen SAM-WM-only Turku result remains in `artifacts/summary.json` and is deliberately kept separate from the matched baseline claims.

## Figures

- `benchmark_overview.svg` — cross-city MAE and frozen-source empirical coverage.
- `horizon_transfer.svg` — actual +1…+6 h MAE curves for the three full models.
- `ablation_study.svg` — full SAM-WM versus five controlled ablations.
- `learning_dynamics.svg` — five-seed Freiburg validation learning dynamics.
- `calibration_efficiency.svg` — OOD coverage and compactness versus OOD error.
- `forecast_trace.svg` — representative six-hour observed/predicted traces.
- `frozen_three_domain.svg` — separate frozen SAM-WM-only Freiburg/Novi Sad/Turku benchmark.

Regenerate SVG/PDF/600-dpi PNG exports with:

```bash
python scripts/plot_paper_elite.py \
  --results results/paper_suite/paper_suite_results.json \
  --histories results/paper_suite/training_histories.json \
  --frozen-summary artifacts/summary.json \
  --out results/paper_suite/figures
```
