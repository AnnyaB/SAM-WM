# Final matched paper-suite evidence

This directory is the lightweight, judge-readable export of the **completed Kaggle paper suite**. The authoritative result object, selected checkpoint, and five publication SVGs are imported byte-for-byte from `SAM-WM-PAPER-SUITE-FINAL-FREIBURG-NOVISAD.zip`. `figures/forecast_trace.svg` is the one intentional layout-only re-render: it reads the **same saved final trace values** from `paper_suite_results.json` and moves the legend outside the plotting rectangle so no label obscures a trajectory. No metric, prediction, target, seed, split, model output, or benchmark value is altered.

## Final protocol

- **40/40 fits:** 8 model/ablation families × 5 seeds (`17, 29, 42, 73, 101`).
- **Train:** Freiburg train split only.
- **Checkpoint selection:** Freiburg validation MAE only.
- **Uncertainty calibration:** Freiburg validation residuals only.
- **ID test:** Freiburg held-out.
- **OOD test:** Novi Sad, zero-shot.
- **Target fine-tuning/recalibration:** none.
- **Context / horizon:** 48 h → +1…+6 h.
- **Turku:** deferred in the final deadline run because the FAIRUrbTemp host was unavailable; see `FINAL_MANIFEST.json`.

## Files

- `paper_suite_results.json` — exact final machine-readable result object; SHA-256 `434e2d1846c9652e07c6aef055812e4333fe99d030e384983bcb40dcae06a0f6`.
- `FINAL_MANIFEST.json` — exact final completion manifest.
- `figures/main_horizon_results.svg` — horizon-wise Freiburg/Novi Sad MAE, five-seed mean ± SD; archive-original SVG.
- `figures/forecast_and_calibration.svg` — matched cross-city accuracy and frozen source calibration; archive-original SVG.
- `figures/samwm_ablations.svg` — full SAM-WM vs five controlled ablations; archive-original SVG.
- `figures/efficiency.svg` — parameter/latency efficiency evidence; archive-original SVG.
- `figures/learning_curves.svg` — actual validation-learning histories; archive-original SVG.
- `figures/forecast_trace.svg` — representative zero-shot Novi Sad forecast trace, layout-only re-rendered from the exact final saved trace values with an external legend.
- `checkpoints/samwm_seed42_best.pt` — selected full SAM-WM checkpoint from the final suite.
- `SELECTED_CHECKPOINT.json` — selection criterion and checksum for that checkpoint.

The original Kaggle archive also contains **all 40 `best.pt` checkpoints, all 40 `history.json` files, and PDF/SVG/600-dpi PNG versions of all six figure families**. Those redundant per-run binaries are intentionally not copied into Git.

To reproduce only the layout-corrected trace after importing the archive artifacts:

```bash
python scripts/plot_forecast_trace.py \
  --results results/paper_suite/paper_suite_results.json \
  --out results/paper_suite/figures
```

This writes SVG, vector PDF and 600-dpi PNG from the stored final trace arrays. It does not retrain or reevaluate any model.

## Important comparison boundary

`iTransformer-adapted` and `TimeMixer-adapted` are matched independent task adapters inspired by the peer-reviewed architectures. They are **not** the authors' official repositories, so this suite supports a matched internal comparison, not a universal or official-SOTA claim.

## Deployment checkpoint is separate

The live CoolWorld application uses the already promoted, hash-locked bundle under `artifacts/deployment/`. Do **not** replace `artifacts/deployment/best.pt` with the paper-suite checkpoint without re-generating and re-validating its calibration, evaluation, promotion and provider-replay artifacts.