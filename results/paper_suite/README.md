# Final matched paper-suite evidence

This directory is the lightweight, judge-readable export of the **completed Kaggle paper suite**. The authoritative result object and the six SVG figures here are copied directly from `SAM-WM-PAPER-SUITE-FINAL-FREIBURG-NOVISAD.zip`; they are not re-drawn or manually transcribed.

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
- `figures/main_horizon_results.svg` — horizon-wise Freiburg/Novi Sad MAE, five-seed mean ± SD.
- `figures/forecast_and_calibration.svg` — matched cross-city accuracy and frozen source calibration.
- `figures/samwm_ablations.svg` — full SAM-WM vs five controlled ablations.
- `figures/efficiency.svg` — parameter/latency efficiency evidence.
- `figures/learning_curves.svg` — actual validation-learning histories.
- `figures/forecast_trace.svg` — representative zero-shot Novi Sad forecast trace.
- `checkpoints/samwm_seed42_best.pt` — selected full SAM-WM checkpoint from the final suite.
- `SELECTED_CHECKPOINT.json` — selection criterion and checksum for that checkpoint.

The original Kaggle archive also contains **all 40 `best.pt` checkpoints, all 40 `history.json` files, and PDF/SVG/600-dpi PNG versions of all six figure families**. Those redundant per-run binaries are intentionally not copied into Git.

## Important comparison boundary

`iTransformer-adapted` and `TimeMixer-adapted` are matched independent task adapters inspired by the peer-reviewed architectures. They are **not** the authors' official repositories, so this suite supports a matched internal comparison, not a universal or official-SOTA claim.

## Deployment checkpoint is separate

The live CoolWorld application uses the already promoted, hash-locked bundle under `artifacts/deployment/`. Do **not** replace `artifacts/deployment/best.pt` with the paper-suite checkpoint without re-generating and re-validating its calibration, evaluation, promotion and provider-replay artifacts.
