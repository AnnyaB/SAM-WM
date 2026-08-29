# SAM-WM matched paper-suite artifact

This directory is the tracked, reviewable subset of the final five-seed Kaggle experiment. The source archive was validated as 40/40 completed training runs.

## Contents

- `paper_suite_results.json` — protocol/index for the tracked result package.
- `summary/*.json` — complete aggregate metrics split by model for all 8 variants × 2 evaluated domains.
- `raw/*.json` — every per-seed Freiburg/Novi Sad evaluation record, including horizon metrics, conformal coverage, traces, parameter counts, latency, and validation-selected checkpoint score.
- `FINAL_MANIFEST.json` — completion manifest from the Kaggle assembly.
- `figures/*.svg` — all six publication-figure families generated from the final experiment artifacts.

The original Kaggle archive additionally contains all 40 `best.pt` checkpoints, all 40 `history.json` files, vector PDF exports, and 600-dpi PNG exports. Those binary files are intentionally not committed to Git; this repository keeps the reviewable machine-readable evidence and editable vector figures lightweight.

## Protocol

- train/select/calibrate: Freiburg only;
- ID: Freiburg held-out;
- OOD: Novi Sad zero-shot;
- no target fine-tuning or target recalibration;
- five seeds: `17, 29, 42, 73, 101`;
- full SAM-WM + iTransformer-adapted + TimeMixer-adapted + five ablations.

FAIRUrbTemp/Turku was deferred in this matched run because the public dataset host was unavailable during the deadline window. The earlier frozen SAM-WM-only Turku evidence remains separate.
