# One-cell Kaggle paper suite

This is the **new research-v2 benchmark namespace**. It does not overwrite the frozen SAM-WM v1 checkpoint, summary, FortyGuard replay, calibration gate, or deployment bundle.

## What the cell runs

`paper` mode runs five seeds (`17, 29, 42, 73, 101`) for:

- full SAM-WM;
- iTransformer-adapted baseline;
- TimeMixer-adapted baseline;
- SAM-WM − SIGReg;
- SAM-WM − conservative exchange;
- SAM-WM − sparse mental map;
- SAM-WM − bounded residual;
- SAM-WM − RH.

Every learned model uses the same Freiburg train/validation/test dates, 48-hour context, six-hour horizon, source-only normalization, validation checkpoint selection, and source-validation conformal calibration. Novi Sad and Turku are evaluated zero-shot without target fine-tuning or target recalibration.

The baseline implementations in this repository are independent task adapters inspired by the published architectures. They are **not** presented as the authors' official source code. References:

- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024 Spotlight. Official repository: `thuml/iTransformer` (MIT).
- Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024. Official repository: `kwuking/TimeMixer` (Apache-2.0).

## Kaggle setup

Use a GPU notebook (T4 is sufficient for these compact adapters) and turn Internet on so the checksum-pinned Freiburg and Novi Sad releases can be downloaded. Add the extracted FAIRUrbTemp DOI dataset as Kaggle input. The loader recursively finds its hourly `.tsv` files and filters the preregistered city `Turku`.

The repository must already be available in the notebook filesystem, for example at `/kaggle/working/SAM-WM`. Do not put a GitHub token directly in notebook code or output.

## Single experiment cell

```python
from pathlib import Path
import os
import subprocess
import sys

REPO = Path("/kaggle/working/SAM-WM")
assert (REPO / "paper_suite.py").is_file(), f"SAM-WM repo not found at {REPO}"

# Prefer a FAIRUrbTemp input whose path name identifies the dataset.
candidates = [
    path
    for path in Path("/kaggle/input").iterdir()
    if "fairurb" in path.name.lower()
]
if not candidates:
    # Conservative fallback: find an input containing SEF/TSV files.
    candidates = [
        path
        for path in Path("/kaggle/input").iterdir()
        if next(path.rglob("*.tsv"), None) is not None
    ]
assert candidates, "Add the extracted FAIRUrbTemp dataset to Kaggle inputs first."
FAIRURB_ROOT = candidates[0]
print("FAIRUrbTemp root:", FAIRURB_ROOT)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)],
    check=True,
)

# Full five-seed paper protocol. Completed per-seed checkpoints are resumed automatically.
subprocess.run(
    [
        sys.executable,
        "paper_suite.py",
        "--config", "config/paper.yaml",
        "--mode", "paper",
        "--fairurb-root", str(FAIRURB_ROOT),
        "--fairurb-city", "Turku",
        "--out", "artifacts/paper_suite",
    ],
    cwd=REPO,
    check=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

print("\nRESULTS:", REPO / "artifacts/paper_suite/paper_suite_results.json")
print("FIGURES:", REPO / "artifacts/paper_suite/figures")
```

For a time-constrained dry run only, change `--mode paper` to `--mode deadline`. That uses three seeds and a shorter early-stopping ceiling. **Deadline-mode numbers must be labelled as preliminary and must not silently replace five-seed paper results.**

## Figure outputs

The command automatically creates each figure as editable SVG, vector PDF, and 600-dpi PNG:

- `main_horizon_results.*` — +1…+6 h curves with mean ± SD;
- `forecast_and_calibration.*` — cross-city MAE and frozen-calibration coverage;
- `samwm_ablations.*` — full SAM-WM against mechanism/data ablations;
- `efficiency.*` — parameter and latency trade-offs;
- `learning_curves.*` — validation MAE over training with multi-seed bands;
- `forecast_trace.*` — representative zero-shot six-hour rollout.

The plotting layer uses a restrained research palette, thin axes, no decorative dashboard styling, embedded vector text, and a consistent SAM-WM salmon/red emphasis. All values come from `paper_suite_results.json` or per-seed history JSON; the plotting code does not hand-enter experimental results.

## After the run

Before any result is added to the public-facing README or paper, audit:

1. all requested seeds completed;
2. no baseline/ablation was tuned on Freiburg test, Novi Sad, or Turku;
3. every domain has the expected observation count;
4. plots match the machine-readable JSON;
5. any claim of improvement is supported by the actual matched comparison.
