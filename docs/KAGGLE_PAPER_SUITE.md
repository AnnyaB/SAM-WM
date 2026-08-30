# Final Kaggle paper suite

The final deadline experiment is frozen in `SAM-WM-PAPER-SUITE-FINAL-FREIBURG-NOVISAD.zip`. It is a **separate research benchmark namespace** and does not overwrite the hash-locked CoolWorld deployment bundle, provider replay, calibration gate or causal-evidence gate.

## Completed protocol

The final suite completed **40/40 fits**: five seeds (`17, 29, 42, 73, 101`) for:

- full SAM-WM;
- iTransformer-adapted baseline;
- TimeMixer-adapted baseline;
- SAM-WM − SIGReg;
- SAM-WM − conservative exchange;
- SAM-WM − sparse mental map;
- SAM-WM − bounded residual;
- SAM-WM − RH.

Every learned model uses the same Freiburg train/validation/test protocol, 48-hour context, six-hour horizon, source-only normalization, validation checkpoint selection and Freiburg-validation conformal calibration. **Novi Sad is evaluated zero-shot with no target fine-tuning and no target recalibration.**

The final deadline run did **not** include Turku. `FINAL_MANIFEST.json` records: `Deferred because FAIRUrbTemp host was unavailable during the deadline run.` Do not mix an earlier frozen Turku result into this matched 40-run comparison.

The two external baselines are independent matched adapters inspired by the published iTransformer and TimeMixer architectures. They are not the authors' official implementations, so the result is an internal matched comparison rather than an official-SOTA claim.

## Exact final archive

The final archive contains:

```text
paper_suite/
├── FINAL_MANIFEST.json
├── paper_suite_results.json
├── figures/
│   ├── efficiency.{pdf,png,svg}
│   ├── forecast_and_calibration.{pdf,png,svg}
│   ├── forecast_trace.{pdf,png,svg}
│   ├── learning_curves.{pdf,png,svg}
│   ├── main_horizon_results.{pdf,png,svg}
│   └── samwm_ablations.{pdf,png,svg}
└── runs/
    └── <8 model families>/seed_<17|29|42|73|101>/
        ├── best.pt
        └── history.json
```

`paper_suite_results.json` SHA-256:

```text
434e2d1846c9652e07c6aef055812e4333fe99d030e384983bcb40dcae06a0f6
```

## Regenerate the publication figures

After extracting the final archive so that `artifacts/paper_suite/` contains the result JSON and `runs/` histories:

```bash
python scripts/plot_results.py \
  --results artifacts/paper_suite/paper_suite_results.json \
  --root artifacts/paper_suite \
  --out artifacts/paper_suite/figures
```

This is the same plotting entry point used by the final Kaggle assembly cell. It produces **editable SVG, vector PDF and 600-dpi PNG** outputs. Experimental values come from `paper_suite_results.json` and the saved per-seed histories; the plotting layer does not hand-enter result values.

## Audit before citing a number

1. Use `results/paper_suite/paper_suite_results.json` as the tracked machine-readable source of truth.
2. Treat all displayed uncertainty as five-seed mean ± SD from that object.
3. Do not claim that the adapted baselines are official-code reproductions.
4. Do not claim universal SOTA: TimeMixer-adapted is marginally better on Freiburg MAE, while SAM-WM's strongest matched result is zero-shot cross-city preservation on Novi Sad.
5. Keep the final matched Freiburg/Novi Sad suite separate from older frozen deployment/Turku evidence.
