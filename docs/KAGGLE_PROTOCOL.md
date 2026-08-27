# Kaggle execution protocol

This is the frozen execution order for the SAM-WM benchmark.

The repository must already contain the full source model, real benchmark loaders, research
ablation/control harness, evaluation gates, provider-evidence path, and UI code. Kaggle is used for
real dataset acquisition, GPU training, measured validation/final/OOD results, and checkpoint
production.

## Runtime

- Accelerator: one Kaggle GPU.
- Internet: ON.
- Repository source: resolve `main` once and record the exact 40-character SHA.
- Private repository: use Kaggle Secret `GITHUB_TOKEN` with read access.
- Freiburg/Novi Sad: fetched through the registered Zenodo loaders with checksums.
- FAIRUrbTemp: attach the official DOI `10.48620/93247` extracted files as a Kaggle Input.
- Persist `/kaggle/working/SAM-WM/artifacts` as notebook output.

The executable reference is `notebooks/SAM_WM_KAGGLE.ipynb`. Every code cell is plain Python and
CI syntax-checks it.

## 1. Bootstrap and verify

The notebook:

1. resolves one exact GitHub commit SHA;
2. safely downloads that archive without printing credentials;
3. rejects archive traversal and link entries;
4. installs the package;
5. runs `make verify`;
6. requires a visible CUDA GPU.

Stop if verification fails.

## 2. Pre-freeze research — validation only

Run:

```bash
python research.py --stage all-pre-freeze --out artifacts/research
```

This stage is forbidden from reading Freiburg held-out or either OOD target.

Frozen seeds:

```text
17, 29, 42, 73, 101
```

The suite executes:

- full SAM-WM;
- three non-trainable validation sanity baselines;
- seven retrained structural ablations;
- `no_sigreg`;
- `temperature_only`.

Artifacts are written below `artifacts/research/` and summarized by:

```text
artifacts/research/PRE_FREEZE_MANIFEST.json
```

That manifest explicitly records `heldout_or_ood_accessed: false`.

## 3. Inspect Freiburg validation only

You may inspect only the full-branch validation artifacts:

```text
artifacts/research/full/seed_17/validation_metrics.json
artifacts/research/full/seed_29/validation_metrics.json
artifacts/research/full/seed_42/validation_metrics.json
artifacts/research/full/seed_73/validation_metrics.json
artifacts/research/full/seed_101/validation_metrics.json
```

A change to architecture, objective, preprocessing, QC, split, graph construction, or
hyperparameters defines a new run. If changed, rerun the entire pre-freeze suite before any
held-out/OOD access.

## 4. Freeze the reported branch

Only after deciding that no more development changes will be made, write:

```text
artifacts/FREEZE_MANIFEST.json
```

It hashes:

- exact GitHub source SHA;
- `config/train.yaml`;
- the complete pre-freeze research manifest;
- all five full SAM-WM checkpoints.

After this point no model/protocol change is allowed for the reported run.

## 5. Freiburg final test

Open once per seed with `--open-heldout`.

The evaluator writes a receipt before scoring and refuses to reopen the same held-out evaluation in
the same output directory.

Expected artifacts:

```text
artifacts/eval/seed_*/freiburg_HELDOUT_OPEN.json
artifacts/eval/seed_*/freiburg_heldout_manifest.json
artifacts/eval/seed_*/freiburg_heldout_metrics.json
```

## 6. OOD-1 — Novi Sad

Run zero-shot after the freeze.

Rules:

- no fine-tuning;
- no hyperparameter selection;
- no OOD-label recalibration;
- no city-specific learned graph parameters beyond deterministic geometry construction.

## 7. OOD-2 — FAIRUrbTemp unseen city

Before viewing any SAM-WM FAIRUrbTemp metric:

1. attach the official DOI dataset;
2. set `FAIR_ROOT`;
3. choose `FAIR_CITY` from metadata/coverage criteria only;
4. keep that city fixed for all five seeds.

Observation-level `qc=` flags are excluded from scoring. There is no OOD fine-tuning or
recalibration.

## 8. Aggregate and plot

Run `summarize.py` on `artifacts/eval` and generate plots only from machine-readable final/OOD
artifacts.

Never manually type benchmark numbers into a figure, README table, or demo.

## Evidence chain

Preserve at minimum:

```text
artifacts/FROZEN_SOURCE_SHA.txt
artifacts/research/PRE_FREEZE_MANIFEST.json
artifacts/research/full/seed_*/best.pt
artifacts/research/full/seed_*/history.json
artifacts/research/full/seed_*/resolved_config.json
artifacts/research/*/seed_*/validation_metrics.json
artifacts/FREEZE_MANIFEST.json
artifacts/eval/seed_*/*_HELDOUT_OPEN.json
artifacts/eval/seed_*/*_heldout_manifest.json
artifacts/eval/seed_*/*_heldout_metrics.json
artifacts/summary.json
artifacts/figures/
```

The final learned checkpoint is promoted to CoolWorld/Hugging Face only after the frozen evaluation
finishes.
