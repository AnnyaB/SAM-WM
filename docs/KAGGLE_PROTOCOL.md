# Kaggle execution protocol

This is the frozen execution order for the SAM-WM benchmark. Do not use the retired
`SAM_WM_V41_KAGGLE_INPUT.zip` workflow.

## Runtime

- Accelerator: one Kaggle GPU.
- Internet: ON. Freiburg and Novi Sad are fetched from their registered Zenodo records by
  the benchmark loaders when absent, with the recorded MD5 checksums enforced.
- Repository source: the notebook resolves `main` once, downloads that exact GitHub tarball,
  and writes `artifacts/FROZEN_SOURCE_SHA.txt`.
- Private repository: add a Kaggle Secret named `GITHUB_TOKEN` with read access. If the
  repository is public, the same bootstrap works without the secret.
- FAIRUrbTemp: attach the official DOI `10.48620/93247` extracted files as a Kaggle Input.
- Persistence: save `/kaggle/working/SAM-WM/artifacts` as notebook output.

The executable reference is `notebooks/SAM_WM_KAGGLE.ipynb`. Its code cells are plain Python
and are syntax-checked by CI.

## 1. Bootstrap and verify

Run the first two notebook code cells. They:

1. resolve one 40-character GitHub commit SHA;
2. download only that source archive without printing credentials;
3. safely reject archive traversal or link entries;
4. install the package;
5. run `make verify`;
6. require a visible CUDA GPU before training.

Do not continue if verification fails.

## 2. Train three seeds

Run seeds `0`, `1`, and `2` on Freiburg. Checkpoints are written to:

```text
artifacts/freiburg/seed_0/best.pt
artifacts/freiburg/seed_1/best.pt
artifacts/freiburg/seed_2/best.pt
```

Checkpoint selection uses Freiburg validation MAE only.

## 3. Validation

Run the validation cell for all three seeds. Validation evidence is preserved separately:

```text
artifacts/eval/seed_*/freiburg_validation_metrics.json
artifacts/eval/seed_*/freiburg_validation_manifest.json
```

If a model/data/config change is made from train/validation evidence, discard that experimental
run and retrain all three seeds before opening any held-out metric.

## 4. Freeze

The freeze cell refuses to continue unless all three validation artifacts exist. It writes:

```text
artifacts/FREEZE_MANIFEST.json
```

containing the resolved source SHA, the frozen config SHA-256, all checkpoint SHA-256 values,
and validation-artifact SHA-256 values.

After this cell, no architecture, hyperparameter, preprocessing, split, or QC change is allowed
for the reported run.

## 5. Freiburg final test

Run once per seed with `--open-heldout`. The evaluator atomically writes
`freiburg_HELDOUT_OPEN.json` before held-out metric computation and refuses a second opening in
that seed output directory.

Final evidence is separate from validation evidence:

```text
artifacts/eval/seed_*/freiburg_heldout_metrics.json
artifacts/eval/seed_*/freiburg_heldout_manifest.json
```

## 6. OOD-1: Novi Sad

Run zero-shot after freeze. There is no fine-tuning and no Novi Sad recalibration. The loader
uses the registered Zenodo archive and MD5 contract. Evidence is written as
`novisad_heldout_*`.

## 7. OOD-2: FAIRUrbTemp

Before viewing a SAM-WM FAIRUrbTemp result:

1. attach the official DOI `10.48620/93247` extracted input;
2. set `FAIR_ROOT`;
3. preregister one `FAIR_CITY`;
4. use the same city for every seed.

The loader excludes observation-level QC-flagged values from scoring. There is no OOD
fine-tuning or recalibration. Evidence is written as `fairurbtemp_heldout_*`.

## 8. Aggregate and plot

`summarize.py` groups results by evaluation, so Freiburg validation cannot be mixed with
Freiburg held-out evidence. The notebook then plots only:

- Freiburg held-out;
- Novi Sad held-out;
- FAIRUrbTemp held-out.

Never manually type benchmark metrics into figures, README tables, or the demo.

## Evidence chain

Preserve at minimum:

```text
artifacts/FROZEN_SOURCE_SHA.txt
artifacts/FREEZE_MANIFEST.json
artifacts/freiburg/seed_*/best.pt
artifacts/freiburg/seed_*/history.json
artifacts/freiburg/seed_*/resolved_config.json
artifacts/eval/seed_*/freiburg_validation_*
artifacts/eval/seed_*/*_HELDOUT_OPEN.json
artifacts/eval/seed_*/*_heldout_manifest.json
artifacts/eval/seed_*/*_heldout_metrics.json
artifacts/summary.json
artifacts/figures/
```

The trained checkpoint is promoted to the deployment/Hugging Face package only after the frozen
evaluation completes.
