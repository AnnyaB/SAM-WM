# Kaggle execution protocol

This is the frozen execution order for the single SAM-WM benchmark.

Kaggle must execute source that is already complete. It is used for real dataset acquisition, GPU
training, measured validation/final/OOD results, and checkpoint production — not for inventing a
second model family or changing SAM-WM after held-out access.

## Runtime

- Accelerator: one Kaggle GPU.
- Internet: ON.
- Repository source: resolve `main` once and record the exact 40-character SHA.
- Private repository: use Kaggle Secret `GITHUB_TOKEN` with read access.
- Freiburg/Novi Sad: fetched through the registered Zenodo loaders with checksums.
- FAIRUrbTemp: attach the extracted official DOI `10.48620/93247` files as a Kaggle Input.
- Persist `/kaggle/working/SAM-WM/artifacts` as notebook output.

The executable reference is `notebooks/SAM_WM_KAGGLE.ipynb`. CI syntax-checks every code cell.

## 1. Bootstrap and verify

The notebook resolves one exact GitHub commit SHA, safely extracts that source without printing the
GitHub token, installs the package, runs `make verify`, and refuses to continue without CUDA.

Stop immediately if verification fails.

## 2. Train only full SAM-WM on Freiburg development data

Run:

```bash
python research.py --out artifacts/research
```

Exactly one model is trained: **SAM-WM**.

Frozen seeds:

```text
17, 29, 42, 73, 101
```

Each seed trains on Freiburg train data, early-stops on Freiburg validation MAE, and writes:

```text
artifacts/research/seed_*/best.pt
artifacts/research/seed_*/history.json
artifacts/research/seed_*/resolved_config.json
artifacts/research/seed_*/dataset_manifest.json
artifacts/research/seed_*/validation_metrics.json
```

The stage finishes by writing:

```text
artifacts/research/PRE_FREEZE_MANIFEST.json
```

No Freiburg held-out, Novi Sad target, or FAIRUrbTemp target may be read before this point.

## 3. Inspect Freiburg validation only and preselect deployment seed

You may inspect only:

```text
artifacts/research/seed_17/validation_metrics.json
artifacts/research/seed_29/validation_metrics.json
artifacts/research/seed_42/validation_metrics.json
artifacts/research/seed_73/validation_metrics.json
artifacts/research/seed_101/validation_metrics.json
```

Then run:

```bash
python promote.py preselect
```

The deployment seed is selected by minimum Freiburg validation MAE, with ascending seed as the
fixed tie-break. Final/OOD evidence is forbidden from influencing this choice.

If any architecture, objective, preprocessing, QC, split, graph construction, or hyperparameter is
changed now, discard this run and restart from step 2 before opening held-out/OOD data.

## 4. Freeze the reported SAM-WM run

Write `artifacts/FREEZE_MANIFEST.json` containing hashes of:

- exact GitHub source SHA;
- `config/train.yaml`;
- `artifacts/research/PRE_FREEZE_MANIFEST.json`;
- all five full SAM-WM checkpoints;
- the validation-only deployment-selection artifact.

After this manifest is written, no model/protocol change is allowed for the reported benchmark run.

## 5. Freiburg final test — once per seed

For every frozen seed:

```bash
python eval.py \
  --checkpoint artifacts/research/seed_<SEED>/best.pt \
  --data freiburg --split heldout --open-heldout \
  --out artifacts/eval/seed_<SEED>
```

The evaluator writes its held-out receipt before computing metrics and refuses to reopen the same
held-out evaluation in the same output directory.

## 6. OOD-1 — Novi Sad zero-shot

For every frozen seed:

```bash
python eval.py \
  --checkpoint artifacts/research/seed_<SEED>/best.pt \
  --data novisad --split heldout --open-heldout \
  --out artifacts/eval/seed_<SEED>
```

Rules: no fine-tuning, no target-driven hyperparameter choice, and no OOD-label recalibration.

## 7. OOD-2 — FAIRUrbTemp unseen city zero-shot

Before viewing any SAM-WM FAIRUrbTemp metric, attach the official DOI files, set `FAIR_ROOT`, and
choose one `FAIR_CITY` using metadata/coverage criteria only. Keep that city identical for all seeds.

For every frozen seed:

```bash
python eval.py \
  --checkpoint artifacts/research/seed_<SEED>/best.pt \
  --data fairurbtemp --root "$FAIR_ROOT" --city "$FAIR_CITY" \
  --split heldout --open-heldout \
  --out artifacts/eval/seed_<SEED>
```

Observation-level `qc=` flags are excluded by the loader. There is no target fine-tuning or
recalibration.

## 8. Aggregate measured evidence

Run `summarize.py` on `artifacts/eval` and generate plots only from saved machine-readable final/OOD
artifacts. Never manually type benchmark values into figures, README tables, the UI, or the video.

## 9. Finalize the already-preselected SAM-WM deployment bundle

After all three frozen evaluations exist:

```bash
python promote.py finalize
```

This copies only the seed that was preselected from Freiburg validation before final/OOD access and
verifies checkpoint/evaluation hashes before producing deployment calibration/evidence manifests.

## 10. Provider/UI work happens after Kaggle

The learned deployment bundle is then tested against recorded real FortyGuard TCM evidence through
`provider_replay.py`, integrated with the existing CoolWorld 3D UI, deployed publicly, visually
verified end-to-end, and recorded in the final ≤3-minute working demo video.

Provider replay is a deployment-domain gate; it is not relabelled as a third research benchmark.

## Evidence chain

Preserve at minimum:

```text
artifacts/FROZEN_SOURCE_SHA.txt
artifacts/research/PRE_FREEZE_MANIFEST.json
artifacts/research/seed_*/best.pt
artifacts/research/seed_*/history.json
artifacts/research/seed_*/resolved_config.json
artifacts/research/seed_*/validation_metrics.json
artifacts/DEPLOYMENT_SELECTION.json
artifacts/FREEZE_MANIFEST.json
artifacts/eval/seed_*/*_HELDOUT_OPEN.json
artifacts/eval/seed_*/*_heldout_manifest.json
artifacts/eval/seed_*/*_heldout_metrics.json
artifacts/summary.json
artifacts/figures/
artifacts/deployment/PROMOTION_MANIFEST.json
```
