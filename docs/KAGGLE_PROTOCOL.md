# Kaggle execution protocol

This is the frozen execution order for the SAM-WM benchmark. Do not use the old `SAM_WM_V41_KAGGLE_INPUT.zip` workflow.

## Kaggle settings

- Accelerator: one GPU is sufficient; T4/P100 are fine.
- Internet: ON for the Freiburg/Novi Sad Zenodo downloads and repository install.
- Persistence: save `/kaggle/working/SAM-WM/artifacts` as notebook output after every completed seed.
- FAIRUrbTemp: attach/extract the official DOI `10.48620/93247` files as a Kaggle input because the repository intentionally does not guess undocumented BORIS archive URLs.

## 1. Clone the frozen commit

After GitHub CI is green, record the exact `main` commit SHA and clone it:

```bash
cd /kaggle/working
git clone https://github.com/AnnyaB/SAM-WM.git
cd SAM-WM
git rev-parse HEAD
python -m pip install -e '.[dev]'
make verify
```

Do not continue if `make verify` fails.

## 2. Train three independent seeds

```bash
python train.py --seed 0 --out artifacts/freiburg
python train.py --seed 1 --out artifacts/freiburg
python train.py --seed 2 --out artifacts/freiburg
```

Expected checkpoint paths:

```text
artifacts/freiburg/seed_0/best.pt
artifacts/freiburg/seed_1/best.pt
artifacts/freiburg/seed_2/best.pt
```

Only Freiburg validation MAE may be used for model selection.

## 3. Inspect validation only

For each seed:

```bash
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data freiburg --split validation --out artifacts/eval/seed_0
```

Repeat for seeds 1 and 2. At this point architecture/configuration changes are still allowed if they were decided from training/validation evidence only. If any change is made, delete the experimental run artifacts and retrain all seeds from scratch before held-out access.

## 4. Freeze

Before opening any final/OOD labels, record:

```bash
git rev-parse HEAD
sha256sum config/train.yaml
sha256sum artifacts/freiburg/seed_*/best.pt
```

No architecture, hyperparameter, preprocessing, split, or QC changes are allowed after this point for the reported run.

## 5. Open Freiburg final test once

For each seed, use a separate output directory. The evaluator writes `freiburg_HELDOUT_OPEN.json` before accessing labels and refuses a second opening in that directory.

```bash
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data freiburg --split heldout --open-heldout --out artifacts/eval/seed_0
```

Repeat for seeds 1 and 2.

## 6. OOD-1: Novi Sad zero-shot

No fine-tuning and no recalibration on Novi Sad labels:

```bash
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data novisad --split heldout --open-heldout --out artifacts/eval/seed_0
```

Repeat for seeds 1 and 2.

## 7. OOD-2: FAIRUrbTemp zero-shot

Attach the official FAIRUrbTemp extracted directory and choose one city before looking at model results. Use the same city for all seeds.

```bash
FAIR=/kaggle/input/<official-fairurbtemp-dataset>/<extracted-root>
CITY='<preregistered-city>'
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data fairurbtemp --root "$FAIR" --city "$CITY" \
  --split heldout --open-heldout --out artifacts/eval/seed_0
```

Repeat for seeds 1 and 2. Observation-level FAIRUrbTemp QC flags are excluded from scoring by the loader.

## 8. Aggregate and plot

```bash
python summarize.py --root artifacts/eval --out artifacts/summary.json
python plot.py \
  artifacts/eval/seed_0/freiburg_metrics.json \
  artifacts/eval/seed_0/novisad_metrics.json \
  artifacts/eval/seed_0/fairurbtemp_metrics.json \
  --out artifacts/figures/seed_0
```

The summary across all three seeds is the result used in the paper/hackathon technical evidence. Never manually type metrics into figures or README tables.

## 9. Save Kaggle artifacts

At minimum preserve:

```text
artifacts/freiburg/seed_*/best.pt
artifacts/freiburg/seed_*/history.json
artifacts/freiburg/seed_*/resolved_config.json
artifacts/eval/seed_*/*_HELDOUT_OPEN.json
artifacts/eval/seed_*/*_manifest.json
artifacts/eval/seed_*/*_metrics.json
artifacts/summary.json
artifacts/figures/
```

These are the evidence chain. The trained checkpoint is promoted to the deployment/Hugging Face package only after this evaluation finishes.
