# Kaggle training / evaluation protocol — SAM-WM v0.6.1

Kaggle is used for **training, calibration, evaluation, and benchmarking only**. FortyGuard API calls remain local/server-side so the API key never enters a notebook output or public dataset.

## Scientific split policy

Never train and report metrics on the same overlapping sequence bundle. Build the full real bundle locally, then create **purged chronological** development, calibration, and final-test partitions:

```bash
python scripts/split_sequence_dataset.py \
  --dataset data/processed/urban_thermal_sequences.npz \
  --manifest data/processed/urban_thermal_sequences.manifest.json \
  --output-dir data/processed/splits
```

The purge is `sequence_len - 1` samples between partitions so overlapping windows do not share source timestamps across boundaries.

Upload these files as a private Kaggle Dataset only if the applicable data terms permit storage:

- `development.npz` + `development.manifest.json`
- `calibration.npz` + `calibration.manifest.json`
- `test.npz` + `test.manifest.json`
- `split_manifest.json`

## 1. Environment

```bash
git clone https://$GITHUB_TOKEN@github.com/AnnyaB/SAM-WM.git
cd SAM-WM
pip install -e '.[ml,research,dev]'
python -m pytest -m 'not live' -q
```

## 2. Verify hashes before any training

```bash
python - <<'PY'
from pathlib import Path
from coolworld.ml.data import DatasetManifest, sha256_file
root=Path('/kaggle/input/<PRIVATE-DATASET>')
for name in ('development','calibration','test'):
    p=root/f'{name}.npz'
    m=DatasetManifest.load(root/f'{name}.manifest.json')
    print(name, m.dataset_id, sha256_file(p)==m.file_sha256, len(m.source_records))
PY
```

Every line must print `True` for the hash check.

## 3. Cheap baselines first

Evaluate persistence and linear-trend baselines on the **final test bundle** before training a neural model:

```bash
python scripts/evaluate_baselines.py \
  --dataset /kaggle/input/<PRIVATE-DATASET>/test.npz \
  --manifest /kaggle/input/<PRIVATE-DATASET>/test.manifest.json \
  --output /kaggle/working/baselines_test.json
```

Do not tune architecture decisions on final-test results. Use development validation for model selection.

## 4. Train on development only

Run at least three deterministic seeds for the paper. Start with one seed as a smoke test.

```bash
python scripts/train_world_model.py \
  dataset_npz=/kaggle/input/<PRIVATE-DATASET>/development.npz \
  dataset_manifest=/kaggle/input/<PRIVATE-DATASET>/development.manifest.json \
  output_dir=/kaggle/working/counterfactual_model_seed42 \
  batch_size=16 epochs=100 seed=42
```

The trainer itself performs a further **purged chronological train/validation split inside development** for early stopping. The final test partition remains untouched.

## 5. Calibrate on calibration only

```bash
python scripts/calibrate_support.py \
  --checkpoint /kaggle/working/counterfactual_model_seed42/model.pt \
  --dataset /kaggle/input/<PRIVATE-DATASET>/calibration.npz \
  --manifest /kaggle/input/<PRIVATE-DATASET>/calibration.manifest.json \
  --output /kaggle/working/counterfactual_model_seed42/support_calibration.json
```

Never fit conformal/support calibration on the final test partition.

## 6. Final held-out evaluation

```bash
python scripts/evaluate_world_model.py \
  --checkpoint /kaggle/working/counterfactual_model_seed42/model.pt \
  --dataset /kaggle/input/<PRIVATE-DATASET>/test.npz \
  --manifest /kaggle/input/<PRIVATE-DATASET>/test.manifest.json \
  --output /kaggle/working/eval_test_seed42.json
```

## 7. Extreme-heat tail robustness

```bash
python scripts/evaluate_ood.py \
  --checkpoint /kaggle/working/counterfactual_model_seed42/model.pt \
  --dataset /kaggle/input/<PRIVATE-DATASET>/test.npz \
  --manifest /kaggle/input/<PRIVATE-DATASET>/test.manifest.json \
  --output /kaggle/working/ood_tail_test_seed42.json
```

This evaluates q95/q99 **temperature-tail robustness on the held-out real test dataset only**. It is not evidence of unseen-city, unseen-climate, or causal intervention generalization.

## 8. Compute/latency benchmark

```bash
python scripts/benchmark_runtime.py \
  --checkpoint /kaggle/working/counterfactual_model_seed42/model.pt \
  --dataset /kaggle/input/<PRIVATE-DATASET>/test.npz \
  --manifest /kaggle/input/<PRIVATE-DATASET>/test.manifest.json \
  --output /kaggle/working/runtime_seed42.json
```

Record parameter count, latency, device, and memory where available. Efficiency claims must be based on measured numbers, not architecture size alone.

## 9. Repetition and reporting

Repeat training/evaluation for the declared seeds. Report mean and uncertainty across seeds. Keep every config, checkpoint hash, dataset hash, output JSON, and seed in the evidence bundle.

## 10. Causal action claims are separate

Passive temperature forecasting does **not** identify the effect of shade, trees, or reflective pavement. Action-effect claims require real non-zero intervention evidence with defensible treated/control construction, pre-trend checks, and sensitivity analysis. Until then the runtime must return `INSUFFICIENT_ACTION_SUPPORT` rather than a cooling recommendation.

## 11. What still requires separate datasets

The following cannot be established from one-city/one-period test data:

- unseen-city OOD,
- unseen-climate/weather-regime OOD,
- cross-season transfer,
- intervention causal transport,
- global deployment validity.

Collect dedicated real holdout datasets for each claim.
