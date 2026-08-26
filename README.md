# SAM-WM

**Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence**

SAM-WM is a compact research system for learning reusable urban thermal mechanisms from real sensor networks, forecasting short-horizon temperature fields, testing zero-shot cross-city transfer, and handing physical intervention decisions to a separate evidence layer (CANDRA). It is designed for the FortyGuard Hackathon'26 and as a reproducible foundation for later peer-reviewed research.

> **Truth boundary.** SAM-WM predicts thermal futures. It does not claim a tree, shade structure, or reflective pavement causes a particular temperature reduction unless independent treated/control evidence supports that effect. CANDRA abstains when such evidence is missing or weak.

## Why this is different

SAM-WM encodes four ideas directly in the model:

1. **Sparse mental map.** A k-nearest-neighbour city graph is computed once from station geometry, so message passing is O(E), not dense O(N²).
2. **Reusable mechanism library.** Future temperature evolves through conservative exchange, optional observed-wind transport, bounded source/sink forcing, and a deliberately small residual mechanism.
3. **Dream + surprise.** The latent state is rolled forward autoregressively; probabilistic forecast error becomes a violation-of-expectation signal rather than an unqualified claim of "understanding".
4. **Evidence-bounded action.** Intervention effects are handled separately by CANDRA. Forecast accuracy alone never licenses a causal cooling claim.

The objective intentionally follows the compact design philosophy of LeWorldModel: a predictive objective plus one representation regularizer,

`L = L_pred + lambda_SIG * L_SIGReg`,

with physical structure expressed in the operators rather than a large hand-weighted penalty collection.

## Repository layout

```text
SAM-WM/
├── train.py                 # primary Freiburg training
├── eval.py                  # ID final test + zero-shot OOD
├── fortyguard_check.py      # one real, auditable API request
├── plot.py                  # paper/demo figures from saved metrics
├── config/                  # frozen experiment protocol
├── src/coolworld/
│   ├── samwm.py             # core world model
│   ├── graph.py             # sparse city graph
│   ├── benchmarks.py        # real dataset loaders
│   ├── experiment.py        # training/eval/calibration
│   ├── candra.py            # causal evidence / abstention
│   ├── fortyguard.py        # crash-safe FortyGuard client
│   └── evidence.py          # content-addressed provenance helpers
├── tests/
├── docs/
└── notebooks/SAM_WM_KAGGLE.ipynb
```

The 3D UI is intentionally **not included in this branch yet**. The existing UI should be previewed and approved locally before it is wired to the trained checkpoint and pushed.

## Primary benchmark — Freiburg

Source: **Street-level weather station network in Freiburg, Germany: Curated dataset from 2022-09-01 to 2023-08-31 [L2]**, DOI `10.5281/zenodo.12732565`.

The official curated file contains hourly air temperature and relative humidity, station identity, and an `observed` / `imputed` label. SAM-WM uses the WSN stations (`FR****`) and preserves that label. Imputed context values can preserve temporal continuity, but **test metrics are computed only on observed target values**.

Frozen split:

```text
train       2022-09-03 .. 2023-04-30
validation  2023-05-01 .. 2023-06-30
final test  2023-07-01 .. 2023-08-31
context     48 h
horizon     6 h
```

This split is **our preregistered SAM-WM protocol**, not a claim that it is the split used by the dataset authors.

## Zero-shot OOD

### OOD-1 — Novi Sad NSUNET

DOI `10.5281/zenodo.7738094`. Twelve urban sites, hourly air temperature, 2016–2017. No fine-tuning and no OOD-label recalibration are permitted. The Freiburg checkpoint and Freiburg validation conformal radius are frozen before evaluation.

### OOD-2 — FAIRUrbTemp

Scientific Data 2026, DOI `10.1038/s41597-026-06804-4`; data DOI `10.48620/93247`. FAIRUrbTemp contains standardized, quality-controlled street-level urban temperature data across 12 European cities in Station Exchange Format (SEF).

The BORIS portal distributes city archives rather than one stable direct file URL. Download and extract the official archive into a Kaggle dataset/input directory, then run:

```bash
python eval.py \
  --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data fairurbtemp \
  --root /kaggle/input/fairurbtemp-extracted \
  --city <city-name> \
  --out artifacts/eval/seed_0
```

The loader fails closed if it cannot unambiguously identify at least five hourly temperature SEF stations with a common interval. It does not silently guess a schema.

## Install

```bash
# Registered local/CI/Kaggle interpreter contract: CPython 3.12.
# Use the interpreter by absolute/path-local executable, not a shell `pytest` shim.
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e '.[dev]'
./.venv/bin/python -m compileall -q src train.py eval.py fortyguard_check.py plot.py summarize.py
./.venv/bin/python -m pytest
```

The registered hackathon environment is CPython 3.12. Do not mix a Python 3.14 venv with a pyenv/conda `pytest` executable; invoke every tool as `python -m ...` through the same interpreter. Passing software tests is **not** a research result.

## Train

```bash
python train.py --config config/train.yaml --seed 0 --out artifacts/freiburg
python train.py --config config/train.yaml --seed 1 --out artifacts/freiburg
python train.py --config config/train.yaml --seed 2 --out artifacts/freiburg
```

No Freiburg final-test labels are used in training or early stopping.

## Evaluate

```bash
# Primary held-out test
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt --data freiburg --out artifacts/eval/seed_0

# OOD-1
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt --data novisad --out artifacts/eval/seed_0

# OOD-2 after adding/extracting the official FAIRUrbTemp archive
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt --data fairurbtemp --root /kaggle/input/fairurbtemp-extracted --city <city> --out artifacts/eval/seed_0
```

Repeat evaluation for every seed. **Do not choose the best seed.** Report mean ± standard deviation across all frozen seeds.

## FortyGuard real API proof

Never commit an API key. Keep it only in a local `.env` / shell environment or Kaggle Secret named `FORTYGUARD_API_KEY`.

```bash
export FORTYGUARD_API_KEY='...'
python fortyguard_check.py \
  --date 2026-08-26 \
  --time 15:00 \
  --aoi examples/sanjose_aoi.geojson \
  --granularity 100
```

The client persists the `activity_id` immediately and resumes polling the same activity after interruption, rather than blindly spending credits by reposting the same request. Completed payloads are stored under `artifacts/fortyguard/`; this directory is git-ignored.

For the final hackathon README, add **one sanitized real request and response excerpt generated from this store**. Never paste the API key. Do not fabricate provider fields.

## Results and figures

Research numbers are intentionally absent before the real runs. After all three seeds finish:

```bash
python plot.py \
  artifacts/eval/seed_0/freiburg_metrics.json \
  artifacts/eval/seed_0/novisad_metrics.json \
  --out artifacts/figures
```

The required report set is:

- train / validation learning curves per seed;
- Freiburg MAE, RMSE, bias, p95 absolute error;
- horizon-wise error (to be added from raw prediction export before paper submission);
- 90% split-conformal coverage;
- zero-shot Novi Sad and FAIRUrbTemp metrics;
- model parameter count and inference latency;
- mechanism-weight diagnostics;
- surprise distribution under ID vs OOD;
- failures and abstentions, not only wins.

Do not write "SOTA", "human-level", "AGI", "cools Earth", or "beats model X" unless the corresponding experiment directly establishes that statement.

## CANDRA

`src/coolworld/candra.py` contains a temporal block-bootstrap difference-in-differences reference estimator and conservative action gate. It is valid only when a genuine intervention/control design is available and its assumptions are justified. Ordinary observational temperature data do not satisfy that requirement by themselves.

## Reproducibility rules

- fixed seeds and chronology;
- immutable dataset checksums where the host exposes stable files;
- no target-label leakage into normalizers;
- validation-only early stopping;
- split conformal calibration on Freiburg validation only;
- no OOD fine-tuning or OOD-label recalibration;
- no API secrets in Git;
- fail-closed dataset parsers;
- no invented intervention delta;
- every final number must be produced from saved artifacts, not manually typed.

## UI and deployment

The existing CoolWorld 3D UI is intentionally held outside this redesign until its visual design is approved. After training, the next branch will connect:

`FortyGuard observed field -> SAM-WM baseline future -> CANDRA evidence gate -> 3D predicted/replay views`.

The live demo must remain usable without login during judging. A green UI state is never treated as scientific evidence by itself.

## Attribution

SAM-WM is an independent project. Its compact predictive + regularization philosophy is inspired by the public LeWorldModel implementation, but it does not copy LeWM's pixel architecture or claim authorship of SIGReg. Dataset and paper sources are listed in `docs/SOURCES.md`.
