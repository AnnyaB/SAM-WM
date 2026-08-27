# SAM-WM

**Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence**

SAM-WM is Krsna's research + deployment project for FortyGuard Hackathon'26. It learns a compact urban thermal world model from real trajectories, evaluates zero-shot transfer to unseen cities, and keeps physical cooling claims behind an independent causal-evidence gate (CANDRA).

> Scientific boundary: SAM-WM forecasts temperature dynamics. It does **not** turn an observational forecast into a causal intervention effect. Missing evidence produces an abstention, never a fabricated cooling number.

## Core idea

The model represents a city as a sparse physical mental map and composes four typed mechanisms at every node/time step:

1. **conservative exchange** — antisymmetric pair flux with a discrete maximum-principle bound;
2. **wind transport** — conservative upwind transport, enabled only when wind is actually observed;
3. **source/sink forcing** — bounded unresolved local forcing;
4. **bounded residual** — deliberately smaller residual capacity for dynamics not captured above.

A compact recurrent latent state performs multi-step "dreaming". A state-dependent router chooses the mechanism mixture. Training uses predictive latent/temperature loss plus SIGReg. The architecture is intentionally sparse: physical locality is encoded by kNN edges with direction and distance rather than dense all-pairs attention.

This is a research hypothesis, not a pre-declared SOTA result. Performance claims are made only from frozen experiment artifacts.

## Repository

```text
SAM-WM/
├── README.md
├── train.py
├── eval.py
├── fortyguard_check.py
├── plot.py
├── summarize.py
├── config/
├── src/coolworld/
│   ├── samwm.py        # core contribution
│   ├── graph.py
│   ├── benchmarks.py
│   ├── experiment.py
│   ├── candra.py
│   ├── fortyguard.py
│   ├── evidence.py
│   └── app.py
├── static/             # 3D evidence UI
├── tests/
├── notebooks/
└── docs/
```

The core contribution remains easy to find in `src/coolworld/samwm.py`, following the same codebase discipline that makes LeWorldModel's contribution easy to inspect in `jepa.py`.

## Real benchmark protocol

**ID training/validation/final test:** Freiburg urban air-temperature sensor network, DOI `10.5281/zenodo.12732565`.

**OOD-1:** Novi Sad urban sensor network, DOI `10.5281/zenodo.7738094`, evaluated zero-shot with no target fine-tuning and no OOD recalibration.

**OOD-2:** FAIRUrbTemp, DOI `10.48620/93247`, evaluated zero-shot on an unseen city. Observation-level QC flags are filtered from scoring; the original dataset intentionally retains flagged measurements, so filtering is the evaluator's responsibility.

The cross-city representation uses city-centred local x/y and relative elevation, not raw absolute latitude/longitude. Missing relative humidity is accompanied by an explicit modality-availability mask. The admissible source scale is derived only from observed **training** temperature increments. Forecast windows must also remain hourly-contiguous; gaps are never silently treated as one physical time step.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
```

Python 3.11 and 3.12 are supported. CI checks both.

## Training

The frozen configuration is `config/train.yaml`.

```bash
python train.py --seed 0 --out artifacts/freiburg
python train.py --seed 1 --out artifacts/freiburg
python train.py --seed 2 --out artifacts/freiburg
```

Model selection uses Freiburg validation MAE only. Do not inspect final-test/OOD results while tuning.

## Validation and held-out evaluation

Validation can be rerun during development:

```bash
python eval.py \
  --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data freiburg --split validation \
  --out artifacts/eval/seed_0
```

Validation and final-test artifacts have different filenames, so the final evaluation cannot overwrite the evidence used for model selection. Held-out metric computation is deliberately gated: `eval.py` atomically writes a receipt before computing held-out metrics and refuses to reopen the same held-out dataset in that output directory.

```bash
python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data freiburg --split heldout --open-heldout --out artifacts/eval/seed_0

python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data novisad --split heldout --open-heldout --out artifacts/eval/seed_0

python eval.py --checkpoint artifacts/freiburg/seed_0/best.pt \
  --data fairurbtemp --root /path/to/extracted/FAIRUrbTemp --city <city> \
  --split heldout --open-heldout --out artifacts/eval/seed_0
```

Repeat with seeds 1 and 2 using separate `artifacts/eval/seed_<n>` directories, then aggregate:

```bash
python summarize.py --root artifacts/eval --out artifacts/summary.json
```

The aggregator groups by evaluation (`freiburg_validation`, `freiburg_heldout`, `novisad_heldout`, `fairurbtemp_heldout`) so development evidence cannot be mixed into final/OOD evidence. Metrics include MAE, RMSE, bias, p95 absolute error, horizon-wise MAE/RMSE, conformal coverage, surprise, parameter count, and inference latency.

For Kaggle, use `notebooks/SAM_WM_KAGGLE.ipynb` and follow `docs/KAGGLE_PROTOCOL.md`. The notebook resolves one exact GitHub source SHA, verifies the repository, records checkpoint/config/validation hashes before held-out evaluation, and uses a Kaggle Secret named `GITHUB_TOKEN` only when private-repository access is required.

## FortyGuard evidence

The FortyGuard API key is never committed. Configure it locally/server-side:

```bash
export FORTYGUARD_API_KEY='...'
```

A real heatmap request is:

```bash
python fortyguard_check.py \
  --date 2026-08-27 --time 12:00 \
  --aoi examples/sanjose_aoi.geojson --granularity 100
```

The client is asynchronous and crash-resumable: it persists the exact request intent before POST, saves the provider `activity_id`, resumes polling the same activity after restart, content-addresses completed responses, and never automatically re-posts an ambiguous request.

## 3D UI

Run locally:

```bash
make serve
```

or with Docker:

```bash
docker build -t sam-wm .
docker run --rm -p 8000:8000 -e FORTYGUARD_API_KEY sam-wm
```

The browser interface separates three truth states:

- **OBSERVED** — real FortyGuard/map evidence;
- **REAL REPLAY** — only attributable intervention pre/post/control evidence;
- **PREDICTED FUTURE** — only after a frozen checkpoint, calibration, compatible context, and CANDRA action evidence are promoted.

The UI never replaces an unavailable upstream field with sample temperatures and never labels a modelled value as observed.

## CANDRA

`src/coolworld/candra.py` provides the conservative action-evidence boundary. Its temporal block-bootstrap difference-in-differences estimator is only valid after the treated/control assumptions have been independently justified. Low support, overlapping uncertainty, or absent intervention evidence produces an abstention.

## FortyGuard Hackathon'26

Current official guidance reviewed through 27 Aug 2026 prioritizes:

- Impact & Relevance — 40%
- Technical Execution — 35%
- Innovation — 15%
- Communication — 10%

Submission deadline: **30 Aug 2026, 11:59 PM GST**. Final delivery requires the official form, a public/no-login live demo, a demo video no longer than 3 minutes, and a code repository. See `docs/HACKATHON.md` and `docs/KAGGLE_PROTOCOL.md`.

## Claim policy

Until frozen experiments prove otherwise, this repository does **not** claim:

- state of the art;
- human-level/child-level general intelligence;
- AGI/ASI;
- a guaranteed urban cooling effect;
- top-0.01% hackathon placement or a specific judging score.

What it does claim is testable: sparse physically typed dynamics, explicit missing-modality handling, training-only physical bounds, zero-shot cross-city evaluation, calibrated uncertainty, auditable FortyGuard evidence, and causal abstention when intervention support is insufficient.

## License

MIT. See `LICENSE`.
