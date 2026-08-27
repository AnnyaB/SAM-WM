# SAM-WM

### Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence

**SAM-WM** is Krsna's compact research world model and CoolWorld deployment stack for the
FortyGuard Hackathon'26. The repository is organized so that the scientific contribution,
benchmark protocol, real-provider evidence path, and 3D application are inspectable without
mixing training artifacts into source control.

> **Scientific boundary.** SAM-WM predicts urban thermal dynamics and supports
> evidence-constrained planning. It does not convert an observational forecast into a causal
> cooling effect. Unsupported interventions return **ABSTAIN / MODEL_NOT_READY** rather than a
> fabricated temperature reduction.

## Model

A city is represented as a sparse physical mental map. At each node and forecast step the model
combines four typed mechanisms:

1. **Conservative exchange** — antisymmetric pair flux with a discrete maximum-principle bound.
2. **Wind transport** — conservative upwind transport, enabled only when wind is actually present.
3. **Source/sink forcing** — bounded unresolved local forcing.
4. **Bounded residual** — deliberately limited residual capacity.

A compact recurrent latent state performs multi-step rollout; a state-dependent router selects the
mechanism mixture. Training combines predictive latent/temperature learning with SIGReg. Spatial
execution is sparse, approximately `O(E)`, rather than dense all-pairs attention.

The design is an original SAM-WM research hypothesis. SIGReg is attributed to the public
LeWorldModel work; SAM-WM does not claim authorship of SIGReg or copy LeWM's pixel architecture.

## Repository

```text
SAM-WM/
├── README.md
├── train.py                 # one full SAM-WM training run
├── research.py              # pre-freeze baselines, 7 ablations, controls, 5 seeds
├── eval.py                  # validation + one-time held-out / zero-shot OOD gates
├── fortyguard_check.py      # bounded real FortyGuard request
├── summarize.py
├── plot.py
├── config/
├── src/coolworld/
│   ├── samwm.py             # core contribution
│   ├── graph.py
│   ├── benchmarks.py
│   ├── experiment.py
│   ├── candra.py
│   ├── fortyguard.py
│   ├── evidence.py
│   └── app.py
├── static/                  # CoolWorld 3D browser interface
├── notebooks/
├── tests/
└── docs/
```

This follows the same codebase discipline that makes LeWorldModel's contribution easy to find:
the core model stays compact, while experiment and deployment concerns remain separate.

## Real benchmark contract

The source repository contains the loaders, checksums, graph construction, chronology, windowing,
normalization rules, QC handling, held-out gates, metrics, and OOD protocol. Dataset binaries and
learned checkpoints are deliberately not committed.

**ID train / validation / final test**

- Freiburg urban air-temperature sensor network — DOI `10.5281/zenodo.12732565`.
- Train-only normalization and train-only physical source-bound estimation.
- Validation-only early stopping.
- Final test opened only after the experiment freeze.

**OOD-1**

- Novi Sad urban sensor network — DOI `10.5281/zenodo.7738094`.
- Zero-shot; no target fine-tuning and no OOD recalibration.

**OOD-2**

- FAIRUrbTemp — DOI `10.48620/93247`.
- One unseen city preregistered before results are viewed.
- Observation-level `qc=` flags are excluded from scoring.
- Zero-shot; no target fine-tuning and no OOD recalibration.

Cross-city inputs use city-centred local x/y and relative elevation rather than absolute
latitude/longitude. Missing relative humidity has an explicit availability channel. Forecast
windows must remain hourly-contiguous.

## Pre-freeze research suite

`research.py` is intentionally **validation-only**. It cannot open Freiburg held-out or either OOD
target. It runs the complete model-selection/diagnostic suite before the final benchmark is touched.

Frozen research seeds:

```text
17, 29, 42, 73, 101
```

Structural ablations, each retrained from scratch on every seed:

```text
no_mental_map
no_exchange
unconstrained_exchange
no_source_sink
no_residual
uniform_router
no_temporal_memory
```

Objective controls:

```text
no_sigreg
temperature_only
```

The `temperature_only` control removes the predictive latent term while retaining the same
temperature likelihood and SIGReg, so the predictive-state objective is tested rather than merely
described.

Validation sanity baselines are computed under the exact same SAM-WM window contract:

```text
persistence
linear_trend
daily_persistence
```

These are sanity baselines, not relabelled external SOTA reproductions. Any future paper comparison
against external author code must preserve the upstream method name, exact source revision, license,
adapter, split, and deviations.

Run all pre-freeze research:

```bash
python research.py --stage all-pre-freeze --out artifacts/research
```

It produces `artifacts/research/PRE_FREEZE_MANIFEST.json` containing hashes of the frozen
development/validation evidence and explicitly records that no held-out/OOD target was accessed.

## Installation and verification

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
```

CI verifies Python 3.11 and 3.12, compiles the source and all executable scripts, runs Ruff lint and
format checks, and runs the test suite.

## Kaggle execution

Use `notebooks/SAM_WM_KAGGLE.ipynb` and `docs/KAGGLE_PROTOCOL.md`.

The notebook:

1. resolves one exact GitHub source SHA and verifies the repository;
2. requires a real Kaggle GPU;
3. executes the validation-only research suite across five frozen seeds;
4. lets the researcher inspect Freiburg validation evidence only;
5. writes `artifacts/FREEZE_MANIFEST.json`;
6. opens Freiburg final test once per seed;
7. evaluates Novi Sad zero-shot;
8. evaluates preregistered FAIRUrbTemp unseen-city zero-shot;
9. aggregates machine-readable evidence and figures.

A material architecture/protocol change after the freeze defines a new research version and requires
a new untouched confirmatory benchmark.

## Metrics

Saved benchmark artifacts include:

- MAE and RMSE;
- bias and p95 absolute error;
- horizon-wise MAE/RMSE;
- split-conformal coverage;
- mean model surprise;
- parameter count;
- inference latency;
- observed target count.

Every final number must come from a saved artifact. Do not manually type metrics into the README,
figures, or demo.

## FortyGuard evidence

The API key is never committed.

```bash
export FORTYGUARD_API_KEY='...'
python fortyguard_check.py \
  --date 2026-08-27 --time 12:00 \
  --aoi examples/sanjose_aoi.geojson --granularity 100
```

The client is asynchronous, crash-resumable and fail-closed: request intent is persisted before
POST, provider `activity_id` is saved, ambiguous POSTs are not blindly repeated, completed
responses are content-addressed, and provenance is preserved for the UI.

## CANDRA action-evidence gate

`src/coolworld/candra.py` contains the conservative causal-evidence boundary. Its temporal
block-bootstrap difference-in-differences estimator is only valid when a genuine treated/control
design and its assumptions are independently justified.

Observational forecasting data alone cannot identify shade/canopy/pavement cooling effects. Low
support, overlapping uncertainty, missing transfer evidence, or absent intervention data produces
an abstention.

## CoolWorld 3D UI

Run locally:

```bash
make serve
```

or:

```bash
docker build -t sam-wm .
docker run --rm -p 8000:8000 -e FORTYGUARD_API_KEY sam-wm
```

The browser separates three truth states:

- **OBSERVED** — real FortyGuard/map evidence;
- **REAL REPLAY** — immutable attributable recorded evidence;
- **PREDICTED FUTURE** — only after a frozen trained checkpoint and calibration are promoted.

The interface contains the real 3D map/building layer, thermal field, AOI, timeline, synchronized
baseline/intervention future views, uncertainty/support, pipeline readiness, evidence/activity
hashes and agent console. It never generates placeholder temperatures or labels model output as
observed.

The final learned checkpoint is intentionally not committed before training. After frozen
evaluation, one selected checkpoint/calibration bundle is promoted into deployment and the UI is
visually tested end-to-end before public hosting and the final demo video.

## FortyGuard Hackathon'26

Official guidance reviewed through **27 Aug 2026 (Day 8)**:

- Impact & Relevance — 40%
- Technical Execution — 35%
- Innovation — 15%
- Communication — 10%
- Deadline — **30 Aug 2026, 11:59 PM GST**
- Final package — submission form, public/no-login live demo, working demo video no longer than
  3 minutes with voiceover, and code repository.

See `docs/HACKATHON.md`.

## Claim policy

Before frozen experiments, this repository does **not** claim:

- universal or state-of-the-art superiority;
- human-child-level general intelligence;
- AGI/ASI;
- a guaranteed causal cooling effect;
- planetary-scale deployment safety;
- a guaranteed hackathon score or placement.

What the source code already makes testable is narrower and stronger: compact sparse world-model
dynamics, physically typed mechanism composition, explicit missing-modality handling, train-only
physical bounds, one-time held-out evaluation, two real zero-shot OOD domains, uncertainty,
surprise, intervention abstention, auditable provider evidence, and a real 3D deployment surface.

## License

MIT. See `LICENSE`.
