# SAM-WM

### Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence

**SAM-WM** is the single research model in this repository. The codebase is organized so the model,
real benchmark protocol, provider-evidence path, causal safety boundary, and CoolWorld 3D application
stay inspectable without mixing learned artifacts into source control.

> **Scientific boundary.** SAM-WM predicts urban thermal dynamics and supports evidence-constrained
> planning. Physical interventions cool the environment; the software does not itself lower
> temperature. Observational forecasts are never relabelled as causal cooling effects. Unsupported
> interventions return **ABSTAIN / MODEL_NOT_READY**.

## Model

A city is represented as a sparse physical mental map. At each forecast step SAM-WM composes four
typed mechanisms:

1. **Conservative exchange** — antisymmetric pair flux with a discrete maximum-principle bound.
2. **Wind transport** — conservative upwind transport, enabled only when wind is actually observed.
3. **Source/sink forcing** — bounded unresolved local forcing derived from Freiburg training data.
4. **Bounded residual** — deliberately limited residual capacity.

A compact recurrent latent state performs multi-step rollout. A state-dependent router composes the
mechanisms, while uncertainty and surprise expose when predictions become less trustworthy. Training
uses predictive latent/temperature learning plus SIGReg. Spatial execution is sparse, approximately
`O(E)`, with deterministic `cKDTree` neighbour construction rather than dense all-pairs distance
matrices.

SAM-WM is an original research hypothesis. SIGReg is attributed to the public LeWorldModel work;
SAM-WM does not claim authorship of SIGReg or copy LeWM's pixel architecture.

## Repository

```text
SAM-WM/
├── README.md
├── train.py                 # one SAM-WM training run
├── research.py              # frozen five-seed SAM-WM development suite
├── eval.py                  # validation + one-time held-out / zero-shot OOD gates
├── promote.py               # validation-only seed selection + frozen promotion
├── provider_replay.py       # real FortyGuard deployment-domain replay gate
├── fortyguard_check.py      # bounded real provider request
├── fortyguard_collect.py    # explicit-credit consecutive provider evidence collector
├── candra_fit.py            # independent intervention-evidence gate
├── summarize.py
├── plot.py
├── config/
├── src/coolworld/
│   ├── samwm.py             # core invention
│   ├── graph.py
│   ├── benchmarks.py
│   ├── experiment.py
│   ├── research.py
│   ├── promotion.py
│   ├── provider.py
│   ├── deployment.py
│   ├── candra.py
│   ├── action_evidence.py
│   ├── fortyguard.py
│   ├── evidence.py
│   └── app.py
├── static/                  # CoolWorld 3D browser interface
├── notebooks/
├── tests/
└── docs/
```

The core research contribution remains concentrated in `samwm.py`; experiment and deployment
machinery are separate so the scientific idea is easy to inspect.

## Real benchmark contract

The repository contains loaders, checksums, chronology, graph construction, train-only
normalization, QC handling, held-out gates, metrics, and OOD rules. Dataset binaries and learned
checkpoints are deliberately not committed.

**Development / ID benchmark — Freiburg**

- Freiburg urban air-temperature sensor network — DOI `10.5281/zenodo.12732565`.
- Train-only normalization and train-only unresolved-source bound estimation.
- Validation-only early stopping and deployment-seed selection.
- Final test opened once only after the source/config/checkpoint freeze.

**OOD-1 — Novi Sad**

- Novi Sad urban sensor network — DOI `10.5281/zenodo.7738094`.
- Zero-shot only: no target fine-tuning, target-driven hyperparameter selection, or OOD
  recalibration.

**OOD-2 — FAIRUrbTemp**

- FAIRUrbTemp — DOI `10.48620/93247`.
- One unseen city chosen from metadata/coverage criteria before SAM-WM metrics are viewed.
- Observation-level `qc=` flags are excluded from scoring.
- Zero-shot only: no target fine-tuning or OOD recalibration.

Cross-city inputs use city-centred local x/y and relative elevation rather than absolute
latitude/longitude. Missing relative humidity has an explicit availability channel. Forecast windows
must remain hourly-contiguous.

## One-model frozen research suite

Only **SAM-WM** is trained and evaluated in the benchmark pipeline. There is no external baseline
model zoo and no persistence/trend surrogate presented as another research model.

Frozen seeds:

```text
17, 29, 42, 73, 101
```

Run the full SAM-WM development suite:

```bash
python research.py --out artifacts/research
```

The output layout is deliberately flat:

```text
artifacts/research/
├── seed_17/
├── seed_29/
├── seed_42/
├── seed_73/
├── seed_101/
└── PRE_FREEZE_MANIFEST.json
```

Each seed contains the full SAM-WM checkpoint, training history, resolved configuration, Freiburg
dataset manifest, and validation metrics. `PRE_FREEZE_MANIFEST.json` hashes only those five SAM-WM
validation artifacts and certifies that Freiburg held-out and both OOD targets were not accessed.

Deployment seed selection is also validation-only:

```bash
python promote.py preselect
```

The selected seed is fixed before final/OOD results are opened.

## Installation and verification

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
```

CI verifies Python 3.11 and 3.12 independently, compiles all executable source, runs Ruff lint and
format checks, and runs the test suite.

## Kaggle execution

Use `notebooks/SAM_WM_KAGGLE.ipynb` and `docs/KAGGLE_PROTOCOL.md`.

The notebook executes one model in one frozen order:

1. resolve one exact GitHub source SHA and run repository verification;
2. require a real Kaggle GPU;
3. train full SAM-WM across the five frozen seeds on Freiburg train/validation only;
4. inspect Freiburg validation evidence only and preselect one deployment seed;
5. write `artifacts/FREEZE_MANIFEST.json`;
6. open Freiburg final test once per seed;
7. evaluate Novi Sad zero-shot;
8. evaluate the preregistered FAIRUrbTemp unseen city zero-shot;
9. aggregate machine-readable evidence and figures;
10. finalize the already-preselected SAM-WM deployment bundle.

A material architecture, objective, preprocessing, QC, split, graph, or hyperparameter change after
the freeze defines a new research version and requires a new untouched confirmatory run.

## Metrics

Saved benchmark artifacts include MAE, RMSE, bias, p95 absolute error, horizon-wise MAE/RMSE,
split-conformal coverage, mean surprise, parameter count, inference latency, and observed target
count. Every reported number must come from a saved machine-readable artifact.

## FortyGuard evidence

The API key is never committed.

```bash
export FORTYGUARD_API_KEY='...'
python fortyguard_check.py \
  --date 2026-08-27 --time 12:00 \
  --aoi examples/sanjose_aoi.geojson --granularity 100
```

The provider client is asynchronous, crash-resumable and fail-closed: request intent is persisted
before POST, provider `activity_id` is saved, ambiguous POSTs are not blindly repeated, completed
responses are content-addressed, and provenance is preserved for the UI.

For a consecutive real replay timeline, `fortyguard_collect.py` is dry-run by default. Potentially
credit-consuming requests occur only with the explicit `--confirm-credit-usage` flag.

A promoted SAM-WM checkpoint is not treated as deployment-ready merely because Freiburg/Novi
Sad/FAIRUrbTemp evaluation succeeded. `provider_replay.py` separately checks the frozen model on
recorded same-grid real FortyGuard TCM evidence before predicted-live mode is enabled.

## CANDRA action-evidence gate

SAM-WM forecasting and intervention causality remain separate. `candra_fit.py` requires a genuine
source treated/control study and an independent transfer treated/control study before an action can
be marked transfer-supported. Ordinary observational predictions cannot create a non-zero cooling
effect.

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

The browser separates **OBSERVED**, **REAL REPLAY**, and **PREDICTED FUTURE** truth states. Predicted
future remains locked until the frozen checkpoint, calibration, benchmark evidence, compatible real
provider context, and provider-replay gate are present. Intervention views remain locked without
independent CANDRA action evidence.

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

Before frozen experiments, this repository does **not** claim universal SOTA superiority,
human-child-level general intelligence, AGI/ASI, guaranteed causal cooling, planetary-scale safety,
or a guaranteed hackathon score. The source code makes a specific SAM-WM hypothesis testable; the
real benchmark evidence determines which claims survive.

## License

MIT. See `LICENSE`.
