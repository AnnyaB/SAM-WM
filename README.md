# SAM-WM · CoolWorld

### Sparse Adaptive Mechanism World Model for evidence-bounded urban thermal intelligence

**CoolWorld** is a real-data urban thermal decision-support system built around one research model: **SAM-WM**. It connects immutable FortyGuard thermal evidence to a compact mechanism-structured world model, visualizes short-horizon city temperature evolution in 3D, ranks persistent future hotspots, exposes uncertainty, and **abstains** when operational or causal evidence is insufficient.

> **Truth boundary.** Software does not physically cool a city or the planet. Trees, shade, reflective materials, building retrofits, water systems, and other physical interventions do. CoolWorld helps decide **where and when to investigate intervention**, then requires treated/control evidence before claiming a numerical cooling effect.

The public demo deliberately separates four states:

1. **Real evidence** — recorded FortyGuard TCM observations.
2. **Research forecast** — the exact frozen SAM-WM +1…+6 h prediction on verified real context.
3. **Operational certification** — a separate provider-replay gate. The current frozen run narrowly fails the fixed coverage requirement and remains labelled as such.
4. **Causal action effect** — unavailable until independent treated/control intervention evidence exists.

No state is made green by changing a threshold after evaluation.

---

## 60-second product flow

```text
FORTYGUARD REAL THERMAL EVIDENCE
65 consecutive recorded frames · one 36-tile San José grid
                    │
                    ▼
              OBSERVE IN 3D
real buildings + real ground thermal field + provenance
                    │
                    ▼
                 SAM-WM
48 h real context → sparse mechanism world model → +1…+6 h future
                    │
                    ▼
          FUTURE HOTSPOT PRIORITY
forecast heat + persistence + uncertainty
                    │
                    ▼
             ENGINEERING REVIEW
candidate trees / shade / reflective surface / other physical action
                    │
                    ▼
          MEASURE TREATED VS CONTROL
FortyGuard / field sensing / independent causal evidence
                    │
                    ▼
             VALIDATE OR ABSTAIN
```

For a first-time user, the UI is intentionally ordered as **Observe → Forecast → Prioritize → Evidence**. Developer-only live API, AOI, and causal-action controls are placed under advanced sections.

---

## What SAM-WM is

SAM-WM is a **Sparse Adaptive Mechanism World Model** for multi-step urban thermal prediction. A city is represented as a sparse physical graph rather than a dense all-pairs map. At each forecast step the model composes typed mechanisms:

1. **Conservative exchange** — antisymmetric pair flux with a discrete maximum-principle bound.
2. **Wind transport** — conservative upwind transport when wind is actually available; exactly disabled otherwise.
3. **Source/sink forcing** — bounded unresolved local forcing.
4. **Bounded residual** — intentionally limited residual capacity.
5. **Adaptive mechanism routing** — state-dependent composition of the available mechanisms.
6. **Recurrent latent dynamics** — multi-step rollout with an explicit latent state.
7. **Uncertainty / surprise** — prediction support is exposed instead of silently treated as certainty.

Spatial execution is sparse, approximately `O(E)`, with deterministic k-nearest-neighbour graph construction. Missing relative humidity and other unavailable modalities are represented explicitly rather than filled with invented measurements.

SAM-WM is the original research hypothesis in this repository. **SIGReg** is attributed to the public LeWorldModel work; this repository does not claim authorship of SIGReg or copy LeWM's pixel architecture.

### Why this is a world-model research problem

The model is trained to roll a compact internal state forward and predict future thermal fields over several hours, not merely classify a static image. The research questions are therefore about **state representation, local interaction structure, mechanism composition, uncertainty, cross-city transfer, and reliable rollout under missing modalities**.

The evidence here supports those concrete claims. It does **not** establish human-child-level general intelligence, AGI/ASI, universal SOTA superiority, causal urban cooling, or planetary-scale validation.

---

## Repository architecture

```text
SAM-WM/
├── README.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── .dockerignore
│
├── train.py                 # one SAM-WM training run
├── research.py              # frozen five-seed SAM-WM development suite
├── eval.py                  # final + zero-shot OOD evaluation
├── promote.py               # validation-only selection / immutable promotion
├── provider_replay.py       # operational-transfer gate on real provider evidence
├── candra_fit.py            # independent causal action-evidence gate
├── fortyguard_check.py      # bounded provider request
├── fortyguard_collect.py    # explicit-credit, crash-resumable collection
├── verify_runtime.py        # offline runtime/evidence integrity audit
├── summarize.py
├── plot.py
│
├── src/coolworld/
│   ├── samwm.py             # core SAM-WM invention
│   ├── graph.py             # sparse city graph
│   ├── benchmarks.py        # Freiburg / Novi Sad / FAIRUrbTemp
│   ├── experiment.py        # normalization, training/eval mechanics
│   ├── research.py          # frozen research suite helpers
│   ├── promotion.py         # selection/freeze/promotion contracts
│   ├── provider.py          # canonical provider timeline + replay validation
│   ├── deployment.py        # frozen real-context inference
│   ├── product_api.py       # product readiness/evidence/hotspot APIs
│   ├── candra.py
│   ├── action_evidence.py
│   ├── fortyguard.py        # provider client
│   ├── evidence.py          # hashes / provenance
│   └── app.py               # FastAPI runtime
│
├── static/
│   ├── index.html           # judge/user-first UI
│   ├── app.js               # real 3D thermal renderer + SAM-WM forecast
│   ├── product.js           # guided flow, hotspot plan, evidence panels
│   ├── styles.css
│   └── product.css
│
├── artifacts/               # only immutable demo/runtime evidence is tracked
├── notebooks/
├── tests/
└── docs/
```

Research/training scratch outputs remain ignored. The hackathon runtime release intentionally tracks the **promoted checkpoint, its immutable manifests, evaluation summary, provider replay record, and recorded provider evidence** so a judge or deployment can reproduce the exact no-login demo without spending new API credits.

---

## Frozen research protocol

Only **SAM-WM** is trained and evaluated in the benchmark pipeline. The five seeds are repeated executions of the same model family, not competing models:

```text
17, 29, 42, 73, 101
```

The protocol is intentionally ordered:

1. Train/validate SAM-WM on Freiburg only.
2. Select deployment seed using **Freiburg validation only**.
3. Seal source/config/checkpoint hashes.
4. Open Freiburg final held-out once.
5. Evaluate Novi Sad zero-shot with no fine-tuning or OOD recalibration.
6. Select the FAIRUrbTemp city from metadata/coverage only, before model metrics.
7. Evaluate preregistered Turku zero-shot with no fine-tuning or OOD recalibration.
8. Promote the already-selected checkpoint.
9. Run a separate real FortyGuard provider-replay compatibility gate.
10. Keep causal intervention effects locked unless independent treated/control evidence is supplied.

Selected deployment seed:

```text
42
```

Exact promoted checkpoint SHA-256:

```text
2be783f8a3b7f755a72a98949397c67dfec3a66a6400d8b98e1e732e0d8b708f
```

A material architecture, objective, preprocessing, QC, split, graph, or hyperparameter change after the freeze is a **new research version** and requires a new untouched confirmatory run.

---

## Real benchmark evidence

All numbers below are read from tracked machine-readable artifacts; they are not UI placeholders.

| Domain | Protocol | Five-seed MAE (°C) | Five-seed RMSE (°C) | Conformal coverage |
|---|---|---:|---:|---:|
| Freiburg | final ID held-out | **1.4515 ± 0.0167** | **2.0483 ± 0.0081** | **90.45% ± 1.26%** |
| Novi Sad | zero-shot OOD-1 | **1.4675 ± 0.0286** | **2.1575 ± 0.0370** | **89.59% ± 1.11%** |
| Turku / FAIRUrbTemp | zero-shot OOD-2 | **1.5549 ± 0.0425** | **2.1944 ± 0.0574** | **88.55% ± 1.93%** |

The frozen model has **117,705 parameters** in these runs. Horizon error grows with rollout length and is reported explicitly in `artifacts/summary.json` rather than hidden behind a single average.

### FortyGuard operational replay

Real provider evidence contains **65 compatible consecutive hourly frames** on one **36-tile** San José grid. The same frozen SAM-WM was replayed without retraining.

```text
Provider replay protocol: SAM_WM_FORTYGUARD_REPLAY_V2
Windows:                  12
MAE:                      2.047516 °C
RMSE:                     2.501741 °C
MAE / conformal radius:   0.637548      (passes ≤ 1.0 criterion)
Empirical coverage:       79.899691%
Fixed minimum coverage:   80.000000%
Operational gate:         FAIL
```

The coverage miss is about **0.1003 percentage points**. The threshold is **not** lowered after seeing the result. Therefore:

- the frozen SAM-WM **research forecast remains inspectable** on the real context;
- it is labelled **not operationally certified**;
- numerical intervention effects remain locked;
- the result is not presented as causal cooling evidence.

This distinction is exposed programmatically by `/api/product-status` and visually by the UI.

---

## CoolWorld 3D interface

### Observe

The app automatically loads the immutable recorded FortyGuard timeline. Users can inspect:

- the real 3D San José basemap/buildings;
- real provider GeoJSON thermal tiles;
- exact timestamps, provider activity IDs and content hashes;
- temperature histogram and summary statistics;
- all 65 recorded frames through the timeline.

Between-frame animation is labelled **visual interpolation**, not a new measurement.

### Forecast

Selecting **SAM-WM FORECAST** runs the exact frozen checkpoint on the verified real context and renders:

- +1…+6 h thermal fields on the real provider grid;
- forecast animation on the 3D city map;
- mean future trajectory;
- split-conformal uncertainty radius;
- checkpoint and context provenance;
- explicit `MODEL PREDICTION — NOT OBSERVED` truth state.

The research forecast endpoint is intentionally separate from operational `/api/forecast`.

### Prioritize future hotspots

`GET /api/hotspots` ranks the forecast grid by future temperature and persistence across the six horizons. The product panel adds a second 3D priority view:

- **yellow → orange → red** means relative priority *within the selected forecast-hotspot set*;
- hover/cards show the true current and predicted °C;
- no temperature is altered to make the visualization dramatic;
- candidate actions are suggestions for engineering review only;
- action `effect_c` is deliberately `null` until causal evidence exists.

This makes the visual answer useful without confusing **absolute temperature** with **relative intervention priority**.

### Evidence

The interface reads its benchmark and provider-replay values from tracked JSON artifacts. A judge can see exactly why a research forecast is available while operational/causal gates remain closed.

---

## Local quick start

Python 3.11 or 3.12 is supported.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,app]'
make verify
python verify_runtime.py
make serve
```

Open:

```text
http://127.0.0.1:8000
```

The recorded demo uses **zero new provider requests**.

### Runtime truth audit

```bash
python verify_runtime.py
```

The audit verifies the checkpoint hash, promotion manifest, provider timeline continuity/grid identity, replay criteria, and product truth states. It never calls FortyGuard.

---

## Docker / Hugging Face Spaces

Build and run locally:

```bash
docker build -t sam-wm-coolworld .
docker run --rm -p 8000:7860 sam-wm-coolworld
```

Then open `http://127.0.0.1:8000`.

The container:

- runs as a non-root user;
- includes only the immutable runtime evidence required for the reproducible demo;
- exposes `/api/health` for health checks;
- defaults to port `7860`, compatible with a Docker Hugging Face Space;
- defaults to `COOLWORLD_LIVE_API_ENABLED=0`.

See `docs/HF_SPACE.md` and `docs/PRODUCTION.md`.

---

## Optional live FortyGuard mode

The public demo should use recorded evidence by default. A provider key is **never** committed and is never exposed to browser JavaScript.

A new live provider request requires **both** server-side controls:

```bash
export FORTYGUARD_API_KEY='...'
export COOLWORLD_LIVE_API_ENABLED=1
```

Then start the app. Without the explicit live flag, `/api/fortyguard/heatmap` returns `403 LIVE_PROVIDER_API_DISABLED` even if a key exists. This prevents a public UI from accidentally spending provider allocation.

For batch collection, `fortyguard_collect.py` remains dry-run by default; credit-consuming collection requires its explicit confirmation flag.

---

## API truth states

| Endpoint | Meaning | Can claim causal cooling? |
|---|---|---|
| `GET /api/product-status` | separated real/model/operational/causal readiness | No |
| `GET /api/evidence-summary` | immutable benchmark + provider replay summary | No |
| `GET /api/evidence/timeline` | recorded real provider fields | Observational only |
| `POST /api/forecast-preview` | frozen SAM-WM research forecast | No |
| `GET /api/hotspots` | future forecast-hotspot priority | No |
| `POST /api/forecast` | operational forecast, only after replay gate passes | No |
| `POST /api/counterfactual` | supported action effect, only after replay + CANDRA evidence | Only within supplied evidence contract |
| `POST /api/fortyguard/heatmap` | optional live provider request | Observational only |

The status API intentionally distinguishes:

```json
{
  "real_provider_evidence_ready": true,
  "model_bundle_promoted": true,
  "research_forecast_ready": true,
  "operational_certified": false,
  "causal_action_ready": false
}
```

Those values are not contradictory; they represent different scientific and deployment gates.

---

## CANDRA action-evidence gate

Forecasting and intervention causality remain separate. `candra_fit.py` requires a genuine source treated/control study and an independent transfer treated/control study before an action can be marked transfer-supported. Ordinary observational forecasts cannot create a non-zero cooling effect.

The current public runtime therefore **does not** display fabricated values such as “tree canopy cools this tile by X °C.” Instead it identifies forecast-persistent hotspots and proposes action categories for investigation.

---

## Reproducibility

```bash
make verify
```

runs compilation, Ruff lint, Ruff format check, and the Python test suite. CI repeats verification on Python 3.11 and 3.12 and also checks browser JavaScript, immutable runtime contracts, and the Docker application smoke path.

Research artifacts are content-addressed or hash-linked wherever possible. The selected seed, freeze manifests, promotion manifest, final/OOD summary, provider activity/content hashes, and replay record are all preserved.

---

## Production/scaling notes

The hackathon deployment is intentionally small and deterministic, but the architecture separates concerns so it can evolve:

- model inference is isolated from HTTP/UI logic;
- provider ingestion is isolated from the model;
- product readiness/evidence APIs are isolated in `product_api.py`;
- immutable demo evidence can later move to object storage;
- the current one-entry process-local forecast cache can later be replaced by Redis or a content-addressed shared cache;
- live provider operations are feature-gated and server-side only;
- horizontal workers can share the same immutable model/evidence bundle, while shared mutable state should move to external storage before multi-replica production.

See `docs/PRODUCTION.md`.

---

## FortyGuard Hackathon'26

Submission-oriented design follows the supplied hackathon emphasis on **Impact & Relevance, Technical Execution, Innovation, and Communication**. The final package should include a public/no-login live app, repository, and a working demo video no longer than three minutes.

The product story is intentionally practical:

> **CoolWorld turns real urban thermal evidence into short-horizon world-model forecasts and future-hotspot priorities, so city teams can focus physical cooling investigation where heat is predicted to persist, while the software refuses to invent unsupported intervention effects.**

See `docs/DEMO.md` for the short judge walkthrough.

---

## Claim policy

### Supported by the current artifacts

- real-data urban temperature forecasting;
- Freiburg final held-out evaluation;
- zero-shot cross-city evaluation on Novi Sad and preregistered Turku;
- +1…+6 h multi-step rollout;
- explicit missing-modality handling;
- uncertainty-aware prediction;
- compact 117,705-parameter model in the frozen runs;
- real FortyGuard evidence integration;
- non-causal future-hotspot prioritization;
- fail-closed operational and causal gates.

### Not supported by the current artifacts

- universal SOTA superiority;
- human-child-level general intelligence;
- AGI/ASI;
- guaranteed or measured cooling from a proposed intervention without treated/control evidence;
- planetary-scale validation or a claim that software itself “cools Earth”;
- guaranteed hackathon score or publication acceptance.

See `docs/CLAIMS.md`.

---

## License

MIT. See `LICENSE`.
