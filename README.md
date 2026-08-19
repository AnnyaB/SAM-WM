# SAM-WM / CoolWorld-SAM

**Action-conditioned world modelling for real urban cooling decisions.**

This is the private research + product repository for the FortyGuard Hackathon'26 build.
The system consumes real thermal/environmental evidence, learns a compact predictive world
model from real trajectories, estimates uncertainty/support for cooling interventions, ranks
physical actions, and renders observed and predicted futures in a 3D browser application.

## What it does — and what it does not do

The software does not physically lower city temperature by itself. Real cooling occurs after a
city/operator deploys a physical action such as shade, canopy, reflective material, scheduling,
or another validated intervention. SAM-WM is the intelligence layer that measures the current
thermal state, predicts candidate intervention outcomes, quantifies uncertainty, and recommends
an action. Real post-deployment observations then become new evidence for evaluation/learning.

No fabricated temperature, intervention effect, model result, activity ID, city geometry, or
customer result is inserted when evidence is missing.

## Learning design

**Primary learning = self-supervised predictive world modelling + supervised probabilistic
forecasting on real trajectories.** The model predicts future latent state and future temperature
from past state plus explicit intervention/action variables. Real intervention records provide the
action-conditioned signal. A separate matched intervention replay estimates measured real-world
effects where before/after/control data exist.

We do **not** use RL for the initial hackathon system: there is no trustworthy global action/reward
log that would justify it. The final action selector is a constrained planner over world-model
predictions and uncertainty. RL can be evaluated later only if a real logged decision dataset exists.

## Architecture

```text
real FortyGuard Temperature API + real urban/intervention sources
                              |
                              v
                    immutable evidence store
                              |
              +---------------+----------------+
              |                                |
              v                                v
   self-supervised/action-WM          real intervention replay
  spatial Transformer + temporal GRU       matched DiD / placebo
              |                                |
              +---------------+----------------+
                              v
                 SAM local action support
              + calibrated uncertainty sets
                              |
                              v
                constrained cooling planner
                              |
                              v
                    bounded agentic workflow
                              |
                              v
                real 3D observed / future UI
```

The agentic layer is deliberately bounded: the core planner and API calls are structured and
auditable. A small open-source LLM can later parse natural-language goals or explain outputs,
but it cannot invent measurements, select an unsupported physical action, or override the planner.

## FortyGuard API

The API is **not training code**. It is the real thermal-intelligence data/service layer used by
the product. The API key stays only in a local `.env` or deployment secret. It is never committed.

The client implements FortyGuard's asynchronous pattern:

```text
POST analysis -> activity_id -> GET /v1/status/{activity_id} -> Completed result
```

The current code integrates heatmap first and is structured for environmental parameters,
satellite/street-view segmentation, and Heat Intelligence to enrich the world state.

## Geographic scale

The architecture is city-provider based and contains no Phoenix-only learning assumption. The
hackathon's FortyGuard data coverage constrains the competition build to U.S. geographies. Any
U.S. city can be used through the same temperature client. A true worldwide deployment requires
additional validated temperature/intervention providers outside FortyGuard coverage and separate
cross-city/cross-country evaluation; the repository does not pretend Phoenix proves global
performance.

## Repository map

- `src/coolworld/fortyguard.py` — real API submit/poll client.
- `src/coolworld/evidence.py` — content-addressed evidence/provenance.
- `src/coolworld/ml/model.py` — action-conditioned JEPA-style predictive world model.
- `src/coolworld/ml/data.py` — real-only sequence dataset + manifest validation.
- `src/coolworld/ml/trainer.py` — reproducible PyTorch training.
- `src/coolworld/ml/evaluate.py` — forecast/uncertainty evaluation.
- `src/coolworld/ml/support.py` — local empirical action support.
- `src/coolworld/ml/calibration.py` — support-stratified conformal calibration.
- `src/coolworld/research/causal.py` — matched intervention replay estimator.
- `src/coolworld/planner.py` — uncertainty-bounded action ranking.
- `src/coolworld/agent.py` — bounded agentic orchestration.
- `src/coolworld/sam.py` — SAM support operator + exact linear reference QCQP.
- `static/index.html` — 3D product UI.
- `scripts/` — real data collection, training, evaluation, intervention replay.
- `configs/` — Hydra experiment configuration.
- `notebooks/KAGGLE_TRAIN_EVAL.md` — exact Kaggle workflow.
- `paper/` — research protocol/checklist.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ml,research,dev]'
cp .env.example .env
```

Then put the real key in `.env` locally:

```text
FORTYGUARD_API_KEY=...
```

Run:

```bash
pytest -m 'not live' -q
ruff check src tests scripts
uvicorn coolworld.api:app --reload
```

Open `http://127.0.0.1:8000`.

## Collect a real heatmap

Create a real GeoJSON AOI file and run:

```bash
python scripts/collect_fortyguard.py \
  --aoi path/to/aoi.geojson \
  --date 2026-08-18 \
  --time 14:00 \
  --granularity 100
```

The raw completed response and provenance hash are stored under the git-ignored evidence directory.

## Kaggle

Kaggle is for GPU training/evaluation after the real dataset bundle and manifest exist. Follow
`notebooks/KAGGLE_TRAIN_EVAL.md`. Do not put the FortyGuard key into a notebook cell or GitHub.

## Current truth boundary

The repository now contains the full **implementation path** from evidence acquisition through
training/evaluation/planning/UI, but there is not yet a trained checkpoint or measured cooling result
inside the repository. Those must be produced from real observations. Until that happens, the app
must say `MODEL_NOT_READY` rather than show a made-up counterfactual.


## v0.4 — actual moving city renderer

The old diagnostic temperature-block view is retired. The live browser now uses pinned
MapLibre GL JS 6.1.0 + deck.gl 9.3.7 and renders:

- real vector-map geography and 3D building extrusion when building heights exist;
- real FortyGuard GeoJSON thermal polygons as a ground-hugging thermal field;
- playback of compatible recorded real heatmap frames from the immutable evidence store;
- smooth visual interpolation between recorded frames (explicitly labelled interpolation);
- baseline vs intervention maps with synchronized 3D cameras;
- future thermal frames produced from the actual PyTorch checkpoint;
- a proposed intervention footprint rendered separately from observed assets.

There are **no hard-coded cooling values** in the browser. `Predict cooling future` remains
blocked until a valid trained checkpoint, real context dataset and support calibration exist.

The model was also corrected so future rollouts receive known future time features and reapply
spatial interaction on every predicted step. It no longer evolves future state from actions alone.


### Model/context binding

Counterfactual inference is cryptographically bound to the same real tile grid used to build the model context dataset. Loading another city with coincidentally similar numeric tile IDs cannot reuse the checkpoint: the backend compares a SHA-256 grid signature built from tile IDs + geographic centroids and returns `MODEL_CONTEXT_GRID_MISMATCH` on mismatch. This prevents a Phoenix/San-José-style geometry mix-up from being visualized as a valid prediction.
