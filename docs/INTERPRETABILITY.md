# SAM-WM / CoolWorld interpretability and decision guide

This document answers five practical questions for a first-time user, engineer, reviewer, or city stakeholder:

1. **What am I looking at?**
2. **What is SAM-WM doing?**
3. **Why did the system highlight this place?**
4. **How much should I trust the result?**
5. **What should happen next in the physical world?**

CoolWorld is an **evidence-bounded urban thermal decision-support system**. It does not physically cool a street, building, city, or the planet. Physical systems and interventions perform cooling. CoolWorld converts measured temperature evidence into short-horizon forecasts and future-hotspot priorities, then keeps numerical intervention claims locked until independent treated/control evidence supports them.

---

## 1. Truth states: know what kind of information you are seeing

The UI deliberately separates four states.

### OBSERVED / REAL EVIDENCE

Source: recorded or explicitly requested FortyGuard TCM thermal evidence.

Meaning: this is a provider measurement/estimate attached to a real timestamp, grid geometry, provider activity identifier, content hash, and grid signature.

Use it for: understanding the measured thermal field and the recent temporal context.

Do **not** interpret it as: a future forecast or proof that an intervention caused cooling.

### SAM-WM RESEARCH FORECAST

Source: the exact promoted/frozen SAM-WM checkpoint applied to the verified recent real context.

Meaning: model prediction of the same grid at +1 through +6 hours.

Use it for: identifying where heat is likely to persist or intensify and where a human engineering review should focus first.

Do **not** interpret it as: an observation, an operational guarantee, or a causal intervention effect.

### OPERATIONAL CERTIFICATION

Source: the fixed FortyGuard provider-replay gate.

Meaning: an independent compatibility check asks whether the frozen model meets predeclared deployment-domain criteria. The current evidence narrowly misses the fixed empirical-coverage requirement, so operational certification remains false.

Use it for: deciding whether the forecast may be used as an operationally certified service. At present, the answer is **no**; the forecast remains research decision support.

### CAUSAL ACTION EFFECT

Source: independent treated/control intervention evidence through the CANDRA evidence contract.

Meaning: a numerical statement such as “this intervention reduced temperature by X °C” is allowed only when the relevant action has supported causal evidence in the requested coverage/horizon regime.

Current public runtime: **not available**. CoolWorld therefore does not fabricate values such as `-2.31 °C` for shade or trees.

---

## 2. What SAM-WM is doing internally

SAM-WM represents the city grid as a **sparse physical graph**. Nodes are observed locations/tiles and edges encode local spatial relationships. It rolls a compact latent state forward over time and composes four typed thermal mechanisms at every forecast step.

### Mechanism A — conservative exchange

Purpose: model local heat exchange between connected locations.

Engineering constraint: pairwise flux is antisymmetric and bounded by a discrete maximum-principle safeguard. The model exposes an exchange-conservation error diagnostic.

Interpretation: this branch asks whether local spatial differences can exchange heat without creating arbitrary net heat through the exchange operator.

### Mechanism B — wind transport

Purpose: represent directional thermal transport when wind is actually observed.

Engineering constraint: conservative upwind transport. If future wind is unavailable, this mechanism is exactly disabled rather than supplied with invented wind.

Current FortyGuard real-context deployment path: wind is unavailable, so the wind mechanism is disabled for that forecast.

### Mechanism C — bounded source / sink

Purpose: represent unresolved local forcing that cannot be explained by pair exchange alone.

Engineering constraint: the step magnitude is bounded by the frozen training-derived source limit.

Interpretation: this can absorb effects such as unobserved local heating/cooling forcing, but it must not be relabelled as a known physical cause.

### Mechanism D — bounded residual

Purpose: provide limited flexibility for dynamics not captured by the typed operators.

Engineering constraint: deliberately bounded residual capacity.

Interpretation: it improves predictive flexibility while preventing the residual branch from becoming an unlimited catch-all.

### Adaptive mechanism router

For every node and future step, SAM-WM produces a four-way mechanism weight vector. These weights are available in the model output as `mechanism_weights`.

They describe **how the predictive model allocates weight among its four internal mechanisms**. They are useful model diagnostics, but they are not causal scientific attribution. A high source/sink weight does not prove that a particular physical source caused the temperature.

### Recurrent latent state

A GRU-based latent state carries temporal information across the rollout. The model updates this state after each predicted thermal step and re-runs sparse message passing over the city graph.

### Uncertainty / surprise

SAM-WM predicts a scale parameter and the deployment bundle carries a frozen split-conformal radius. The UI exposes the conformal interval rather than presenting a point forecast as certain.

---

## 3. Why a tile is highlighted as a future hotspot

The hotspot planner does **not** use a hand-written red area or a mock ranking.

For each real grid tile it:

1. runs the frozen +1…+6 h SAM-WM forecast;
2. computes the tile's future temperature trajectory;
3. ranks tiles by mean predicted future heat;
4. checks how often the tile remains in the selected hottest fraction across the six forecast horizons;
5. reports current observed °C, +6 h forecast °C, future maximum, persistence, location, and the frozen uncertainty radius.

The yellow → orange → red priority layer is **relative priority within the forecast field**. It is intentionally separate from the absolute-temperature legend.

A red priority tile therefore means:

> “This tile is among the forecast-persistent hotter locations in the current planning window.”

It does **not** mean:

> “This tile is guaranteed to be dangerous,” “this intervention will cool it by a known amount,” or “the model discovered a causal mechanism.”

---

## 4. How to read the main results

### Current mean / P95 / maximum / minimum

These summarize the loaded observed field.

- **Mean**: average temperature across the current grid.
- **P95**: a high-end field statistic; useful for seeing whether a small hot tail exists even when the mean is moderate.
- **Maximum / minimum**: extrema among loaded tiles.

Action: use these to understand the current field, not to choose an intervention alone.

### +1…+6 h forecast trajectory

This is the model's expected thermal evolution over the next six hourly horizons.

Action: focus on **persistent** hotspots rather than reacting only to a single hottest instant.

### Conformal radius / interval

This is the frozen calibration envelope applied to the forecast.

Action: if two candidate locations differ by much less than the uncertainty scale, avoid overinterpreting their ordering; treat them as practically similar priorities until more evidence is available.

### Operational replay status

The provider replay is a deployment-domain compatibility gate, not a model-training score.

Current result: empirical coverage is `79.899691%` against the fixed `80.000000%` minimum, while the replay MAE/conformal-radius criterion passes. Because the coverage criterion fails, the system keeps `operational_certified=false`.

Action: use the current San José output as **research decision support only**. Do not silently lower the threshold after seeing the result.

### Benchmark evidence

The Freiburg final, Novi Sad zero-shot OOD, and preregistered Turku/FAIRUrbTemp zero-shot OOD results answer a different question: whether the frozen research model can forecast real urban temperature fields and transfer across cities without target fine-tuning.

Action: use these results to assess the research model's generalization evidence, not as proof of causal cooling or planetary-scale validity.

---

## 5. What a real engineer should do after the UI highlights a hotspot

The product is designed around a closed physical loop:

```text
MEASURE
  FortyGuard / field observations
       ↓
UNDERSTAND + FORECAST
  SAM-WM + uncertainty
       ↓
PRIORITIZE
  persistent future hotspots
       ↓
ENGINEERING REVIEW
  inspect site constraints and candidate interventions
       ↓
IMPLEMENT PHYSICAL ACTION
  only after feasibility / safety / permissions
       ↓
MEASURE TREATED + CONTROL
  pre/post and comparable control evidence
       ↓
ESTIMATE EFFECT
  causal evidence contract
       ↓
KEEP / MODIFY / REJECT
  update future decisions
```

### Candidate intervention categories

The UI may suggest categories such as tree canopy, shade, or reflective surfaces as **engineering options to investigate**. It does not assert that one is best for a tile without evidence.

A real deployment should add site-specific constraints before action, including as relevant:

- ownership and permissions;
- pedestrian / traffic / emergency access;
- underground and overhead utilities;
- tree species, water demand, roots and maintenance;
- shade geometry and structural loads;
- material reflectance, glare and surface-use constraints;
- vulnerable populations and use patterns;
- cost, installation time and maintenance burden;
- weather, season and local microclimate;
- control-site comparability and measurement plan.

### Minimum evidence loop for claiming cooling

Before the UI may show an evidence-backed numerical intervention effect:

1. define the intervention and coverage precisely;
2. define treated and comparable control areas;
3. collect pre-intervention evidence;
4. implement the physical intervention;
5. collect post-intervention treated/control evidence;
6. estimate the effect with uncertainty;
7. validate transfer on independent evidence;
8. encode only the supported regime in the CANDRA action artifact;
9. keep the system fail-closed outside that support.

This is how CoolWorld can become part of an **actual urban-cooling engineering process** without confusing prediction with causation.

---

## 6. What “interpretable” means here

CoolWorld currently provides three levels of interpretability.

### Data interpretability

Every displayed field declares whether it is observed, visually interpolated, model-predicted, or intervention-evidence-derived. Provider activity/content identities and checkpoint/context hashes preserve provenance.

### Model-structure interpretability

SAM-WM uses named typed mechanisms with explicit bounded/conservative contracts and exposes per-step mechanism routing weights internally.

### Decision interpretability

The hotspot API explains why a tile was prioritized using forecast temperature and persistence, and the UI states what a user may and may not conclude from that ranking.

This is **not** a claim that the model has human consciousness, human-child intelligence, or causal understanding. Those claims are not established by the experiments.

---

## 7. Fail-closed states and what to do

| State | Meaning | Correct response |
|---|---|---|
| `MODEL_NOT_READY` | promoted model bundle is unavailable/invalid | verify checkpoint + calibration + evaluation hashes |
| `REAL_CONTEXT_REQUIRES_*` | insufficient consecutive real context | collect/attach valid consecutive provider frames; do not fabricate |
| `RESEARCH_FORECAST_READY_OPERATIONAL_CERTIFICATION_FAILED` | frozen forecast works, replay gate does not pass | research-only use; investigate transfer/calibration in a new version |
| `CANDRA_ACTION_EVIDENCE_REQUIRED` | no supported causal action effect | collect genuine treated/control intervention evidence |
| `REQUEST_GRID_DOES_NOT_MATCH_REAL_CONTEXT` | request geometry differs from verified context | stop; rebuild context for the requested grid |
| `LIVE_PROVIDER_API_DISABLED` | public-safe mode blocks new provider spend | use recorded evidence, or deliberately enable server-side live mode |

---

## 8. Public-demo versus scaled production

The hackathon runtime is a reproducible, fail-closed **production-hardened demo**, not a claim of planet-scale operational capacity.

For a larger city/enterprise deployment, keep the scientific contract but move mutable operational state into managed services:

- object storage for immutable provider/evidence blobs;
- PostgreSQL for job/provenance metadata;
- Redis or another shared content-addressed cache;
- queue/workers for asynchronous provider jobs;
- horizontally replicated stateless API workers;
- dedicated batched model-inference service as demand grows;
- authentication/RBAC for operational actions;
- request quotas and rate limiting;
- managed secrets;
- OpenTelemetry/metrics/logging and alerting;
- model registry and signed promotion manifests;
- canary/rollback deployment;
- continuous drift/calibration monitoring;
- explicit human approval for physical-action recommendations.

The current code is separated into provider ingestion, model inference, evidence validation, product API, and UI layers so those replacements can be made without rewriting the research model.

---

## 9. The defensible one-sentence explanation

> **CoolWorld uses real FortyGuard thermal evidence and a frozen, uncertainty-aware SAM-WM world model to forecast short-horizon urban heat, explain which locations remain future hotspots, and guide engineers toward places worth investigating for physical cooling—while refusing to invent operational or causal claims that the evidence does not support.**

That is the intended scientific, product, and engineering interpretation of the current system.
