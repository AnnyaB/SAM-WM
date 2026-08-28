# CoolWorld user guide

This guide answers the questions a first-time user or judge should be able to answer without reading the implementation:

1. **Where do I start?**
2. **What is the coloured mask?**
3. **What is SAM-WM doing?**
4. **Why did this tile become a hotspot?**
5. **What does the operational validation message mean?**
6. **What should a real engineer do next?**
7. **How do I restart or end the walkthrough?**

The primary UI includes a guided demo sequence. The recorded demo never needs to make a new FortyGuard request.

---

## 1. Start here

Open the app and use the guided sequence:

```text
START
  │
  ▼
1 · MEASURED CITY
  │
  ▼
2 · SAM-WM FORECAST
  │
  ▼
3 · PERSISTENT HEAT
  │
  ▼
4 · FIELD TEST
  │
  ▼
5 · MEASURE THE RESULT
```

Use **Back** and **Next** to move through the flow. Use **Restart** to return to the measured starting state. Finishing the walkthrough does not alter any data, model, threshold, or provider evidence.

### Measured city

The app loads the immutable recorded FortyGuard timeline. The demo bundle contains:

- 65 consecutive compatible hourly frames;
- one 36-tile San José provider grid;
- recorded timestamps from 2026-08-19 08:00 through 2026-08-22 00:00;
- provider activity/content provenance preserved in the tracked evidence artifacts.

The large left-hand 3D city is the primary view.

### Timeline / playback

The control below the 3D map is a **time navigator**, not a video recorder.

In measured mode:

- press `▶` to step through the 65 stored hourly provider fields;
- drag the slider to inspect a specific recorded hour;
- each stop is a stored provider observation.

In forecast mode:

- press `▶` to step through the six SAM-WM future states;
- drag the slider from `+1 h` to `+6 h`;
- fractional transitions are visual interpolation between hourly future states, not additional model observations.

### Forecast

The app runs the exact frozen promoted SAM-WM checkpoint on the real provider context:

```text
latest 48 recorded hourly fields
              │
              ▼
           SAM-WM
              │
              ▼
+1 h +2 h +3 h +4 h +5 h +6 h
```

The forecast is a model prediction, not a new observation.

### CITY MODEL · SAM-WM

The pulsing **CITY MODEL · SAM-WM** button in the top bar opens a separate model inspector without shrinking the normal 3D city view.

It shows:

```text
48 h measured city history
          │
          ▼
36 provider-grid tiles
          │
          ▼
local sparse city graph
          │
          ▼
routed thermal mechanisms
          │
          ▼
recurrent +1…+6 h rollout
```

The inspector explains the deployed mechanism families:

- **conservative exchange** — antisymmetric heat exchange between neighbouring tiles;
- **bounded source/sink** — constrained unresolved local thermal forcing;
- **bounded residual** — limited learned correction;
- **wind transport** — conservative upwind transport only when wind is available, disabled for this recorded rollout when no wind field is supplied;
- **daily + seasonal clock** — explicit diurnal and annual time features;
- **recurrent city memory** — each forecast state conditions the next hour.

Close the inspector to return to the 3D city and continue the guide.

### Prioritize

CoolWorld ranks forecast tiles and identifies the selected hottest fraction that remains hot across the six forecast horizons. This is a planning aid: it tells an engineer where to investigate first.

### Real-world action

CoolWorld does not claim that software physically cools a location. A city or site team must choose and implement a physical intervention, then measure its effect.

---

## 2. What the coloured mask means

The main map has two different meanings depending on mode.

### Measured thermal field

Every coloured polygon is **one provider tile inside the recorded FortyGuard AOI**.

```text
coloured polygon = evidence exists for this provider tile
uncoloured city   = outside this recorded evidence bundle
```

Uncoloured areas are **not automatically cooler**. They are simply outside the 36-tile demo field.

The thermal legend is dynamically stretched over the actual loaded temperature range to make spatial contrast visible. Therefore:

> **Red means the warm end of the currently loaded field range. It is not an automatic health-danger threshold.**

The legend remains in true degrees Celsius.

### SAM-WM forecast field

The same real 36-tile geometry is coloured using SAM-WM's predicted temperature for the selected future hour. The data have changed from observation to model prediction; the geometry has not been invented.

### Future hotspot-priority map

The dedicated hotspot map is different again. Yellow → orange → red is a **relative priority ranking among the selected forecast hotspots**, not a temperature scale.

The priority map always keeps true current and predicted °C in the hover/cards.

---

## 3. How to read the lower dashboard

### City Field Distribution

A histogram of the 36 displayed tile temperatures at the selected hour.

- horizontal position = temperature in °C;
- bar height = number of tiles in that temperature bin.

Use it to see whether the field is tightly clustered or has a warmer/cooler tail. It is not a time-series chart.

### Selected Hour

Summary statistics for the field currently visible on the map:

- mean;
- P95;
- maximum;
- minimum;
- tile count.

Move the timeline and these values update with the selected measured or forecast hour.

### Forecast Range + Validation

Shows the forecast horizon, calibrated prediction-band radius and recorded field-replay coverage.

The prediction band is an uncertainty interval around the forecast. Replay coverage is the fraction of recorded replay targets that landed inside that interval. It evaluates the model forecast; it is not a FortyGuard API health indicator.

### 6-Hour Priority-Zone Outlook

Shows the mean temperature trajectory across the currently prioritized future-hotspot locations from `+1 h` to `+6 h`.

It is intentionally a priority-zone trajectory, not the average of the entire city.

---

## 4. Why is a hotspot shown in that place?

`GET /api/hotspots` uses the frozen SAM-WM forecast only.

For each tile:

1. collect its six forecast temperatures (+1…+6 h);
2. compute its mean future temperature;
3. rank all tiles by that mean;
4. keep the requested hottest fraction (20% by default);
5. for each of the six horizons, compute which tiles are in the same hottest fraction;
6. report **persistence** = fraction of horizons in which the tile stays in that top zone.

So a high-priority tile is not selected because its polygon looks red. It is selected because **SAM-WM predicts it to remain relatively hot across the short future rollout**.

The hotspot card reports:

- current observed °C;
- +6 h predicted °C;
- maximum predicted °C over the six horizons;
- conformal uncertainty radius;
- persistence in the selected top-temperature fraction.

The UI also includes a **Why this location?** explanation below each card.

---

## 5. What SAM-WM is doing

SAM-WM is not an LLM chat layer and not a static single-frame classifier. The frozen experiment uses a compact mechanism-structured predictive world model.

### Temporal state

SAM-WM receives 48 hours of real city context. It therefore conditions on recent thermal evolution rather than only the latest field.

### Sparse physical graph

The city field is represented as nodes connected through a sparse neighbourhood graph. This gives the model explicit local spatial interaction structure.

### Mechanism composition

At each rollout step the model combines constrained mechanism families:

- **conservative exchange** for local pairwise thermal exchange;
- **optional wind transport**, exactly disabled if wind is unavailable;
- **bounded source/sink forcing** for local unresolved forcing;
- **bounded residual** for limited flexible correction;
- **state-dependent routing** to combine the available mechanisms.

### Recurrent world rollout

A latent recurrent state is rolled forward repeatedly to generate +1…+6 h future fields. The output is therefore a multi-step world rollout, not six unrelated static predictions.

### Uncertainty and abstention

The frozen Freiburg validation calibration supplies a conformal radius. CoolWorld exposes that uncertainty and keeps separate gates for:

- forecasting;
- operational provider validation;
- causal intervention evidence.

This is one of the important differences from a UI that simply shows a forecast and treats it as certain or actionable.

---

## 6. What the field-validation result means

It does **not** mean the FortyGuard API failed.

The API successfully produced the 65 compatible recorded real frames used by the demo.

The separate provider-replay protocol asks a deployment question: does the frozen uncertainty calibration meet the pre-set transfer gate on recorded FortyGuard fields?

Measured result:

```text
MAE / conformal radius    0.637548   required <= 1.0   PASS
empirical coverage        79.899691%
pre-set minimum coverage  80.000000%
```

The coverage gap is about **0.1003 percentage points**. The threshold was fixed before evaluation and was not lowered afterward.

---

## 7. How this can support real urban cooling

CoolWorld is the sensing/prediction/prioritization layer of a physical engineering loop:

```text
FortyGuard / sensors
observe thermal field
        │
        ▼
SAM-WM
forecast short-horizon evolution
        │
        ▼
CoolWorld
rank persistent future hotspots
        │
        ▼
Engineer / city operator
inspect constraints and select feasible action
        │
        ▼
Physical intervention
canopy / shade / materials / controls / other engineering
        │
        ▼
Treated + matched control measurement
        │
        ▼
Causal validation
keep / modify / reject the action
```

A forecast is useful because it can help decide **where and when to investigate** before resources are deployed. It does not, by itself, prove how much a tree, shade structure, reflective surface, or other intervention will cool that site.

### What should happen after a hotspot appears?

1. **Inspect the site** — land use, vulnerable users, geometry, existing shade, materials, ownership, safety, maintenance, and feasibility.
2. **Choose a candidate intervention** based on physical and operational constraints.
3. **Define treatment and control areas before implementation.**
4. **Measure comparable pre/post conditions.**
5. **Estimate the causal effect.**
6. Only then promote a numerical intervention effect into decision support.

---

## 8. Recorded mode vs optional live FortyGuard mode

The public demo is designed to be deterministic and credit-safe.

### Recorded-evidence mode — default

```bash
COOLWORLD_LIVE_API_ENABLED=0
```

This uses the tracked immutable provider evidence and makes no new provider request.

### Optional live provider mode

A live request requires both server-side controls:

```bash
FORTYGUARD_API_KEY=...
COOLWORLD_LIVE_API_ENABLED=1
```

The API key is never stored in browser JavaScript or committed to GitHub.

For a public hackathon demo, recorded-evidence mode is preferable because judges can reproduce the same evidence without consuming provider credits.

---

## 9. End, restart, and repeat

- **Start** begins at the measured city.
- **Next** advances through forecast, persistent heat, field-test design and measurement.
- **Back** revisits the prior stage.
- **Restart** returns to the measured starting point.
- The timeline play/pause control only animates the current measured or forecast sequence.

No guided-navigation control changes SAM-WM weights, recorded provider evidence, calibration, operational thresholds, or causal-effect availability.

---

## 10. Scaling beyond the hackathon demo

The current repository is a production-hardened reference/hackathon runtime, not a finished municipal control platform.

For multi-city production, the intended scaling path is:

- object storage for immutable evidence and large provider payloads;
- geospatial PostgreSQL/TimescaleDB for indexed city/time queries;
- Redis or another shared content-addressed inference cache;
- queue-backed provider and inference jobs;
- horizontally scaled stateless API workers;
- a model registry with immutable releases, canarying, and rollback;
- managed secrets, RBAC, audit logs, rate limiting, and per-tenant quotas;
- telemetry, calibration/drift monitoring, and alerting;
- human approval before any physical intervention recommendation is operationalized.

---

## 11. How to verify the demo without Chrome developer tools

Safari is sufficient. You do not need browser developer tools for the core verification.

Run:

```bash
make verify-demo
python verify_runtime.py
```

Then start the application with live provider calls disabled:

```bash
export COOLWORLD_LIVE_API_ENABLED=0
unset FORTYGUARD_API_KEY
uvicorn coolworld.app:app --host 127.0.0.1 --port 7860
```

The public-safe verification must leave `live_provider_api_enabled` false. The immutable runtime verifier itself makes zero FortyGuard network calls.

---

## 12. Claim boundary

### Supported by the frozen artifacts

- real recorded FortyGuard evidence integration;
- multi-step +1…+6 h SAM-WM urban thermal forecasting;
- Freiburg final-ID benchmark;
- Novi Sad zero-shot OOD-1;
- preregistered Turku/FAIRUrbTemp zero-shot OOD-2;
- uncertainty-aware forecasting;
- future-hotspot prioritization as non-causal planning support;
- evidence-triggered abstention.

### Not established by the frozen artifacts

- human-child-level general intelligence;
- AGI/ASI;
- universal SOTA superiority;
- guaranteed/measured physical cooling from a proposed action without treated/control evidence;
- operational certification for this exact replay result;
- planetary-scale validation or a claim that software itself cools Earth.
