# FortyGuard API integration audit — 2026-08-19

## Official contract that SAM-WM follows

- API key in the `api-key` header.
- asynchronous POST submission -> `activity_id` -> `GET /v1/status/{activity_id}` polling.
- heatmap granularity 60/80/100 m.
- TCM vs `time_of_measure` / `exceedance` / `persistence` are semantically different layers and must not be mixed.
- Heat Intelligence completion may return a streamed PDF and is persisted as immutable binary evidence.

## Current SAM-WM client coverage

Implemented in `FortyGuardClient`:

- `/v1/heatmap`
- `/v1/env_params`
- `/v1/satellite`
- `/v1/streetview`
- `/v1/heat_intelligence`
- unified `/v1/status/{activity_id}` polling

The app exposes heatmap/environmental/satellite/street-view routes. v0.6.1 adds the missing Heat Intelligence API route.

## Temperature Property

FortyGuard's public documentation states that the Enterprise API has six POST analysis endpoints and names Temperature Property as a Premium capability, but the currently retrievable Temperature Property page does not expose a usable request schema. SAM-WM deliberately does **not** invent this request contract. Add the sixth integration only from the exact official schema/recording.

## Do not call every endpoint for every heatmap

Endpoint use is task-driven to avoid credit waste and semantic errors:

- Heatmap: thermal state / baseline field.
- Environmental Parameters: heat-stress/environment context when required.
- Satellite + Street View segmentation: real urban morphology context / intervention feasibility.
- Heat Intelligence: evidence-backed planning/report context.
- Temperature Property: property-level workflow once its exact official contract is confirmed.

Using more endpoints is not automatically better. The organizer webinar specifically warns that the wrong analysis layer can produce a confident wrong answer.
