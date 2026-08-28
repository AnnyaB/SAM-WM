# CoolWorld production architecture

CoolWorld is designed as a small, reproducible research-to-product runtime. The hackathon deployment is intentionally conservative: immutable evidence and one frozen checkpoint are packaged with the application, while new provider requests and causal intervention claims remain explicit opt-ins.

## Runtime layers

```text
browser / city user
       │
       ▼
FastAPI product + evidence API
       │
       ├── immutable recorded FortyGuard evidence
       ├── promoted SAM-WM checkpoint + calibration + final/OOD manifest
       ├── frozen real-context inference
       ├── non-causal hotspot prioritization
       └── optional feature-gated live provider client
```

The model is not embedded in browser JavaScript. Inference and provider credentials stay server-side.

## Truth-state separation

`/api/product-status` keeps independent deployment concepts independent:

- `real_provider_evidence_ready`
- `model_bundle_promoted`
- `research_forecast_ready`
- `operational_certified`
- `causal_action_ready`

A research forecast can be valid and reproducible even when a stricter operational transfer gate fails. A forecast can also be valid without establishing a causal intervention effect.

## Immutable demo evidence

The public demo packages:

- promoted seed-42 `best.pt`;
- deployment calibration/evaluation/promotion manifests;
- final/OOD summary;
- recorded FortyGuard evidence and hashes;
- frozen provider replay result.

`verify_runtime.py` checks the artifact relationships without making network calls.

## Provider security and cost control

A live request requires both:

```text
FORTYGUARD_API_KEY=<server-side secret>
COOLWORLD_LIVE_API_ENABLED=1
```

The public-safe default is `COOLWORLD_LIVE_API_ENABLED=0`. Therefore a leaked or accidentally configured key alone cannot make the public UI spend provider allocation.

Never place the key in:

- source code;
- committed `.env` files;
- frontend JavaScript;
- browser local storage;
- screenshots or demo logs.

## Current caching

`product_api.py` uses a one-entry process-local forecast cache keyed by immutable artifact identity and recent real-frame identity. This avoids repeatedly reloading the frozen model for identical hotspot requests in the single-worker demo.

This cache is intentionally small and non-authoritative. For multi-replica production, replace it with a content-addressed shared cache (for example Redis) and put immutable provider evidence/model artifacts in object storage.

## Horizontal scaling path

Before running multiple replicas:

1. move immutable evidence/model bundles to versioned object storage;
2. move mutable provider-ingestion state to a transactional datastore;
3. replace process-local inference cache with shared content-addressed cache;
4. use idempotency keys for provider activities;
5. enforce authenticated/rate-limited live provider operations;
6. add observability for latency, request errors, model failures, and abstentions;
7. keep model/evidence hashes in every prediction audit record.

The model itself is compact (117,705 parameters in the frozen evaluation), so inference compute is not the dominant scaling concern in this demo. Data provenance, state coordination, and safe provider access matter more.

## Reliability policy

The runtime should fail closed when any of these occur:

- checkpoint/calibration/evaluation hash mismatch;
- insufficient consecutive real context;
- provider grid change;
- non-finite model output;
- failed operational replay when `/api/forecast` is requested;
- missing independent action evidence when `/api/counterfactual` is requested;
- live provider endpoint disabled or key absent.

Research preview and hotspot prioritization remain explicitly non-actionable when operational certification is absent.

## Container contract

The Docker image:

- uses Python 3.11 slim;
- runs as non-root UID 10001;
- exposes port 7860 by default;
- includes only required runtime artifacts;
- has an HTTP healthcheck;
- defaults live provider calls off;
- uses one worker by default to keep the demo cache simple and deterministic.

For production behind a reverse proxy, preserve forwarded headers and terminate TLS upstream.

## Model evolution

Do not silently replace the frozen model behind an existing evidence bundle. A new architecture, objective, preprocessing rule, graph rule, QC rule, hyperparameter set, or checkpoint requires a new version and new untouched confirmatory evaluation.

That versioning discipline is more important than presenting a permanently green dashboard.
