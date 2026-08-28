# Hugging Face Docker Space deployment

CoolWorld is packaged for a Docker-based Hugging Face Space. The checked-in runtime contains the frozen model bundle and immutable recorded provider evidence required for the no-login demo.

## Recommended public configuration

- SDK: **Docker**
- Hardware: CPU is sufficient for the compact demo model; GPU is not required for the intended judge path.
- Port: `7860`
- Visibility: public for hackathon judging if the competition requires a no-login URL.
- Live provider calls: **disabled**.

Do not set `COOLWORLD_LIVE_API_ENABLED=1` on the public judge Space.

The normal public demo requires no FortyGuard secret because it replays the already-recorded, provenance-preserving evidence.

## Build contract

The repository Dockerfile:

```text
copies source + static UI
copies promoted deployment artifacts
copies immutable recorded FortyGuard evidence
installs the app
runs as a non-root user
serves FastAPI/Uvicorn on $PORT (default 7860)
health-checks /api/health
```

Local smoke test:

```bash
docker build -t sam-wm-coolworld .
docker run --rm -p 8000:7860 sam-wm-coolworld
```

Then verify:

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/product-status
curl -fsS 'http://127.0.0.1:8000/api/evidence/timeline?limit=1'
```

Open `http://127.0.0.1:8000` and test the complete Observe → Forecast → Prioritize → Evidence flow before publishing.

## Optional server-side secret

If a later private deployment genuinely needs new provider requests, configure the FortyGuard key as a **Space secret**, never a normal variable rendered into frontend code. A live call still requires the separate server-side feature flag:

```text
FORTYGUARD_API_KEY=<secret>
COOLWORLD_LIVE_API_ENABLED=1
```

This two-part control prevents an accidentally configured key from creating spend through the public UI.

## Public acceptance checklist

Before using the Space URL in a submission:

- open it in an incognito/private browser with no Hugging Face login;
- confirm the 3D map loads;
- confirm the recorded 65-frame evidence loads automatically;
- select `SAM-WM FORECAST` and confirm +1…+6 h model frames render;
- run `PRIORITIZE` and confirm the relative yellow→red future-hotspot view plus true °C cards render;
- confirm Evidence shows Freiburg, Novi Sad, Turku, and the FortyGuard replay FAIL truthfully;
- confirm no browser network request exposes `FORTYGUARD_API_KEY`;
- confirm the live provider button is disabled in the public-safe mode;
- confirm `/api/counterfactual` remains fail-closed without independent action evidence;
- confirm page reload works from a fresh session.

## Scaling later

A hackathon Space is not the final municipal deployment architecture. Before multi-replica or high-volume operation, move mutable state and shared caching out of the process as described in `docs/PRODUCTION.md`.
