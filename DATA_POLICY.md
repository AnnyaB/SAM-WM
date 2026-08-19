# Real-Evidence Data Policy

## Non-negotiable rule

Empirical claims in CoolWorld-SAM may be produced only from real, attributable
observations or real intervention records.

Accepted evidence classes:

- `FORTYGUARD_LIVE`: response returned by a live FortyGuard activity.
- `FORTYGUARD_RECORDED_LIVE`: immutable local copy of a prior live response,
  retaining its real activity ID and content hash.
- `CITY_OPEN_DATA`: data retrieved from an official public city/government service.
- `REAL_INTERVENTION_STUDY`: observations/results from an attributable real-world
  intervention study.

There is intentionally **no simulator/synthetic evidence kind** in the codebase.

## Forbidden

- generated/fabricated temperatures;
- random fallback observations;
- invented GPS points or treatment sites represented as observed evidence;
- a synthetic dataset presented as validation;
- manually typed model metrics presented as experiment output;
- LLM-generated measurements;
- fake API responses;
- silently replacing a failed API call with sample data;
- fabricated customers, deployments, lives saved, or cooling effects;
- treating pavement surface-temperature reduction as equivalent to human heat
  exposure or ambient-air reduction without real evidence.

## Failure behavior

Missing evidence -> `DATA_UNAVAILABLE`.
Invalid provenance -> `INVALID_EVIDENCE`.
Mechanism/data incompatibility -> `CONFLICT`.
Weak intervention identification -> report a wider uncertainty set or abstain.

The UI must display these states; it must never hide them behind a plausible
number.
