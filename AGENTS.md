# Engineering contract

## Scientific integrity

1. Never add a fake measurement, synthetic fallback, or hard-coded model result.
2. A failed external source must fail closed.
3. Never put `FORTYGUARD_API_KEY` in source, tests, logs, screenshots, URLs, or
   client-side JavaScript.
4. Every empirical artifact needs source, retrieval time, request identity, and
   SHA-256 content hash.
5. A user-facing number must be traceable to evidence or an explicitly named
   mathematical transformation of that evidence.
6. Do not call an approximation exact.
7. Do not state a causal/counterfactual effect unless the identification assumptions
   and evidence support it.

## Code quality

- Small typed modules; no monolithic notebook as source of truth.
- Prefer stdlib/native platform/existing dependencies before adding another package.
- Comments explain assumptions, invariants, provenance, numerical approximations,
  or non-obvious failure behavior—not obvious syntax.
- Configuration belongs in typed settings / Hydra config, not scattered constants.
- Core transformations are pure where practical and covered by deterministic tests.
- Network integration tests are marked `live` and never silently mocked into success.
- Optimize only after profiling; preserve correctness/provenance boundaries.

## Research workflow

- Claim -> test -> raw result -> evaluation artifact -> figure/table.
- No manual result transcription into the paper.
- Preserve negative findings and mechanism/data conflicts.
