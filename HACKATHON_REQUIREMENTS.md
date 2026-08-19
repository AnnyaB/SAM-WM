# FortyGuard Hackathon'26 — frozen build requirements

This is the implementation contract for the hackathon branch. When the general public API documentation is broader than the participant handbook, this branch follows the stricter hackathon rule.

## Challenge
Use FortyGuard hyper-local temperature intelligence as a **central** component of something a real client would actually use to solve a genuine urban-heat problem. Build for commercialisation, not a throwaway demo.

## Chosen track
**Primary: Track 1 — Resilient Cities & Infrastructure.**

The target use case is the digital-twin workflow: test physical cooling interventions such as tree canopy or reflective paving before implementation.

## Hard constraints
- Build window: 18–30 August 2026 (GST / UTC+4).
- Deadline: 30 August 2026, 11:59 PM GST.
- Hackathon geography: U.S. coordinates only.
- Hackathon date floor: 2021-01-01.
- Heatmap forecast horizon: at most 12 hours ahead.
- API key stays server-side and is sent in the `api-key` header.
- FortyGuard is asynchronous: submit -> `activity_id` -> poll -> Completed/Failed.
- Participant access is Premium.

## Judging priorities
- Impact & Relevance: 40%
- Technical Execution: 35%
- Innovation: 15%
- Communication: 10%

Research sophistication must strengthen a measurable cooling decision rather than hide the product behind theory.

## Submission package
1. Live demo URL that judges can open without login/install.
2. <= 3 minute video showing the running product.
3. GitHub/GitLab code. Private is acceptable when the FortyGuard judge account is added as collaborator.
4. Submission form and required project metadata.

## Evidence behavior
No API failure, missing model, missing intervention evidence, or missing geometry is replaced by generated data. The product returns an explicit unavailable/abstain state.
