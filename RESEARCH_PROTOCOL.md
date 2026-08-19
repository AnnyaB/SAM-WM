# Real-world research protocol

## Research question
When a world model is asked about a cooling intervention, can it distinguish effects that are empirically identified from effects that are merely plausible extrapolations, and can this distinction improve intervention planning?

## Retrospective real-intervention evidence
For each eligible Phoenix intervention:
1. retrieve official intervention geometry/status/date metadata;
2. define pre/post windows before inspecting outcomes;
3. retrieve FortyGuard observations at matched local times/seasons;
4. construct untreated comparisons using only pre-treatment covariates;
5. check common support and pre-trends;
6. estimate effect using a declared design such as matched difference-in-differences/event study;
7. quantify uncertainty at the correct independent unit;
8. run placebo dates/sites and sensitivity analyses;
9. store activity IDs, source references, hashes, matching configuration and commit.

No intervention without credible timing/control information is promoted to causal evidence.

## Primary outcome
Change in FortyGuard ambient-air thermal burden at 2 m. Persistence/exceedance and comfort metrics may be supporting outcomes when computed from valid inputs. Phoenix/ASU pavement-surface or radiant-temperature evidence remains separately labelled and is never silently converted into ambient-air effect.

## Predictive world model
The representation/dynamics model learns temporal-spatial structure from real observations. Bulk use of FortyGuard outputs for training is enabled only under applicable organizer/data-use terms. Until then, FortyGuard can be used for live inference/validation and compatible open real datasets can be used for pretraining.

## SAM
SAM estimates empirical support for the requested intervention direction and intersects data-consistent and validated mechanism-consistent sets. Contradictions are exposed, not averaged away. The exact linear solver verifies linear mathematics; nonlinear claims require separate calibration/coverage evidence.

## Generalization
Planned real-data tests: geographic holdout, heat-regime holdout, intervention-type holdout where enough interventions exist, temporal/horizon holdout, and cross-city validation when compatible real intervention data exist. A global claim is not made from Phoenix alone.
