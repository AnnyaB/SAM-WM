# Frozen research protocol

## Primary question

Can a compact world model with a sparse city mental map and a small typed mechanism library learn
short-horizon urban thermal dynamics on Freiburg and transfer without fine-tuning to independent
urban sensor networks?

## What the source code is designed to test

1. Forecast quality on a preregistered Freiburg held-out period.
2. Calibration of Freiburg validation-derived split-conformal uncertainty on final test and OOD.
3. Zero-shot transfer to Novi Sad.
4. Zero-shot transfer to one preregistered unseen FAIRUrbTemp city.
5. Sparse `O(E)` graph execution and exact conservation of the constrained exchange branch.
6. Whether surprise increases under distribution shift.
7. Whether structural ablations degrade validation performance in ways consistent with the proposed
   physical/temporal design.
8. Whether removing the predictive latent objective or SIGReg changes validation behavior.

## Pre-freeze research suite

Development/validation-only research is executed by `research.py`.

Frozen seeds:

```text
17, 29, 42, 73, 101
```

Structural ablations:

```text
no_mental_map
no_exchange
unconstrained_exchange
no_source_sink
no_residual
uniform_router
no_temporal_memory
```

Objective controls:

```text
no_sigreg
temperature_only
```

Sanity baselines:

```text
persistence
linear_trend
daily_persistence
```

None of these pre-freeze experiments may open Freiburg held-out, Novi Sad targets, or FAIRUrbTemp
targets.

## Claims this protocol cannot establish by itself

- universal SOTA;
- human-child equivalence;
- AGI or ASI;
- a causal cooling effect of shade, canopy, pavement, or any other intervention;
- planetary-scale Earth cooling;
- global deployment safety;
- industrial scalability without explicit load/scale measurements.

## Leakage rules

- Fit normalization only from Freiburg training.
- Derive the unresolved source bound only from Freiburg training observations.
- Use Freiburg validation only for development/model-selection evidence.
- Write and hash the complete pre-freeze evidence bundle before opening held-out/OOD targets.
- Open Freiburg held-out only after the source/config/checkpoint freeze.
- OOD targets are evaluation-only: no fine-tuning, recalibration, hyperparameter selection, or
  target-driven city choice.
- Choose the FAIRUrbTemp city using metadata/coverage criteria before viewing SAM-WM results.
- A material model/protocol change after the freeze defines a new version and requires a new
  untouched confirmatory benchmark for confirmatory claims.

## Causal boundary

CANDRA may only produce a non-zero intervention effect when a genuine intervention/control design
with justified assumptions is available. Ordinary observational forecasting data are not sufficient
to identify a causal cooling effect.

## Reporting boundary

Every reported number must be generated from saved machine-readable artifacts. The README, paper,
figures, UI, and video must not contain manually invented benchmark or intervention values.
