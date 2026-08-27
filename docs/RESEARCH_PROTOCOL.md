# Frozen SAM-WM research protocol

## Primary question

Can one compact sparse adaptive mechanism world model learn short-horizon urban thermal dynamics on
Freiburg and transfer zero-shot to independent urban sensor networks without target fine-tuning?

## Model under study

There is exactly one trained research model in this protocol: **SAM-WM**.

The benchmark pipeline does not train or score external baseline models, persistence models,
linear-trend models, or alternative model families. The scientific object being tested is the full
SAM-WM architecture defined in `src/coolworld/samwm.py`.

## What the frozen source tests

1. Freiburg development learning and validation-selected checkpointing.
2. Forecast quality on the preregistered Freiburg held-out period.
3. Freiburg-validation-derived split-conformal calibration on final test and OOD.
4. Zero-shot transfer to Novi Sad.
5. Zero-shot transfer to one preregistered unseen FAIRUrbTemp city.
6. Sparse `O(E)` graph execution and constrained exchange invariants.
7. Surprise, uncertainty, parameter count and inference latency under the same frozen model.
8. Deployment-domain replay on recorded real FortyGuard TCM evidence after benchmark promotion.

## Frozen development suite

`research.py` trains the same full SAM-WM architecture across five fixed seeds:

```text
17, 29, 42, 73, 101
```

Outputs:

```text
artifacts/research/seed_17/
artifacts/research/seed_29/
artifacts/research/seed_42/
artifacts/research/seed_73/
artifacts/research/seed_101/
artifacts/research/PRE_FREEZE_MANIFEST.json
```

Only Freiburg train/validation data may be accessed during this stage. The manifest records
`model: SAM-WM`, hashes the five validation artifacts, and certifies
`heldout_or_ood_accessed: false`.

## Leakage rules

- Fit normalization only from Freiburg training.
- Derive the unresolved source bound only from Freiburg training observations.
- Use Freiburg validation only for early stopping and deployment-seed preselection.
- Preselect the deployment seed before opening Freiburg held-out or either OOD target.
- Hash source, configuration, pre-freeze manifest and all five SAM-WM checkpoints before final/OOD
  evaluation.
- Open Freiburg held-out only after the freeze and only once per seed/output receipt.
- Novi Sad and FAIRUrbTemp are zero-shot evaluation only: no fine-tuning, recalibration,
  target-driven hyperparameter selection, or graph learning from target labels.
- Choose the FAIRUrbTemp city using metadata/coverage criteria before viewing SAM-WM metrics.
- A material architecture/objective/preprocessing/QC/split/graph/hyperparameter change after freeze
  defines a new research version.

## Evaluation order

```text
Freiburg train/validation
→ deployment-seed preselection from Freiburg validation only
→ cryptographic freeze
→ Freiburg held-out once
→ Novi Sad zero-shot
→ FAIRUrbTemp zero-shot
→ aggregate evidence
→ finalize the preselected SAM-WM deployment bundle
→ real FortyGuard provider replay gate
```

## Causal boundary

SAM-WM is a forecasting/world-model system. A temperature forecast does not identify the causal
effect of shade, canopy, pavement, water, or another intervention. CANDRA may enable a non-zero
action effect only when a genuine source treated/control design and an independent transfer
treated/control design are supplied and their uncertainty/support gates pass.

## Reporting boundary

Every benchmark, calibration, latency, surprise, or intervention number shown in the README, paper,
figures, UI, or video must be read from saved machine-readable evidence. No hand-entered benchmark or
cooling values are permitted.

## Claims this protocol cannot establish by itself

Passing the source/CI protocol does not prove universal SOTA, human-child equivalence, AGI/ASI,
planetary-scale cooling, global deployment safety, or guaranteed hackathon ranking. Those claims
require evidence beyond source-code correctness.
