# Frozen research protocol

## Question

Can a compact world model with a sparse city mental map and a small typed mechanism library learn short-horizon urban thermal dynamics on Freiburg and transfer without fine-tuning to independent urban sensor networks?

## Claims that this protocol can test

1. Forecast quality on a preregistered Freiburg held-out period.
2. Calibration of Freiburg validation-derived split-conformal uncertainty on final test and OOD.
3. Zero-shot transfer to Novi Sad and selected FAIRUrbTemp city data.
4. O(E) graph operator execution and exact conservation of the exchange branch.
5. Whether model surprise increases under distribution shift.

## Claims this protocol cannot establish by itself

- universal SOTA;
- AGI or human-child equivalence;
- causal cooling effect of shade, canopy, pavement, or any other intervention;
- planetary-scale Earth cooling;
- global deployment safety.

## Leakage rules

- Fit temperature/RH/static normalization only from Freiburg training.
- Use Freiburg validation only for early stopping and conformal radius.
- Open Freiburg final test only after configuration is frozen.
- OOD targets are evaluation-only: no fine-tuning, recalibration, hyperparameter selection, or city-specific graph learning beyond geometry construction.
- A material model/protocol change after seeing final-test results defines a new version and requires a new untouched final benchmark for confirmatory claims.
