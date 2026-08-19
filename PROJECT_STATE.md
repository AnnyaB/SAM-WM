# Project State — SAM-WM / CoolWorld-SAM

## Mission
Build a real-evidence, action-conditioned urban world model that helps planners choose physical heat-mitigation interventions under uncertainty. The software is an intelligence and decision layer; the actual cooling comes from deployed physical interventions.

## Scientific boundary
- Observed real world: real FortyGuard/city/open-data evidence only.
- Model prediction: explicitly labelled prediction.
- Real replay: only after a real intervention with valid before/after/control evidence.
- Missing evidence -> unavailable/abstain, never fabricated output.

## Current method
A compact JEPA-style state encoder + spatial interaction + temporal memory + action/future conditioning predicts future tile temperatures. Action support and calibration gate counterfactual use. SAM research focuses on partial counterfactual identification when action support is uneven.

## Current state
- Frontend/browser shell works.
- Local unit/integration tests previously passed before this v0.6 research patch.
- Python 3.14 local install required PyArrow >=23,<26 rather than the earlier <22 bound.
- GitHub repository `AnnyaB/SAM-WM` was intentionally left empty until the project is ready for an initial clean push.
- Training/evaluation/calibration on real data are still pending.

## Next empirical milestone
Collect a small repeated real FortyGuard AOI timeline, build the immutable sequence bundle, run cheap baselines, then train/evaluate the compact world-model baseline on Kaggle.
