# Support-Aware Action-Conditioned World Models for Real Urban Cooling

## Abstract
TODO after experiments. Do not insert numerical results before the corresponding immutable output artifact exists.

## 1. Introduction
Urban heat mitigation is an action-selection problem under partial observability, non-stationary weather and uneven historical intervention support. Temperature mapping alone cannot answer what physical intervention should be deployed or whether the predicted effect is supported by evidence.

## 2. Problem formulation
Define real spatiotemporal observations, action fields, exogenous variables, prediction horizon and intervention queries. Separate passive forecasting from causal/action-conditioned prediction.

## 3. Method
### 3.1 Evidence-aware world state
### 3.2 Compact JEPA-style predictive representation
### 3.3 Spatial-temporal transition model
### 3.4 Action-conditioned imagined rollouts
### 3.5 Support-aware uncertainty and SAM partial identification
### 3.6 Abstention and conflict detection

## 4. Data and evidence protocol
Document FortyGuard requests, hashes, AOIs, timestamps, city/open-data intervention records and all exclusions.

## 5. Experiments
Use `EXPERIMENT_MATRIX.md`. Include cheap baselines before sophisticated models.

## 6. Results
TODO. No invented tables.

## 7. Discussion
Distinguish temperature forecast skill from causal intervention validity. Discuss surface/air/radiant-temperature semantics separately.

## 8. Limitations
Current U.S. hackathon data coverage, intervention availability, distribution shift, uncertainty calibration and external validity.

## 9. Deployment
Planner-facing digital twin that exports a proposed action footprint and evidence manifest; it does not claim to physically cool a city until the corresponding intervention is deployed.
