# SAM-WM research positioning and conference-readiness audit

This document separates **implemented contribution**, **empirical evidence**, and **future research**. It is deliberately stricter than a hackathon pitch: nothing here establishes universal state of the art, AGI, planetary-scale cooling, or a causal cooling effect.

## 1. Research thesis

SAM-WM is a compact **mechanism-constrained sparse world model** for multistep urban thermal dynamics. Its implemented hypothesis is that a learned latent world model for a continuous physical field can retain structural properties that an unconstrained sequence model does not provide automatically:

1. sparse local interaction instead of all-to-all mixing;
2. antisymmetric pair exchange with exact global exchange conservation;
3. conservative wind transport, exactly disabled when wind is unavailable;
4. bounded unresolved forcing;
5. recurrent latent state for multistep rollout;
6. predictive scale/surprise plus source-calibrated conformal uncertainty;
7. fail-closed separation between forecasting, operational certification, and causal intervention evidence.

The frozen experiment trains on Freiburg, evaluates held-out Freiburg, transfers the same model zero-shot to Novi Sad and the preregistered FAIRUrbTemp city (Turku), then replays the selected checkpoint on recorded FortyGuard provider fields.

## 2. Structural properties of the implemented operator

These statements follow from `src/coolworld/samwm.py`; they do not depend on a particular checkpoint.

### Proposition 1 — exchange conservation

For edge `(i,j)`, SAM-WM forms

`f_ij = k_ij (T_j - T_i)`, with `k_ij >= 0`,

adds `+f_ij` to node `i`, and `-f_ij` to node `j`. Therefore the sum of the exchange update over all nodes is zero up to floating-point arithmetic. Learning changes the conductance but cannot break pairwise antisymmetry.

### Proposition 2 — conservative wind transport

The upwind transport term subtracts an edge flux at its source and adds the same flux at its destination, so the global transport update sums to zero. When observed wind is unavailable the implementation returns `zeros_like(temp)` exactly; it does not invent a hidden wind field.

### Proposition 3 — local discrete maximum principle for exchange

Conductance normalization enforces `sum_j k_ij <= eta`, with `0 < eta <= 0.5`. For the exchange-only update,

`T'_i = (1 - sum_j k_ij) T_i + sum_j k_ij T_j`.

The coefficients are non-negative and sum to one, so the updated value remains in the convex hull of the current node and its neighbours. **This guarantee is for the exchange operator only.** The full SAM-WM is not globally energy-conserving because source/residual forcing can add or remove local heat.

### Proposition 4 — bounded non-conservative forcing

Let `S > 0` be the training-derived one-step source bound, `rho in [0,1]` the residual fraction, and `r2,r3` the source/residual softmax routing weights. The heads use `tanh`, so

`|source_i| <= S r2`, and `|residual_i| <= rho S r3`.

Since `r2 + r3 <= 1`,

`|source_i| + |residual_i| <= S (r2 + rho r3) <= S`.

This bounds unresolved forcing per rollout step. It is **not** a claim that SAM-WM contains a complete first-principles urban surface-energy-balance model.

### Proposition 5 — sparse graph execution

Message passing, exchange, and transport operate on the stored edge list, giving graph interaction cost `O(E)` rather than `O(N^2)`.

## 3. Relationship to MLML world-model research

SAM-WM should be positioned as **complementary** to MLML's compositional/theory-building program, not as a claim that it supersedes Dreamweaver or NEO.

### Dreamweaver

Dreamweaver learns compositional static/dynamic concepts from video with Recurrent Block-Slot Units. Its paper identifies open directions including probabilistic uncertainty and harder/complex scenes, and discusses limits of longer-horizon prediction and object interaction in evaluated settings.

SAM-WM addresses a different setting:

| Question | Dreamweaver focus | SAM-WM implemented focus |
|---|---|---|
| Representation | object/concept composition from pixels | local continuous physical-field state |
| Dynamics | learned visual concept dynamics | typed sparse graph mechanisms |
| Uncertainty | probabilistic extension is an open direction | Laplace predictive scale + conformal calibration |
| Interaction | object-centric interactions | explicit local node interactions |
| Deployment evidence | visual research datasets | three urban datasets + recorded provider replay |
| Causal intervention effect | not its objective | intentionally unclaimed without treated/control evidence |

These are not head-to-head results and must not be presented as such.

### NEO / Learning-to-Theorize

NEO learns executable latent programs as a learned Language of Thought. SAM-WM does **not** learn symbolic theories or latent programs and must not claim to solve NEO's theory-induction problem. Its distinct research question is whether a small continuous-field world model can be structurally constrained, uncertainty-aware, sparse, transferable, and connected safely to physical intervention evidence.

A defensible positioning is:

> MLML work asks how world models can discover compositional concepts and executable theories. SAM-WM asks how a compact world model for a continuous physical field can expose typed mechanisms, structural invariants, uncertainty, cross-city transfer, and a fail-closed path from prediction to physical intervention evidence.

## 4. Evidence already present

All values below are machine-readable in `artifacts/summary.json` and `artifacts/deployment/fortyguard_replay.json`.

| Domain | Protocol | MAE (°C) | RMSE (°C) | Conformal coverage |
|---|---|---:|---:|---:|
| Freiburg | held-out ID | 1.4515 ± 0.0167 | 2.0483 ± 0.0081 | 90.45% ± 1.26% |
| Novi Sad | zero-shot OOD | 1.4675 ± 0.0286 | 2.1575 ± 0.0370 | 89.59% ± 1.11% |
| Turku | zero-shot OOD | 1.5549 ± 0.0425 | 2.1944 ± 0.0574 | 88.55% ± 1.93% |

The model has **117,705 parameters**. The frozen FortyGuard replay uses 65 compatible hourly frames, 48 hours of context, and a six-hour rollout. Replay interval coverage is **79.899691%** against the preregistered **80.0%** minimum, therefore operational certification remains **FAIL**.

That failure must not be erased by lowering the gate or tuning calibration on those same replay targets. A valid improvement needs a new protocol whose calibration choices are fixed without using its evaluation targets.

Publication figures are regenerated from the tracked JSON evidence with:

```bash
python scripts/plot_results.py
```

## 5. What is still missing for a serious ICLR/ICML-style submission

The current evidence is meaningful, but it is **not yet a complete top-conference package**.

1. **Matched baselines on identical splits:** last-value persistence, same-hour daily persistence, linear/trend, a learned monolithic recurrent baseline, and preferably a graph-only unconstrained neural model.
2. **Mechanism ablations:** remove/disable SIGReg, sparse mental-map updates, conservative exchange, bounded residual forcing, time features, and exogenous modalities one at a time.
3. **Five seeds for learned baselines/ablations**, with mean ± standard deviation and horizon-wise curves.
4. **Calibration analysis:** coverage and interval width by horizon/domain; calibration-vs-sharpness trade-off.
5. **Robustness:** missing RH, sensor dropout, graph perturbation, temporal gaps, and stronger domain shift.
6. **Efficiency comparisons:** parameters, training time, inference latency, and memory against learned baselines.
7. **Intervention validation** before any causal cooling claim: physics simulation, natural experiment, or real treated/control field deployment with held-out validation.

The frozen v1 protocol intentionally evaluates only the full SAM-WM. Baseline and ablation experiments therefore belong to a separately versioned research protocol; they must not retroactively rewrite the frozen result.

## 6. What “cooling Earth” can mean scientifically

Software does not physically remove heat from a city. The defensible physical loop is:

`measure -> forecast -> prioritize -> physically intervene -> measure treated/control -> validate`.

Trees, shade structures, reflective materials, water strategies, retrofits, or other physical interventions alter the urban energy balance. SAM-WM can help decide **where and when to investigate**; CANDRA prevents an observational forecast from being mislabeled as a causal cooling effect.

A later action-conditioned SAM-WM can become a genuine counterfactual intervention world model by representing treatment variables (for example canopy, shade, or albedo changes), coupling them to physically meaningful forcing terms, predicting `T(a)` versus `T(0)`, and validating predicted `Delta T` against independent interventions. Until that evidence exists, intervention cooling magnitudes remain unreported.

## 7. Related literature that the paper must engage

World models:

- Maes et al., **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**, 2026, arXiv:2603.19312.
- Baek et al., **Dreamweaver: Learning Compositional World Models from Pixels**, ICLR 2025, arXiv:2501.14174.
- Baek et al., **Learning to Theorize the World from Observation**, ICML 2026 Oral, arXiv:2605.03413.

Urban cooling / intervention modeling:

- **A causal deep learning framework for simulating the thermal impact of urban land use changes**, *Sustainable Cities and Society* 148 (2026), 107657, DOI: 10.1016/j.scs.2026.107657.
- **Physics-informed machine learning for mapping the heat mitigation potential of vegetation in Singapore**, *Sustainable Cities and Society* 142 (2026), 107317, DOI: 10.1016/j.scs.2026.107317.
- **Machine learning-based design optimization of urban greening for cooling in tropical cities**, *Building and Environment* 295 (2026), 114468, DOI: 10.1016/j.buildenv.2026.114468.

These urban-heat works make the evidence standard explicit: moving from temperature prediction to intervention design requires treatment/design variables, physical or causal structure, and validation of counterfactual cooling outcomes.

## 8. Paper-level contribution statement to test, not assume

A future paper can test this statement:

> **SAM-WM is a compact sparse world model for continuous urban thermal fields that composes learned local latent dynamics with structurally constrained exchange/transport operators, bounded unresolved forcing, and calibrated uncertainty; we evaluate whether these constraints improve multistep accuracy, robustness, transfer, and efficiency relative to matched unconstrained baselines.**

The words **“improve relative to matched baselines”** become a result claim only after the missing experiments support them.
