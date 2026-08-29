# SAM-WM research positioning and evidence audit

This document separates **implemented contribution**, **supported empirical findings**, **structural guarantees**, and **open research questions**. Nothing here claims universal state of the art, AGI/ASI, causal urban cooling, or planetary-scale validation.

## 1. Research thesis

SAM-WM is a compact **mechanism-structured sparse world model** for multi-step urban thermal fields. The hypothesis is that a continuous-field world model can improve transfer and reliability by combining learned latent dynamics with explicit local structure and bounded operators:

1. sparse local interaction instead of dense all-pairs mixing;
2. antisymmetric pair exchange with exact global exchange conservation for the exchange term;
3. conservative wind transport, exactly disabled when wind is unavailable;
4. bounded unresolved source/sink forcing;
5. bounded free residual capacity;
6. recurrent latent state for multi-step rollout;
7. predictive scale/surprise and source-frozen split-conformal calibration;
8. fail-closed separation between forecast, operational certification, and causal intervention evidence.

The final matched paper suite trains/selects/calibrates on Freiburg only, evaluates Freiburg held-out, and transfers the same models zero-shot to Novi Sad. The matched suite contains five seeds each for SAM-WM, two adapted contemporary time-series baselines, and five mechanism/data ablations.

## 2. Structural properties

These statements follow from `src/coolworld/samwm.py`; they do not depend on one checkpoint.

### Exchange conservation

For edge `(i,j)`, SAM-WM forms

`f_ij = k_ij (T_j - T_i)`, with `k_ij >= 0`,

adds `+f_ij` to node `i`, and `-f_ij` to node `j`. Therefore the sum of the exchange update over all nodes is zero up to floating-point arithmetic.

### Conservative wind transport

The upwind transport term subtracts an edge flux at its source and adds the same flux at its destination, so the global transport update sums to zero. When observed wind is unavailable, the implementation returns zero exactly.

### Local maximum-principle bound for exchange

Conductance normalization enforces `sum_j k_ij <= eta`, with `0 < eta <= 0.5`. The exchange-only update is a convex combination of the current node and neighbours, so it remains inside their local value hull.

This guarantee applies **only to exchange**. The full model is not globally energy-conserving because source/residual terms can add or remove local forcing.

### Bounded unresolved forcing

The source head and residual head use `tanh` and fixed routing weights. Their combined one-step magnitude is bounded by the training-derived source scale. This limits free forcing but is not a claim of a complete urban surface-energy-balance model.

### Sparse execution

Message passing, exchange, and transport operate on a stored edge list, giving graph interaction cost `O(E)` rather than `O(N^2)`.

## 3. Matched paper-suite evidence

Machine-readable source: `results/paper_suite/`.

Protocol:

- Freiburg train only;
- Freiburg validation for checkpoint selection only;
- Freiburg validation for conformal radius only;
- Freiburg held-out test;
- Novi Sad zero-shot OOD;
- no target fine-tuning;
- no target recalibration;
- 48 h context, six-hour rollout;
- seeds `17, 29, 42, 73, 101`.

### Full models

| Model | Freiburg MAE (°C) | Novi Sad MAE (°C) | Freiburg→Novi gap | Novi coverage |
|---|---:|---:|---:|---:|
| **SAM-WM** | 1.4515 ± 0.0149 | **1.4675 ± 0.0256** | **+0.0159 (+1.10%)** | **89.59% ± 0.99%** |
| iTransformer-adapted | 1.6560 ± 0.0758 | 4.1367 ± 0.6737 | +2.4806 (+149.8%) | 50.71% ± 7.42% |
| TimeMixer-adapted | **1.4424 ± 0.0326** | 2.7799 ± 0.7516 | +1.3375 (+92.7%) | 63.73% ± 17.47% |

The central result is **cross-city preservation**, not universal source-domain superiority. TimeMixer-adapted is slightly better on Freiburg MAE, but both adapted baselines degrade strongly under zero-shot Novi Sad transfer while SAM-WM remains near its source-domain error.

Relative to the matched adapters on Novi Sad, SAM-WM has:

- **64.53% lower MAE** than iTransformer-adapted;
- **47.21% lower MAE** than TimeMixer-adapted.

The adapters are independent task implementations inspired by the published architectures. They are not the original authors' official code, so these results do not establish superiority to the official iTransformer or TimeMixer papers.

## 4. Ablation evidence

| Variant | Freiburg MAE | Novi Sad MAE | Interpretation |
|---|---:|---:|---|
| **Full SAM-WM** | 1.4515 | 1.4675 | reference |
| − SIGReg | **1.3022** | 1.5669 | source fit improves; OOD and OOD coverage worsen |
| − exchange | 1.4494 | 1.4715 | forecast MAE essentially tied |
| − mental map | 1.4807 | 1.4872 | worse ID and OOD |
| − residual | 1.4572 | 1.5008 | slightly worse ID, worse OOD |
| − RH | 1.5065 | **1.4434** | RH helps Freiburg; target missing-modality mismatch remains |

Supported interpretations:

- **SIGReg:** evidence is consistent with a regularization trade-off. Removing it improves Freiburg MAE by ~10.3%, but worsens Novi Sad MAE by ~6.8% and reduces Novi Sad empirical coverage from ~89.6% to ~85.5%.
- **Sparse mental map:** removing it worsens both domains, supporting its contribution to predictive performance in this setting.
- **Bounded residual:** removing it worsens Novi Sad by ~2.27%, supporting limited free correction capacity.
- **Exchange:** this benchmark does not show a meaningful MAE gain from the exchange operator. Its demonstrated property is structural conservation, not large predictive advantage. Targeted stress tests are still needed.
- **RH:** removing RH worsens Freiburg by ~3.79% but improves Novi Sad by ~1.64%. Novi Sad has no RH, so this result highlights a modality-shift problem rather than proving RH is unhelpful.

## 5. Efficiency

| Model | Params | Freiburg latency/window | Freiburg MAE |
|---|---:|---:|---:|
| SAM-WM | 117,705 | 0.525 ms | 1.4515 |
| iTransformer-adapted | 350,598 | **0.059 ms** | 1.6560 |
| TimeMixer-adapted | **58,527** | 0.761 ms | **1.4424** |

Measured on the Kaggle Tesla T4 evaluator. These numbers are implementation-specific rather than hardware-independent microbenchmarks.

SAM-WM is ~3× smaller than the iTransformer adapter, but not the smallest or fastest model. Its evidence is strongest on **OOD transfer + calibration at compact scale**.

## 6. Frozen deployment evidence remains separate

The hackathon deployment uses a promoted frozen checkpoint and recorded FortyGuard provider evidence. The earlier SAM-WM-only frozen benchmark included Freiburg, Novi Sad, and preregistered Turku. That v1 result remains immutable and separate from the new matched baseline suite.

The real FortyGuard provider replay uses 65 compatible hourly frames on a 36-tile San José grid. Replay coverage is **79.899691%** against the fixed **80.0%** minimum, so operational certification remains **FAIL**.

This failure is retained. The threshold is not lowered after evaluation. Research forecast availability therefore does not imply operational certification, and neither implies a causal cooling effect.

## 7. What is still missing for a serious conference submission

The new evidence materially strengthens SAM-WM, but a top-conference package still needs:

1. official-code or independently validated reproductions of key external baselines;
2. simple non-neural baselines such as persistence and trend under the exact same protocol;
3. additional graph/recurrent baselines to isolate which inductive bias matters;
4. explicit statistical tests / confidence intervals for paired seed differences;
5. stress tests for sensor dropout, graph perturbation, temporal gaps, and systematic modality loss;
6. a rerun of the matched suite on a second OOD city when FAIRUrbTemp access is available;
7. targeted experiments that test the exchange invariant under synthetic/physical perturbations rather than only aggregate MAE;
8. intervention-conditioned data or simulation before any causal cooling claim;
9. independent treated/control field validation for real intervention effects.

## 8. Relationship to LeWM / MLML-style world-model research

SAM-WM is complementary to compositional and theory-building world-model work rather than a claim to supersede it.

- **LeWM / SIGReg:** SAM-WM adapts and attributes SIGReg, but operates on sparse continuous urban thermal fields rather than raw-pixel JEPA dynamics.
- **Dreamweaver:** focuses on compositional visual concepts; SAM-WM focuses on typed local continuous-field mechanisms and uncertainty.
- **NEO / Learning-to-Theorize:** learns executable theories/programs; SAM-WM does not claim latent program induction.

A defensible positioning is:

> SAM-WM asks whether a compact continuous-field world model can combine learned latent dynamics with explicit sparse interaction structure, bounded mechanisms, uncertainty, and fail-closed evidence gates to transfer across cities without target fine-tuning.

The new matched suite provides positive evidence for that transfer question, while also exposing where the design remains weak.

## 9. Physical intervention claim boundary

The scientifically valid loop is:

`measure → forecast → prioritize → physically intervene → measure treated/control → validate`.

SAM-WM currently covers the first three stages. It does not prove the temperature effect of a future tree, shade structure, albedo change, or retrofit. CANDRA keeps those causal claims locked until intervention evidence exists.
