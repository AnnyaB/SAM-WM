# SAM-WM research positioning

## Current evidence status

SAM-WM is an original **mechanism-structured world-model hypothesis** for continuous urban thermal fields. It is not a renamed LeWM model and it does not claim a new physical law. Its original contribution is the implemented factorization of sparse graph state, conservative pair exchange, optional conservative wind transport, bounded source/sink forcing, bounded residual correction, adaptive mechanism routing, recurrent latent rollout and source-frozen uncertainty calibration.

The final matched paper suite completed 40/40 fits. All models were trained and selected only on Freiburg; Novi Sad was evaluated zero-shot without target fine-tuning or recalibration.

| Model | Freiburg MAE | Novi Sad MAE | Novi coverage |
|---|---:|---:|---:|
| **SAM-WM** | 1.4515 ± 0.0149 | **1.4675 ± 0.0256** | **89.59% ± 0.99%** |
| iTransformer-adapted | 1.6560 ± 0.0758 | 4.1367 ± 0.6737 | 50.71% ± 7.42% |
| TimeMixer-adapted | **1.4424 ± 0.0326** | 2.7799 ± 0.7516 | 63.73% ± 17.47% |

The strongest supported finding is therefore **cross-city preservation**, not universal source-domain superiority. TimeMixer-adapted slightly wins Freiburg MAE, while SAM-WM preserves both error and source-frozen calibration much better on zero-shot Novi Sad under the matched protocol.

## What is theoretically structured

For city graph `G=(V,E)`, node temperature `T_i^t` and latent state `z_i^t`, the implemented rollout is

`T_i^(t+1) = T_i^t + Δ_i^ex + Δ_i^wind + Δ_i^src + Δ_i^res`.

For the exchange operator, each physical edge uses a non-negative symmetric learned conductance and antisymmetric pair flux `F_ij = k_ij (T_j - T_i)`. The edge contribution added to one endpoint is subtracted from the other, so the **exchange term alone** has zero global sum up to floating-point error. Wind transport is also implemented conservatively and is exactly disabled when future wind is unavailable.

Those are structural invariants of the operators. They do **not** imply that the complete model conserves global energy: learned source/sink and residual terms are intentionally allowed to add or remove local forcing. SAM-WM is therefore mechanism-structured and physics-inspired, not a full first-principles urban surface-energy-balance model.

## What the ablations support

- `−SIGReg`: Freiburg fit improves substantially, but Novi Sad MAE and coverage worsen. This is consistent with a source-fit/transfer regularization trade-off.
- `−mental map`: both Freiburg and Novi Sad MAE worsen, supporting sparse state-dependent message passing in this setting.
- `−residual`: Novi Sad worsens more than Freiburg, supporting bounded free correction capacity for transfer.
- `−exchange`: aggregate MAE is essentially tied. The operator's current evidence is its conservation invariant, not a large forecasting gain.
- `−RH`: Freiburg worsens while Novi Sad improves slightly because Novi Sad has no RH. This exposes a real missing-modality shift that future training should address.

## Attribution boundary

SIGReg is not original to SAM-WM. **Sketched Isotropic Gaussian Regularization (SIGReg)** was introduced by LeJEPA (Balestriero and LeCun, 2025). SAM-WM uses an attributed adaptation of that regularization idea; the sparse continuous-field mechanism architecture, routing and evidence-bounded deployment logic are separate contributions.

The benchmark iTransformer and TimeMixer code is also explicitly labelled `*-adapted`: these are matched independent task implementations inspired by the published architectures, not official-code reproductions. The repository therefore does not claim official SOTA superiority.

## Remaining conference-level work

A serious paper submission still needs official-code external baselines, simple persistence/trend baselines, paired statistical testing, matched Turku evaluation, systematic sensor/modality dropout, graph perturbation and temporal-gap stress tests, and targeted experiments designed specifically to test the conservative exchange invariant.

Causal intervention claims need a different evidence class entirely. Forecast accuracy cannot establish the temperature effect of a future tree, shade structure, albedo change or retrofit. The valid sequence remains `measure → forecast → prioritize → intervene → measure treated/control → validate`.

Peer-reviewed GraphCast, NeuralGCM and GenCast motivate stronger learned/physical hybrid dynamics and probabilistic forecasting; Dreamweaver, PlanDQ and Learning-to-Theorize/NEO motivate compositional, planning and explanatory world-model representations. These are future research directions, not capabilities claimed by the present implementation.
