# Paper track

Working research question:

> How should an action-conditioned urban world model represent intervention outcomes when
> historical data provide uneven support for different cooling actions and contexts?

The paper is separate from the hackathon pitch. Product claims must be supported by live
system behavior; research claims require controlled experiments, real intervention replay,
coverage/calibration analysis, ablations, and comparison with strong baselines.

## Required experiment table before submission

- observational forecasting: persistence/seasonal baseline, MLP/LSTM, Transformer baseline,
  action-conditioned WM;
- world-model objective ablations: no latent predictive loss, no action conditioning, no
  spatial attention;
- support/uncertainty: ensemble-only, conformal-only, SAM support-aware calibration;
- real intervention replay: treated/control design, placebo dates/sites, pretrend diagnostics;
- geographic OOD: held-out neighborhoods/cities;
- intervention OOD if enough real interventions exist;
- latency and memory for the live product.

No SOTA, novelty, causal, global-generalization, or life-saving claim is written until the
corresponding evidence exists.
